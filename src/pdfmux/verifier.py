"""Certify Anything — audit ANY extraction engine's output against the source PDF.

This is pdfmux's audit layer turned into a standalone verifier. Point it at a
source PDF plus an *external* extraction (produced by Reducto, Mistral OCR,
LlamaParse, Docling, an in-house parser — anything) and it re-derives what the
source actually contains, then scores the extraction against it.

It answers the question every RAG/agent pipeline silently gets wrong:

    "The engine returned exit-code 0 and some JSON. Did it actually extract
     the document, or did it silently drop pages / hallucinate / mangle tables?"

The whole thing reuses pdfmux's existing audit signals — the same 5-check
per-page confidence score (:func:`pdfmux.audit.score_page`) and the same
source-of-truth per-page extraction (:func:`pdfmux.audit.audit_document`) that
power multi-pass extraction. We are not inventing a new judge; we are pointing
the judge we already own at somebody else's output.

Output is a **certification manifest**:

  * per-page confidence, coverage, alignment, hallucination-risk
  * detected silent drops (source page has text, extraction returned nothing)
  * table / heading integrity flags
  * an overall PASS / REVIEW / FAIL verdict
  * a SHA-256-anchored, timestamped, content-signed record

Honest limits (see module-level ``LIMITATIONS``): we can only verify what we
can cheaply re-derive from the source. Scanned pages with no digital text layer
are marked ``unverifiable`` rather than silently passed — we refuse to certify
what we ourselves cannot read.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pdfmux.audit import (
    EMPTY_TEXT_THRESHOLD,
    GOOD_TEXT_THRESHOLD,
    audit_document,
    score_page,
)

# ---------------------------------------------------------------------------
# Tunables (env-overridable knobs live in the CLI, not here)
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = "certify-anything/v1"

# Coverage = extracted_chars / source_chars. Below this, the extraction is
# missing a large chunk of the page's content even if it isn't fully empty.
LOW_COVERAGE_THRESHOLD = 0.60

# Alignment = fraction of source content-tokens present in the extraction.
# This catches garbled / reordered / partially-dropped text that a raw char
# count would miss.
LOW_ALIGNMENT_THRESHOLD = 0.55

# Hallucination-risk = fraction of extraction content-tokens with NO support
# in the source page. High risk on a substantial page = invented text.
HIGH_HALLUCINATION_THRESHOLD = 0.45

# Per-page confidence (reused pdfmux score_page) below which a page is flagged.
LOW_CONFIDENCE_THRESHOLD = 0.55

# A page only counts toward hallucination scoring once the extraction has this
# many content tokens — tiny fragments produce noisy ratios.
MIN_TOKENS_FOR_HALLUCINATION = 12

# When an extraction page slot is empty, we only declare a *silent drop* if the
# source page's content is also absent from the extraction as a WHOLE. If it
# turns up elsewhere (engine merged pages, or emitted an unsegmented blob), it
# was not dropped — just re-paginated. Above this alignment against the whole
# extraction, we treat the content as recovered.
SILENT_DROP_RECOVERY_THRESHOLD = 0.50

# Below this alignment against the whole extraction, a substantial source page
# is considered genuinely missing (a real silent drop) even in unsegmented mode.
MISSING_CONTENT_THRESHOLD = 0.15

LIMITATIONS = (
    "Cannot verify content it cannot re-derive from the source PDF. Pages with "
    "no digital text layer (pure scans) are marked 'unverifiable' — pdfmux's "
    "cheap re-derivation reads no text there, so silent-drop and alignment "
    "cannot be judged without running full OCR on the source.",
    "Requires the source PDF. A manifest cannot be produced from the extraction alone.",
    "Alignment is lexical (content-token overlap), not semantic. A faithful "
    "paraphrase or translation would score as low alignment. Extraction "
    "engines are expected to transcribe, not rewrite, so this is the correct "
    "conservative bias for a certifier.",
    "Adds latency: the source PDF is re-extracted once through pdfmux's fast "
    "pass to establish ground truth.",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageVerification:
    """Verification result for a single page of an external extraction."""

    page_num: int  # 0-indexed
    source_chars: int
    extracted_chars: int
    coverage: float  # extracted content vs source content (0.0–1.0, capped)
    confidence: float  # pdfmux score_page on the extracted text (0.0–1.0)
    alignment: float  # fraction of source content-tokens present in extraction
    hallucination_risk: float  # fraction of extraction tokens absent from source
    silent_drop: bool  # source has real text, extraction returned ~nothing
    table_integrity: bool  # source table structure preserved (True if no table)
    heading_integrity: bool  # source headings preserved (True if no heading)
    verdict: str  # "pass" | "review" | "fail" | "unverifiable"
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["page"] = self.page_num + 1  # human-friendly 1-indexed alias
        d["flags"] = list(self.flags)
        return d


@dataclass(frozen=True)
class CertificationManifest:
    """A signed, timestamped certification of an external extraction."""

    schema_version: str
    source: str
    source_sha256: str
    extraction_sha256: str
    engine: str
    tool: str  # "pdfmux verify vX.Y.Z"
    timestamp: str  # ISO-8601 UTC
    page_count: int
    page_aligned: bool  # did the extraction expose real per-page structure?
    verdict: str  # "PASS" | "REVIEW" | "FAIL"
    confidence: float  # content-weighted overall confidence
    coverage: float  # overall content coverage
    silent_drops: tuple[int, ...]  # 1-indexed page numbers
    low_confidence_pages: tuple[int, ...]  # 1-indexed
    review_pages: tuple[int, ...]  # 1-indexed
    unverifiable_pages: tuple[int, ...]  # 1-indexed
    summary: str
    pages: tuple[PageVerification, ...] = ()
    signature: str = ""  # content hash over the canonical manifest body

    # -- serialization -----------------------------------------------------

    def _body(self) -> dict[str, Any]:
        """Canonical manifest body (everything the signature covers)."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "extraction_sha256": self.extraction_sha256,
            "engine": self.engine,
            "tool": self.tool,
            "timestamp": self.timestamp,
            "page_count": self.page_count,
            "page_aligned": self.page_aligned,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "coverage": round(self.coverage, 4),
            "silent_drops": list(self.silent_drops),
            "low_confidence_pages": list(self.low_confidence_pages),
            "review_pages": list(self.review_pages),
            "unverifiable_pages": list(self.unverifiable_pages),
            "summary": self.summary,
            "pages": [p.to_dict() for p in self.pages],
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        body["signature"] = self.signature
        body["limitations"] = list(LIMITATIONS)
        return body

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        return _manifest_markdown(self)


def _compute_signature(body: dict[str, Any]) -> str:
    """Deterministic content signature over the canonical manifest body.

    The OSS core signs with a SHA-256 content hash: it makes the manifest
    tamper-evident (any edit changes the signature) and reproducible. The paid
    cloud tier replaces this with a real asymmetric signature (Ed25519) bound to
    a pdfmux signing key, which is what makes a manifest externally *verifiable*
    by a third party / regulator. The hashing surface is identical, so a
    cloud-signed manifest and an OSS-signed manifest cover the same bytes.
    """
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tokenization / integrity helpers
# ---------------------------------------------------------------------------

# Content tokens: alphanumeric runs of length >= 2, lowercased. This ignores
# markdown syntax (#, |, -, *), punctuation, and single stray characters so an
# engine that emits clean markdown isn't penalized for the syntax it added.
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏЀ-ӿ؀-ۿ]{2,}")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


# Ubiquitous function words carry no evidence that a page's *content* was
# extracted — they appear on every page. Excluding them keeps alignment and
# hallucination measuring distinctive content, so a dropped page isn't masked
# by shared stopwords. Deliberately small and language-light (the certifier is
# lexical, not linguistic — see LIMITATIONS on paraphrase/translation).
_STOPWORDS = frozenset(
    """
    the a an and or but of to in on at by for with from as is are was were be been
    being it its this that these those he she they we you i him her them us your our
    my his their which who whom whose what when where why how all any both each few
    more most other some such no nor not only own same so than too very can will just
    into over under out up down off above below then once here there about against
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _token_multiset(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in _tokens(text):
        if t in _STOPWORDS:
            continue
        counts[t] = counts.get(t, 0) + 1
    return counts


def _alignment(source_text: str, extracted_text: str) -> float:
    """Fraction of source content-token *occurrences* present in extraction.

    Multiset containment: if the source says "total" three times and the
    extraction says it once, that's 1/3 credit for that token. This rewards
    faithful transcription and penalizes silent partial drops.
    """
    src = _token_multiset(source_text)
    if not src:
        return 1.0  # nothing to cover
    ext = _token_multiset(extracted_text)
    covered = 0
    total = 0
    for tok, n in src.items():
        total += n
        covered += min(n, ext.get(tok, 0))
    return covered / total if total else 1.0


def _hallucination_risk(source_text: str, extracted_text: str) -> float:
    """Fraction of extraction content-token occurrences with no source support."""
    ext = _token_multiset(extracted_text)
    ext_total = sum(ext.values())
    if ext_total < MIN_TOKENS_FOR_HALLUCINATION:
        return 0.0  # too small to judge reliably
    src = _token_multiset(source_text)
    unsupported = 0
    for tok, n in ext.items():
        supported = min(n, src.get(tok, 0))
        unsupported += n - supported
    return unsupported / ext_total if ext_total else 0.0


def _has_table(text: str) -> bool:
    # Two or more pipe rows = a markdown table (header + at least one row).
    return len(_TABLE_ROW_RE.findall(text)) >= 2


def _has_heading(text: str) -> bool:
    return bool(_HEADING_RE.search(text))


# ---------------------------------------------------------------------------
# Extraction parsing — accept whatever the engine emitted
# ---------------------------------------------------------------------------

# Page-splitter markers seen in the wild for flat markdown/text dumps.
_PAGE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"<!--\s*page[:\s]*\d+\s*-->"  # <!-- page: 3 -->
    r"|\{?\d+\}?\s*-+\s*page\s*break"  # loose "page break"
    r"|={3,}\s*page\s*\d+\s*={3,}"  # === Page 3 ===
    r")\s*(?:\n|$)",
    re.IGNORECASE,
)

_TEXT_KEYS = ("text", "markdown", "md", "content", "value", "body")
_PAGE_NUM_KEYS = ("page", "page_num", "page_number", "pageNumber", "index", "page_no")


@dataclass(frozen=True)
class ParsedExtraction:
    """Normalized view of an external extraction."""

    pages: dict[int, str]  # 0-indexed page -> text
    page_aligned: bool  # True if real per-page structure was found
    raw_sha256: str


def _page_from_obj(obj: dict[str, Any]) -> tuple[int | None, str | None]:
    """Pull (page_index_0based_or_None, text_or_None) from a dict record."""
    page_val: int | None = None
    for k in _PAGE_NUM_KEYS:
        if k in obj and isinstance(obj[k], (int, float)):
            page_val = int(obj[k])
            break
    text_val: str | None = None
    for k in _TEXT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            text_val = v
            break
    return page_val, text_val


def _normalize_page_index(raw: int, seen_zero: bool) -> int:
    """Engines use 0- or 1-indexed page numbers; normalize to 0-indexed.

    If we ever see page 0, the source is 0-indexed. Otherwise assume 1-indexed
    (the far more common convention) and subtract one.
    """
    if seen_zero:
        return raw
    return max(0, raw - 1)


def _parse_json_extraction(data: Any) -> tuple[dict[int, str], bool]:
    """Return (pages_0indexed, page_aligned) from parsed JSON of many shapes."""
    records: list[dict[str, Any]] = []

    # Shape: {"pages": [...]}, {"result": {"pages":[...]}}, {"chunks":[...]}, etc.
    if isinstance(data, dict):
        for container_key in ("pages", "chunks", "blocks", "result", "data", "results"):
            inner = data.get(container_key)
            if isinstance(inner, dict):
                inner = inner.get("pages") or inner.get("chunks") or inner.get("blocks") or inner
            if isinstance(inner, list):
                records = [r for r in inner if isinstance(r, dict)]
                if records:
                    break

        # Shape: {"1": "text", "2": "text"} — dict of pagenum -> text
        if not records:
            numeric = {
                int(k): v for k, v in data.items() if str(k).isdigit() and isinstance(v, str)
            }
            if numeric:
                seen_zero = 0 in numeric
                pages = {_normalize_page_index(k, seen_zero): v for k, v in numeric.items()}
                return pages, True

        # Shape: single flat text field on the top-level object.
        if not records:
            _, whole = _page_from_obj(data)
            if whole is not None:
                return _split_flat_text(whole)

    # Shape: top-level list of page/chunk records.
    elif isinstance(data, list):
        if data and all(isinstance(x, str) for x in data):
            # list of page strings
            return {i: t for i, t in enumerate(data)}, True
        records = [r for r in data if isinstance(r, dict)]

    if not records:
        return {}, False

    # Do any records carry an explicit page number?
    raw_pages = [_page_from_obj(r)[0] for r in records if _page_from_obj(r)[0] is not None]
    has_page_nums = len(raw_pages) >= max(1, len(records) // 2)
    seen_zero = any(p == 0 for p in raw_pages)

    pages: dict[int, list[str]] = {}
    for i, rec in enumerate(records):
        pnum, text = _page_from_obj(rec)
        if text is None:
            text = ""
        if has_page_nums and pnum is not None:
            idx = _normalize_page_index(pnum, seen_zero)
        else:
            idx = i  # positional fallback
        pages.setdefault(idx, []).append(text)

    merged = {k: "\n\n".join(t for t in v if t) for k, v in pages.items()}
    return merged, True


def _split_flat_text(text: str) -> tuple[dict[int, str], bool]:
    """Split a flat markdown/text dump into pages.

    Tries, in order: form-feed (\\f, the PDF page-break convention), explicit
    page markers, then a horizontal-rule heuristic. If nothing splits, returns
    the whole document as a single un-page-aligned blob (page_aligned=False)
    so per-page silent-drop detection is honestly disabled rather than faked.
    """
    if "\f" in text:
        parts = text.split("\f")
        return {i: p for i, p in enumerate(parts)}, True

    if _PAGE_MARKER_RE.search(text):
        parts = _PAGE_MARKER_RE.split(text)
        parts = [p for p in parts if p is not None]
        return {i: p for i, p in enumerate(parts)}, True

    return {0: text}, False


def parse_extraction(
    extracted: str | Path | dict | list,
    *,
    fmt: str = "auto",
) -> ParsedExtraction:
    """Normalize any external extraction into 0-indexed page text.

    Args:
        extracted: A path to a .json/.md/.txt file, OR an already-loaded
            dict/list (JSON), OR a raw string of markdown/text.
        fmt: "auto" | "json" | "markdown" | "text". "auto" sniffs by extension
            and content.

    Returns:
        ParsedExtraction with pages, page_aligned flag, and the SHA-256 of the
        raw extraction bytes (for the manifest).
    """
    raw_bytes: bytes
    data: Any = None
    is_json = False

    if isinstance(extracted, (dict, list)):
        data = extracted
        is_json = True
        raw_bytes = json.dumps(extracted, sort_keys=True, ensure_ascii=False).encode("utf-8")
    else:
        if isinstance(extracted, Path) or (
            isinstance(extracted, str) and _looks_like_path(extracted)
        ):
            path = Path(extracted)
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            ext = path.suffix.lower()
            if fmt == "auto":
                fmt = (
                    "json"
                    if ext == ".json"
                    else "markdown"
                    if ext in {".md", ".markdown"}
                    else "text"
                )
        else:
            raw_text = str(extracted)

        raw_bytes = raw_text.encode("utf-8")

        if fmt == "json" or (fmt == "auto" and raw_text.lstrip()[:1] in "{["):
            try:
                data = json.loads(raw_text)
                is_json = True
            except (json.JSONDecodeError, ValueError):
                is_json = False

    raw_sha = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()

    if is_json:
        pages, aligned = _parse_json_extraction(data)
        if not pages:
            # JSON parsed but no recognizable text — treat as empty extraction.
            return ParsedExtraction(pages={}, page_aligned=False, raw_sha256=raw_sha)
        return ParsedExtraction(pages=pages, page_aligned=aligned, raw_sha256=raw_sha)

    # Markdown / plain text. raw_text is always defined here: the dict/list
    # branch returns early via is_json, every other path sets raw_text.
    pages, aligned = _split_flat_text(raw_text)
    return ParsedExtraction(pages=pages, page_aligned=aligned, raw_sha256=raw_sha)


def _looks_like_path(s: str) -> bool:
    if "\n" in s or len(s) > 400:
        return False
    return Path(s).exists()


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------


def _verify_page(
    page_num: int,
    source_text: str,
    extracted_text: str,
    *,
    recovery_text: str | None = None,
) -> PageVerification:
    """Score one page of an extraction against the source page text.

    Args:
        page_num: 0-indexed page number.
        source_text: Ground-truth text for this page (from pdfmux's own pass).
        extracted_text: The engine's text for this page slot.
        recovery_text: The WHOLE extraction's text. Used to distinguish a true
            silent drop (content gone everywhere) from a re-pagination (the
            engine put this page's content in a different slot). Without it,
            an empty slot with substantial source is treated as a silent drop.
    """
    source_text = source_text or ""
    extracted_text = extracted_text or ""
    source_chars = len(source_text.strip())
    extracted_chars = len(extracted_text.strip())

    flags: list[str] = []

    # --- Unverifiable: the source has no digital text we can cheaply read. ---
    # We refuse to certify what we cannot re-derive. If the source page is a
    # pure scan (no text layer), pdfmux's fast pass reads ~nothing, so we
    # cannot judge whether the engine's OCR output is faithful.
    if source_chars < EMPTY_TEXT_THRESHOLD:
        if extracted_chars < EMPTY_TEXT_THRESHOLD:
            # Both empty — genuinely blank page. Clean pass.
            return PageVerification(
                page_num=page_num,
                source_chars=source_chars,
                extracted_chars=extracted_chars,
                coverage=1.0,
                confidence=1.0,
                alignment=1.0,
                hallucination_risk=0.0,
                silent_drop=False,
                table_integrity=True,
                heading_integrity=True,
                verdict="pass",
                flags=("blank_page",),
            )
        # Engine produced text from a source we can't read cheaply.
        conf = score_page(extracted_text)
        return PageVerification(
            page_num=page_num,
            source_chars=source_chars,
            extracted_chars=extracted_chars,
            coverage=1.0,
            confidence=conf,
            alignment=0.0,
            hallucination_risk=0.0,
            silent_drop=False,
            table_integrity=True,
            heading_integrity=True,
            verdict="unverifiable",
            flags=("source_no_text_layer",),
        )

    # --- Empty slot: source clearly has text, engine returned ~nothing here. ---
    if source_chars >= GOOD_TEXT_THRESHOLD and extracted_chars < EMPTY_TEXT_THRESHOLD:
        # Before crying "silent drop", check whether this page's content lives
        # elsewhere in the extraction (merged/re-paginated, not lost).
        if recovery_text is not None:
            recovered = _alignment(source_text, recovery_text)
            if recovered >= SILENT_DROP_RECOVERY_THRESHOLD:
                return PageVerification(
                    page_num=page_num,
                    source_chars=source_chars,
                    extracted_chars=extracted_chars,
                    coverage=round(recovered, 4),
                    confidence=round(recovered, 4),
                    alignment=round(recovered, 4),
                    hallucination_risk=0.0,
                    silent_drop=False,
                    table_integrity=True,
                    heading_integrity=True,
                    verdict="review",
                    flags=("content_off_page",),
                )
        flags.append("silent_drop")
        return PageVerification(
            page_num=page_num,
            source_chars=source_chars,
            extracted_chars=extracted_chars,
            coverage=0.0,
            confidence=0.0,
            alignment=0.0,
            hallucination_risk=0.0,
            silent_drop=True,
            table_integrity=not _has_table(source_text),
            heading_integrity=not _has_heading(source_text),
            verdict="fail",
            flags=tuple(flags),
        )

    # --- Standard scoring path. ---
    confidence = score_page(extracted_text)
    coverage = min(1.0, extracted_chars / source_chars) if source_chars else 1.0
    alignment = _alignment(source_text, extracted_text)
    hallu = _hallucination_risk(source_text, extracted_text)

    # Integrity: did the extraction preserve tables / headings the source had?
    source_has_table = _has_table(source_text)
    table_integrity = (not source_has_table) or _has_table(extracted_text)
    source_has_heading = _has_heading(source_text)
    heading_integrity = (not source_has_heading) or _has_heading(extracted_text)

    if coverage < LOW_COVERAGE_THRESHOLD:
        flags.append("low_coverage")
    if alignment < LOW_ALIGNMENT_THRESHOLD:
        flags.append("low_alignment")
    if hallu > HIGH_HALLUCINATION_THRESHOLD:
        flags.append("hallucination_risk")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append("low_confidence")
    if not table_integrity:
        flags.append("table_dropped")
    if not heading_integrity:
        flags.append("headings_dropped")

    # Verdict. Silent drop already handled. A page fails only on catastrophic
    # loss (both coverage and alignment gone) — otherwise it's a review.
    if coverage < 0.25 and alignment < 0.35:
        verdict = "fail"
        flags.append("severe_content_loss")
    elif flags:
        verdict = "review"
    else:
        verdict = "pass"

    return PageVerification(
        page_num=page_num,
        source_chars=source_chars,
        extracted_chars=extracted_chars,
        coverage=round(coverage, 4),
        confidence=round(confidence, 4),
        alignment=round(alignment, 4),
        hallucination_risk=round(hallu, 4),
        silent_drop=False,
        table_integrity=table_integrity,
        heading_integrity=heading_integrity,
        verdict=verdict,
        flags=tuple(flags),
    )


def verify_extraction(
    source_pdf: str | Path,
    extracted: str | Path | dict | list,
    *,
    engine: str = "external",
    fmt: str = "auto",
    source_pages: dict[int, str] | None = None,
) -> CertificationManifest:
    """Audit an external extraction of ``source_pdf`` and return a manifest.

    This is the public entry point. It re-derives the source page text with
    pdfmux's own audit pass, aligns the external extraction to it, scores every
    page, and emits a signed certification manifest.

    Args:
        source_pdf: Path to the original PDF (the ground truth).
        extracted: The engine's output — a path to .json/.md/.txt, a loaded
            dict/list, or a raw string.
        engine: Label for the engine under audit (goes in the manifest).
        fmt: Extraction format hint ("auto" | "json" | "markdown" | "text").
        source_pages: Optional pre-computed 0-indexed source page text, to skip
            re-extraction (used by batch mode and tests).

    Returns:
        A :class:`CertificationManifest`.
    """
    source_pdf = Path(source_pdf)
    if not source_pdf.exists():
        raise FileNotFoundError(f"Source PDF not found: {source_pdf}")

    parsed = parse_extraction(extracted, fmt=fmt)

    # Ground truth: pdfmux's own per-page fast extraction of the source.
    if source_pages is None:
        source_pages = _source_page_text(source_pdf)

    source_sha = _file_sha256(source_pdf)

    n_src = len(source_pages)
    whole_ext_text = "\n\n".join(parsed.pages[i] for i in sorted(parsed.pages) if parsed.pages[i])

    # Do the source and extraction paginations agree closely enough to compare
    # page-by-page? Real per-page engine outputs (Reducto/LlamaParse JSON, the
    # ARK 433/433 case) do — a dropped page shows up as an empty slot in an
    # otherwise-aligned array. A single-blob extraction (or one whose page
    # boundaries differ wildly) does not, and positional mapping there would
    # fabricate silent drops. Tolerance scales with document length.
    tol = max(1, round(0.1 * n_src)) if n_src else 1
    paginations_agree = parsed.page_aligned and n_src > 0 and abs(len(parsed.pages) - n_src) <= tol

    page_results: list[PageVerification] = []

    if paginations_agree:
        # --- Mode A: segmented, per-page comparison (with recovery guard). ---
        all_indices = set(source_pages) | set(parsed.pages)
        n_pages = (max(all_indices) + 1) if all_indices else 1
        for idx in range(n_pages):
            src = source_pages.get(idx, "")
            ext = parsed.pages.get(idx, "")
            page_results.append(_verify_page(idx, src, ext, recovery_text=whole_ext_text))
        effective_page_aligned = True
    else:
        # --- Mode B: unsegmented — compare each source page against the WHOLE
        # extraction by content presence. Catches genuinely-missing pages
        # (silent drops) without fabricating them from pagination mismatch. ---
        page_results = _verify_unsegmented(source_pages, whole_ext_text)
        effective_page_aligned = False

    return _build_manifest(
        source_pdf=source_pdf,
        source_sha=source_sha,
        extraction_sha=parsed.raw_sha256,
        engine=engine,
        page_aligned=effective_page_aligned,
        pages=page_results,
        source_pages=source_pages,
    )


def _verify_unsegmented(
    source_pages: dict[int, str],
    whole_ext_text: str,
) -> list[PageVerification]:
    """Verify an unsegmented extraction (blob / mismatched pagination).

    We cannot trust the extraction's page boundaries, so each source page is
    checked for *presence* in the whole extraction. Content that is all there,
    just not segmented, passes; content genuinely missing from the entire
    extraction is a silent drop. Confidence/coverage/hallucination are computed
    once at the whole-document level and shared across rows (they can't be
    attributed per page without boundaries).
    """
    whole_source = "\n\n".join(source_pages[i] for i in sorted(source_pages) if source_pages[i])
    src_chars = len(whole_source.strip())
    ext_chars = len(whole_ext_text.strip())

    whole_conf = score_page(whole_ext_text) if ext_chars else 0.0
    whole_cov = min(1.0, ext_chars / src_chars) if src_chars else 1.0
    whole_hallu = _hallucination_risk(whole_source, whole_ext_text)

    rows: list[PageVerification] = []
    for idx in sorted(source_pages):
        src = source_pages[idx] or ""
        src_page_chars = len(src.strip())

        if src_page_chars < EMPTY_TEXT_THRESHOLD:
            # Blank source page — nothing to lose.
            rows.append(
                PageVerification(
                    page_num=idx,
                    source_chars=src_page_chars,
                    extracted_chars=0,
                    coverage=1.0,
                    confidence=1.0,
                    alignment=1.0,
                    hallucination_risk=0.0,
                    silent_drop=False,
                    table_integrity=True,
                    heading_integrity=True,
                    verdict="pass",
                    flags=("blank_page", "unsegmented_extraction"),
                )
            )
            continue

        alignment = _alignment(src, whole_ext_text)
        flags = ["unsegmented_extraction"]

        if src_page_chars >= GOOD_TEXT_THRESHOLD and alignment < MISSING_CONTENT_THRESHOLD:
            # Substantial source page whose content is absent from the entire
            # extraction — a genuine silent drop, detected without pagination.
            flags.insert(0, "silent_drop")
            rows.append(
                PageVerification(
                    page_num=idx,
                    source_chars=src_page_chars,
                    extracted_chars=0,
                    coverage=0.0,
                    confidence=0.0,
                    alignment=round(alignment, 4),
                    hallucination_risk=0.0,
                    silent_drop=True,
                    table_integrity=not _has_table(src),
                    heading_integrity=not _has_heading(src),
                    verdict="fail",
                    flags=tuple(flags),
                )
            )
            continue

        # Content present (fully or partly) somewhere in the extraction.
        if alignment < LOW_ALIGNMENT_THRESHOLD:
            flags.append("low_alignment")
        if whole_hallu > HIGH_HALLUCINATION_THRESHOLD:
            flags.append("hallucination_risk")
        verdict = "review" if len(flags) > 1 else "pass"
        rows.append(
            PageVerification(
                page_num=idx,
                source_chars=src_page_chars,
                extracted_chars=0,  # not attributable per page
                coverage=round(whole_cov, 4),
                confidence=round(whole_conf, 4),
                alignment=round(alignment, 4),
                hallucination_risk=round(whole_hallu, 4),
                silent_drop=False,
                table_integrity=True,
                heading_integrity=True,
                verdict=verdict,
                flags=tuple(flags),
            )
        )
    return rows


def _with_flag(pv: PageVerification, flag: str) -> PageVerification:
    if flag in pv.flags:
        return pv
    return PageVerification(**{**asdict(pv), "flags": pv.flags + (flag,)})


def _build_manifest(
    *,
    source_pdf: Path,
    source_sha: str,
    extraction_sha: str,
    engine: str,
    page_aligned: bool,
    pages: list[PageVerification],
    source_pages: dict[int, str],
) -> CertificationManifest:
    from pdfmux import __version__

    silent_drops = tuple(p.page_num + 1 for p in pages if p.silent_drop)
    low_conf = tuple(
        p.page_num + 1
        for p in pages
        if p.confidence < LOW_CONFIDENCE_THRESHOLD and p.verdict not in {"unverifiable", "fail"}
    )
    review_pages = tuple(p.page_num + 1 for p in pages if p.verdict == "review")
    unverifiable = tuple(p.page_num + 1 for p in pages if p.verdict == "unverifiable")
    fail_pages = [p for p in pages if p.verdict == "fail"]

    # Content-weighted overall confidence + coverage, weighted by source chars
    # (a 3000-char page matters more than a 40-char page) — mirrors pdfmux's own
    # compute_document_confidence weighting.
    total_weight = sum(max(1, p.source_chars) for p in pages) or 1
    confidence = sum(p.confidence * max(1, p.source_chars) for p in pages) / total_weight
    coverage = sum(p.coverage * max(1, p.source_chars) for p in pages) / total_weight

    # Overall verdict.
    if fail_pages or silent_drops:
        verdict = "FAIL"
    elif review_pages or low_conf or unverifiable:
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    summary = _build_summary(
        engine=engine,
        page_count=len(pages) if page_aligned else len(source_pages) or 1,
        verdict=verdict,
        silent_drops=silent_drops,
        low_conf=low_conf,
        review_pages=review_pages,
        unverifiable=unverifiable,
        confidence=confidence,
        coverage=coverage,
        page_aligned=page_aligned,
    )

    manifest = CertificationManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source=source_pdf.name,
        source_sha256=source_sha,
        extraction_sha256=extraction_sha,
        engine=engine,
        tool=f"pdfmux verify {__version__}",
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        page_count=len(pages) if page_aligned else (len(source_pages) or 1),
        page_aligned=page_aligned,
        verdict=verdict,
        confidence=round(confidence, 4),
        coverage=round(coverage, 4),
        silent_drops=silent_drops,
        low_confidence_pages=low_conf,
        review_pages=review_pages,
        unverifiable_pages=unverifiable,
        summary=summary,
        pages=tuple(pages),
    )
    # Sign last, over the frozen body.
    signature = _compute_signature(manifest._body())
    return CertificationManifest(
        **{**asdict(manifest), "pages": tuple(pages), "signature": signature}
    )


def _build_summary(
    *,
    engine: str,
    page_count: int,
    verdict: str,
    silent_drops: tuple[int, ...],
    low_conf: tuple[int, ...],
    review_pages: tuple[int, ...],
    unverifiable: tuple[int, ...],
    confidence: float,
    coverage: float,
    page_aligned: bool,
) -> str:
    bits = [f"{engine}: {verdict}"]
    if silent_drops:
        bits.append(
            f"{len(silent_drops)} page(s) SILENTLY DROPPED (pages {_fmt_pages(silent_drops)})"
        )
    if review_pages:
        bits.append(f"{len(review_pages)} page(s) need review")
    if low_conf:
        bits.append(f"{len(low_conf)} low-confidence page(s)")
    if unverifiable:
        bits.append(f"{len(unverifiable)} unverifiable page(s) (source has no text layer)")
    bits.append(
        f"overall confidence {confidence:.0%}, coverage {coverage:.0%} across {page_count} page(s)"
    )
    if not page_aligned:
        bits.append(
            "extraction was not page-segmented — per-page silent-drop detection unavailable"
        )
    return "; ".join(bits) + "."


def _fmt_pages(pages: tuple[int, ...], limit: int = 8) -> str:
    if len(pages) <= limit:
        return ", ".join(str(p) for p in pages)
    return ", ".join(str(p) for p in pages[:limit]) + f", … (+{len(pages) - limit})"


# ---------------------------------------------------------------------------
# Batch — the ARK "M pages silently dropped across N docs" report
# ---------------------------------------------------------------------------


@dataclass
class BatchCertification:
    """Aggregate certification over many documents — the killer report."""

    engine: str
    tool: str
    timestamp: str
    doc_count: int
    manifests: list[CertificationManifest] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (name, message)

    @property
    def total_pages(self) -> int:
        return sum(m.page_count for m in self.manifests)

    @property
    def total_silent_drops(self) -> int:
        return sum(len(m.silent_drops) for m in self.manifests)

    @property
    def total_low_confidence(self) -> int:
        return sum(len(m.low_confidence_pages) for m in self.manifests)

    @property
    def docs_failed(self) -> list[CertificationManifest]:
        return [m for m in self.manifests if m.verdict == "FAIL"]

    @property
    def docs_review(self) -> list[CertificationManifest]:
        return [m for m in self.manifests if m.verdict == "REVIEW"]

    @property
    def docs_passed(self) -> list[CertificationManifest]:
        return [m for m in self.manifests if m.verdict == "PASS"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "tool": self.tool,
            "timestamp": self.timestamp,
            "doc_count": self.doc_count,
            "total_pages": self.total_pages,
            "total_silent_drops": self.total_silent_drops,
            "total_low_confidence": self.total_low_confidence,
            "docs_failed": len(self.docs_failed),
            "docs_review": len(self.docs_review),
            "docs_passed": len(self.docs_passed),
            "documents": [m.to_dict() for m in self.manifests],
            "errors": [{"document": n, "error": e} for n, e in self.errors],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        return _batch_markdown(self)


def verify_batch(
    pairs: list[tuple[Path, Path]],
    *,
    engine: str = "external",
    fmt: str = "auto",
) -> BatchCertification:
    """Certify many (source_pdf, extraction) pairs and aggregate the result.

    This is the ARK story as a product: point it at a directory of PDFs plus a
    directory of an engine's outputs and get back "M pages silently dropped
    across N docs, here's the diff."

    Args:
        pairs: List of (source_pdf_path, extraction_path).
        engine: Label for the engine under audit.
        fmt: Extraction format hint.

    Returns:
        A :class:`BatchCertification`.
    """
    from pdfmux import __version__

    batch = BatchCertification(
        engine=engine,
        tool=f"pdfmux verify {__version__}",
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        doc_count=len(pairs),
    )
    for source_pdf, extraction in pairs:
        try:
            manifest = verify_extraction(source_pdf, extraction, engine=engine, fmt=fmt)
            batch.manifests.append(manifest)
        except Exception as exc:  # noqa: BLE001 — one bad doc shouldn't kill the run
            batch.errors.append((Path(source_pdf).name, str(exc)))
    return batch


# ---------------------------------------------------------------------------
# Source ground-truth extraction (reuses pdfmux's own audit pass)
# ---------------------------------------------------------------------------


def _source_page_text(source_pdf: Path) -> dict[int, str]:
    """Re-derive per-page source text using pdfmux's own audit extraction.

    This deliberately reuses :func:`pdfmux.audit.audit_document` — the exact
    same column-reorder + heading-injection + pymupdf4llm pass that powers
    pdfmux's multi-pass extraction. The verifier's ground truth is pdfmux's own
    reading of the document, which is the honest thing to certify against.
    """
    audit = audit_document(source_pdf)
    return {pa.page_num: pa.text for pa in audit.pages}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Engine adapters — the `--engine <name>` live-call path
# ---------------------------------------------------------------------------
#
# `pdfmux verify --engine <name>` runs the named engine on the source PDF and
# then audits its output — a one-shot "call + certify". Only the `pdfmux`
# self-adapter is wired live here (it dogfoods the whole path end-to-end and is
# fully runnable). Third-party adapters are documented stubs: each raises a
# clear error explaining the contract and the env var / package needed to
# implement it. The contract for every adapter is identical:
#
#     adapter(source_pdf: Path) -> str   # returns a JSON/markdown extraction
#
# so wiring a new engine is a ~15-line function that calls the engine's SDK and
# returns whatever it emits. The verifier normalizes the rest.


def _adapter_pdfmux(source_pdf: Path) -> str:
    """Live adapter: run pdfmux's own extraction and return its JSON output.

    Fully implemented — this is the reference adapter and makes
    ``pdfmux verify --engine pdfmux --source doc.pdf`` runnable end-to-end. It
    also means pdfmux can certify *itself*, which is exactly the reduction-to-
    practice evidence the patent wants: the audit mechanism scoring an
    independent extraction.
    """
    from pdfmux.pipeline import process

    result = process(file_path=source_pdf, output_format="json", quality="standard")
    return result.text  # already a JSON string with a "pages" array


def _stub_adapter(name: str, env_hint: str, pkg_hint: str):
    def _adapter(source_pdf: Path) -> str:
        raise NotImplementedError(
            f"The '{name}' engine adapter is documented but not wired in the OSS "
            f"core. To certify {name} output today, run {name} yourself and pass "
            f"the result with:  pdfmux verify --source {source_pdf.name} "
            f"--extracted <{name}_output.json>\n"
            f"To implement the live adapter: install {pkg_hint}, read {env_hint}, "
            f"call the engine on the PDF, and return its raw output string. The "
            f"adapter contract is:  adapter(source_pdf: Path) -> str."
        )

    return _adapter


# Registry. Live adapters map to real callables; stubs carry the contract doc.
ENGINE_ADAPTERS = {
    "pdfmux": _adapter_pdfmux,
    "reducto": _stub_adapter("reducto", "REDUCTO_API_KEY", "the `reductoai` SDK"),
    "mistral": _stub_adapter("mistral", "MISTRAL_API_KEY", "the `mistralai` SDK (OCR API)"),
    "mistral_ocr": _stub_adapter("mistral_ocr", "MISTRAL_API_KEY", "the `mistralai` SDK (OCR API)"),
    "llamaparse": _stub_adapter("llamaparse", "LLAMA_CLOUD_API_KEY", "the `llama-parse` package"),
    "docling": _stub_adapter("docling", "(none — local)", "the `docling` package"),
    "unstructured": _stub_adapter(
        "unstructured", "UNSTRUCTURED_API_KEY", "the `unstructured` package"
    ),
}


def run_engine(name: str, source_pdf: Path) -> str:
    """Run a named engine on the source PDF and return its raw extraction.

    Args:
        name: Engine key (see :data:`ENGINE_ADAPTERS`).
        source_pdf: Path to the source PDF.

    Returns:
        The engine's raw output (JSON or markdown string), ready for
        :func:`parse_extraction`.

    Raises:
        KeyError: Unknown engine.
        NotImplementedError: Engine adapter is a documented stub.
    """
    key = name.lower().strip()
    if key not in ENGINE_ADAPTERS:
        known = ", ".join(sorted(ENGINE_ADAPTERS))
        raise KeyError(f"Unknown engine '{name}'. Known: {known}")
    return ENGINE_ADAPTERS[key](source_pdf)


def verify_with_engine(
    source_pdf: str | Path,
    engine: str,
) -> CertificationManifest:
    """Run ``engine`` on ``source_pdf`` then certify its output."""
    source_pdf = Path(source_pdf)
    raw = run_engine(engine, source_pdf)
    return verify_extraction(source_pdf, raw, engine=engine, fmt="auto")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_VERDICT_BADGE = {
    "PASS": "✅ PASS",
    "REVIEW": "⚠️  REVIEW",
    "FAIL": "❌ FAIL",
    "pass": "✅",
    "review": "⚠️",
    "fail": "❌",
    "unverifiable": "❓",
}


def _manifest_markdown(m: CertificationManifest) -> str:
    lines: list[str] = []
    lines.append(f"# Certification Manifest — {m.source}")
    lines.append("")
    lines.append(f"**Verdict:** {_VERDICT_BADGE.get(m.verdict, m.verdict)}  ")
    lines.append(f"**Engine audited:** `{m.engine}`  ")
    lines.append(f"**Overall confidence:** {m.confidence:.0%}  ·  **Coverage:** {m.coverage:.0%}  ")
    lines.append(f"**Pages:** {m.page_count}  ·  **Tool:** {m.tool}  ")
    lines.append(f"**Timestamp:** {m.timestamp}  ")
    lines.append("")
    lines.append(f"> {m.summary}")
    lines.append("")

    if m.silent_drops:
        lines.append(f"## ❌ Silent drops ({len(m.silent_drops)})")
        lines.append("")
        lines.append(
            "Pages where the source has substantial text but the engine "
            "returned nothing — while reporting success:"
        )
        lines.append("")
        lines.append(f"`pages {_fmt_pages(m.silent_drops, limit=100)}`")
        lines.append("")

    if m.page_aligned:
        lines.append("## Per-page results")
        lines.append("")
        lines.append("| Page | Verdict | Conf | Cover | Align | Halluc | Src ch | Ext ch | Flags |")
        lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|:---|")
        for p in m.pages:
            badge = _VERDICT_BADGE.get(p.verdict, p.verdict)
            flags = ", ".join(p.flags) if p.flags else "—"
            lines.append(
                f"| {p.page_num + 1} | {badge} | {p.confidence:.0%} | {p.coverage:.0%} | "
                f"{p.alignment:.0%} | {p.hallucination_risk:.0%} | {p.source_chars} | "
                f"{p.extracted_chars} | {flags} |"
            )
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Source SHA-256:** `{m.source_sha256}`")
    lines.append(f"- **Extraction SHA-256:** `{m.extraction_sha256}`")
    lines.append(f"- **Signature:** `{m.signature}`")
    lines.append(f"- **Schema:** `{m.schema_version}`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in LIMITATIONS:
        lines.append(f"- {lim}")
    lines.append("")
    lines.append(
        "_Signed with a SHA-256 content hash (tamper-evident, reproducible). "
        "Cloud tier adds an Ed25519 signature bound to a pdfmux key for "
        "third-party verification._"
    )
    lines.append("")
    return "\n".join(lines)


def _batch_markdown(b: BatchCertification) -> str:
    lines: list[str] = []
    lines.append(f"# Certification Report — `{b.engine}`")
    lines.append("")
    lines.append(
        f"Ran **{b.engine}** through pdfmux's certifier on **{b.doc_count} document(s)** "
        f"({b.total_pages} pages)."
    )
    lines.append("")
    lines.append(f"- ❌ **{b.total_silent_drops} pages silently dropped**")
    lines.append(f"- ⚠️  **{b.total_low_confidence} low-confidence pages**")
    lines.append(
        f"- Documents: {len(b.docs_passed)} PASS · {len(b.docs_review)} REVIEW · "
        f"{len(b.docs_failed)} FAIL"
    )
    if b.errors:
        lines.append(f"- 🛑 {len(b.errors)} document(s) errored during verification")
    lines.append("")
    lines.append("| Document | Verdict | Conf | Cover | Silent drops | Review | Pages |")
    lines.append("|:---|:---:|---:|---:|---:|---:|---:|")
    for m in b.manifests:
        badge = _VERDICT_BADGE.get(m.verdict, m.verdict)
        lines.append(
            f"| {m.source} | {badge} | {m.confidence:.0%} | {m.coverage:.0%} | "
            f"{len(m.silent_drops)} | {len(m.review_pages)} | {m.page_count} |"
        )
    lines.append("")
    if b.errors:
        lines.append("## Errors")
        lines.append("")
        for name, err in b.errors:
            lines.append(f"- `{name}`: {err}")
        lines.append("")
    lines.append(f"_{b.tool} · {b.timestamp}_")
    lines.append("")
    return "\n".join(lines)
