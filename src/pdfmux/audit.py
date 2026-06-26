"""Per-page quality auditing — the core of multi-pass extraction.

Fast-extracts every page individually, then scores each one to determine
which pages need re-extraction with OCR.

Confidence scoring uses 5 concrete checks per page:
    1. Character density  — enough text for the page to be useful
    2. Alphabetic ratio   — meaningful chars vs garbage/encoding noise
    3. Word structure     — average word length in normal range (2-20)
    4. Whitespace sanity  — not too much, not too little
    5. Encoding quality   — no mojibake patterns
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm

from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)

# --- Thresholds ---
GOOD_TEXT_THRESHOLD = 200  # chars — above this, page is probably fine
MINIMAL_TEXT_THRESHOLD = 50  # chars — below this with images = bad
EMPTY_TEXT_THRESHOLD = 20  # chars — below this = empty regardless
PAGE_WINDOW = 50  # pages per batch for windowed processing

# --- Monotonic repair guard (§5.8) ---------------------------------------
# A page's native text is "trusted" once its audit score clears this bar; a
# trusted span may only be *augmented* (region OCR), never wholesale-replaced
# by full-page OCR or an LLM — that path can only ever degrade good text.
NATIVE_TRUST_THRESHOLD = float(os.environ.get("PDFMUX_NATIVE_TRUST", "0.80"))
# A repair candidate is accepted only if it beats the original audit score by
# more than this margin. Default 0.0 = strictly-better. Raise it to demand a
# bigger improvement before disturbing the existing text.
REPAIR_MARGIN = float(os.environ.get("PDFMUX_REPAIR_MARGIN", "0.0"))
# Hard-fail tolerances — a candidate that trips any of these is rejected
# regardless of its audit-score delta (it degraded a signal we trust).
_ALPHA_COLLAPSE_DROP = 0.25  # alphabetic-ratio drop that signals OCR garbage
_SUSPICIOUS_SHRINK_FRACTION = 0.50  # full replacement losing >50% of text


# ---------------------------------------------------------------------------
# Legacy compat: PageAudit / DocumentAudit (used by existing tests + CLI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageAudit:
    """Quality assessment for a single page."""

    page_num: int  # 0-indexed
    text: str
    text_len: int
    image_count: int
    quality: str  # "good" | "bad" | "empty"
    reason: str


@dataclass
class DocumentAudit:
    """Quality assessment for the entire document."""

    pages: list[PageAudit]
    total_pages: int

    @property
    def good_pages(self) -> list[int]:
        return [p.page_num for p in self.pages if p.quality == "good"]

    @property
    def bad_pages(self) -> list[int]:
        return [p.page_num for p in self.pages if p.quality == "bad"]

    @property
    def empty_pages(self) -> list[int]:
        return [p.page_num for p in self.pages if p.quality == "empty"]

    @property
    def needs_ocr(self) -> bool:
        return len(self.bad_pages) + len(self.empty_pages) > 0


# ---------------------------------------------------------------------------
# Per-page confidence scoring — 5 concrete checks
# ---------------------------------------------------------------------------

# Mojibake patterns that signal encoding corruption
_MOJIBAKE_RE = re.compile(r"â€|Ã©|Ã¨|â€™|ï¿½")


def score_page(text: str, image_count: int = 0) -> float:
    """Compute a confidence score for a single page's text (0.0–1.0).

    Five checks, each can subtract from a starting score of 1.0:

    1. Character density — is there enough text?
    2. Alphabetic ratio — is it meaningful text or garbage?
    3. Word structure — are words a normal length?
    4. Whitespace sanity — not too much consecutive whitespace?
    5. Encoding quality — no mojibake?
    """
    stripped = text.strip()
    if not stripped:
        return 0.0

    score = 1.0

    # 1. Character density
    char_count = len(stripped)
    if char_count < EMPTY_TEXT_THRESHOLD:
        return 0.0  # effectively empty
    elif char_count < MINIMAL_TEXT_THRESHOLD:
        score -= 0.3
    elif char_count < GOOD_TEXT_THRESHOLD:
        score -= 0.1 if image_count == 0 else 0.2

    # 2. Alphabetic ratio — what fraction of non-space chars are letters?
    non_space = re.sub(r"\s", "", stripped)
    if non_space:
        alpha_count = sum(1 for c in non_space if c.isalpha())
        alpha_ratio = alpha_count / len(non_space)
        if alpha_ratio < 0.3:
            score -= 0.25  # mostly numbers/symbols/garbage
        elif alpha_ratio < 0.5:
            score -= 0.1

    # 3. Word structure — average word length should be 2-20
    words = stripped.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 25:
            score -= 0.15  # single chars or concatenated garbage

    # 4. Whitespace sanity — excessive runs of spaces
    wide_spaces = len(re.findall(r"  {5,}", text))
    if wide_spaces > 10:
        score -= 0.1

    # 5. Encoding quality — mojibake detection
    mojibake_count = len(_MOJIBAKE_RE.findall(text))
    if mojibake_count > 5:
        score -= 0.2
    elif mojibake_count > 0:
        score -= 0.05

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Monotonic repair guard (§5.8) — the single accept/reject gate for every
# re-extraction candidate (region OCR, full-page OCR, vision LLM, agentic).
#
# Three conjoined conditions, in order:
#   1. Additive-patch-only for trusted native spans — a high-scoring native
#      page may only be *augmented*, never wholesale-replaced.
#   2. Hard-fail signals — reject if the candidate degrades a trusted signal
#      (introduces mojibake, collapses the alpha ratio, suspiciously shortens,
#      or loses headings/tables) regardless of the score delta.
#   3. Calibrated audit-delta gate — accept only if the candidate's audit
#      score strictly beats the original's by more than REPAIR_MARGIN.
# Every call returns the before/after scores and a reason so the caller can
# record the attempt in the decision trace, accepted or rejected.
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#+\s", re.MULTILINE)


def _alpha_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are letters (0.0 if none)."""
    non_space = re.sub(r"\s", "", text)
    if not non_space:
        return 0.0
    return sum(1 for c in non_space if c.isalpha()) / len(non_space)


def _heading_count(text: str) -> int:
    """Number of markdown heading lines."""
    return len(_HEADING_RE.findall(text))


def _pipe_table_rows(text: str) -> int:
    """Number of markdown pipe-table rows (lines starting and ending with '|')."""
    return sum(
        1
        for line in text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    )


def repair_score_delta(before: str, after: str, image_count: int = 0) -> tuple[float, float]:
    """Audit scores of the original and candidate text: ``(score_before, score_after)``.

    The single calibrated signal both extraction paths now share — replacing
    the old char-length and raw-confidence comparisons that could disagree.
    """
    return score_page(before, image_count), score_page(after, image_count)


def _repair_hard_fail(before: str, after: str) -> str | None:
    """Return a rejection reason if the candidate degrades a trusted signal, else None.

    These checks fire regardless of the audit-score delta: a candidate can post
    a higher score while still throwing away structure or injecting garbage, and
    we never want that to overwrite the original.
    """
    before_s, after_s = before.strip(), after.strip()

    # Introduces mojibake the original didn't have.
    before_moji = len(_MOJIBAKE_RE.findall(before))
    after_moji = len(_MOJIBAKE_RE.findall(after))
    if after_moji > before_moji:
        return f"candidate introduced mojibake ({before_moji} → {after_moji})"

    # Collapses the alphabetic ratio — a hallmark of OCR garbage replacing prose.
    before_alpha = _alpha_ratio(before)
    after_alpha = _alpha_ratio(after)
    if before_alpha > 0 and after_alpha < before_alpha - _ALPHA_COLLAPSE_DROP:
        return f"candidate collapsed alpha ratio ({before_alpha:.2f} → {after_alpha:.2f})"

    # Suspiciously shortens a non-trivial original (a full replacement that
    # dropped most of the text is almost always a worse extraction).
    if len(before_s) >= EMPTY_TEXT_THRESHOLD and len(after_s) < _SUSPICIOUS_SHRINK_FRACTION * len(
        before_s
    ):
        return f"candidate dropped >{int((1 - _SUSPICIOUS_SHRINK_FRACTION) * 100)}% of text"

    # Loses headings or tables the original had.
    if _heading_count(after) < _heading_count(before):
        return "candidate lost headings present in the original"
    if _pipe_table_rows(after) < _pipe_table_rows(before):
        return "candidate lost table rows present in the original"

    return None


def accept_repair(
    original: str,
    candidate: str,
    *,
    image_count: int = 0,
    additive: bool = False,
    margin: float = REPAIR_MARGIN,
    trust_threshold: float = NATIVE_TRUST_THRESHOLD,
) -> tuple[bool, float, float, str]:
    """The monotonic repair guard. Returns ``(accepted, score_before, score_after, reason)``.

    Args:
        original: The page's current text.
        candidate: The re-extraction candidate to consider.
        image_count: Image count, forwarded to the audit scorer.
        additive: True when the candidate *augments* the original (region OCR
            appends recovered text and preserves the native span). Additive
            candidates are exempt from the trusted-native replacement bar, since
            they cannot overwrite good text — but they still face the hard-fail
            and audit-delta checks.
        margin: Minimum audit-score improvement required to accept.
        trust_threshold: Audit score above which native text is "trusted" and a
            full (non-additive) replacement is barred.
    """
    score_before, score_after = repair_score_delta(original, candidate, image_count)

    if not candidate.strip():
        return False, score_before, score_after, "candidate produced no text"

    # (1) Trusted native span — augment only, never wholesale-replace.
    if not additive and score_before >= trust_threshold:
        return (
            False,
            score_before,
            score_after,
            f"trusted native span (score {score_before:.2f} ≥ {trust_threshold:.2f}) — "
            f"full replacement barred",
        )

    # (2) Hard-fail signals — reject a degrading candidate outright.
    hard_fail = _repair_hard_fail(original, candidate)
    if hard_fail is not None:
        return False, score_before, score_after, hard_fail

    # (3) Calibrated audit-delta gate. Both arms are monotonic — quality never
    # decreases. An additive patch (region OCR) appends recovered text without
    # touching the native span, so it only needs to be non-decreasing; a full
    # replacement must *strictly* beat the original by the margin to earn the
    # right to discard the existing text.
    if additive:
        if score_after >= score_before:
            return (
                True,
                score_before,
                score_after,
                f"additive patch preserved audit score ({score_before:.2f} → {score_after:.2f})",
            )
        return (
            False,
            score_before,
            score_after,
            f"additive patch reduced audit score ({score_before:.2f} → {score_after:.2f})",
        )

    if score_after > score_before + margin:
        return (
            True,
            score_before,
            score_after,
            f"audit score improved {score_before:.2f} → {score_after:.2f}",
        )
    return (
        False,
        score_before,
        score_after,
        f"audit score not improved beyond margin {margin:.2f} "
        f"({score_before:.2f} → {score_after:.2f})",
    )


def compute_document_confidence(
    pages: list[PageResult],
    *,
    ocr_page_count: int = 0,
    unrecovered_count: int = 0,
) -> tuple[float, list[str]]:
    """Content-weighted document confidence + warnings.

    Longer pages contribute more to the average — a 3000-char page
    matters more than a 50-char page.

    Each page is re-scored against its actual text via ``score_page`` —
    we cannot trust the extractor's optimistic ``confidence=1.0`` default
    on a page that ended up with zero characters extracted.

    Returns:
        (confidence, warnings) tuple.
    """
    warnings: list[str] = []

    if not pages:
        warnings.append("Empty output — extraction may have failed")
        return 0.0, warnings

    # Re-score each page against its actual extracted text. The extractor
    # writes confidence=1.0 by default at yield time; that has to be
    # reconciled with what the page actually contains. Without this,
    # an HTML-as-PDF or a blank page sails through with confidence 1.0.
    page_scores: list[float] = []
    for p in pages:
        page_scores.append(score_page(p.text, p.image_count))

    # Content-weighted average over the *re-scored* per-page numbers.
    # Pages with zero text count as 0 weight (used to be max(1, ...) which
    # let empty-page documents register full confidence).
    total_chars = sum(p.char_count for p in pages)
    if total_chars == 0:
        warnings.append("No text extracted from any page")
        return 0.0, warnings

    weighted_sum = sum(score * p.char_count for score, p in zip(page_scores, pages, strict=False))
    score = weighted_sum / total_chars

    # OCR penalty — small noise penalty per OCR'd page
    if ocr_page_count > 0:
        ocr_ratio = ocr_page_count / len(pages)
        ocr_penalty = min(0.15, ocr_ratio * 0.2)
        score -= ocr_penalty

    # Unrecovered penalty
    if unrecovered_count > 0:
        unrec_ratio = unrecovered_count / len(pages)
        penalty = min(0.4, unrec_ratio * 0.5)
        score -= penalty
        warnings.append(
            f"{unrecovered_count} pages could not be recovered. "
            f"Install pdfmux[ocr] for better results."
        )

    # Empty page detection — always warn with page numbers
    empty_pages = [p for p in pages if p.char_count < 20]
    if empty_pages:
        if len(empty_pages) <= 5:
            page_nums = ", ".join(str(p.page_num + 1) for p in empty_pages)
            warnings.append(f"{len(empty_pages)} empty page(s) detected (pages: {page_nums})")
        else:
            first_five = ", ".join(str(p.page_num + 1) for p in empty_pages[:5])
            warnings.append(f"{len(empty_pages)} empty pages detected (first 5: {first_five}, ...)")

    # Sparse page detection (non-empty but low text)
    sparse = [p for p in pages if 20 <= p.char_count < 100]
    if sparse and len(sparse) / len(pages) > 0.25:
        warnings.append(f"{len(sparse)} pages have very little text")

    # Structure bonus — if any page has markdown headings
    has_structure = any(re.search(r"^#+\s", p.text, re.MULTILINE) for p in pages)
    if has_structure:
        score += 0.03

    return max(0.0, min(1.0, score)), warnings


# ---------------------------------------------------------------------------
# Audit pipeline entry point
# ---------------------------------------------------------------------------


def audit_document(file_path: str | Path) -> DocumentAudit:
    """Fast-extract every page and score quality individually.

    Each page is classified:
      - "good":  text_len >= 200, OR text_len >= 50 with no images
      - "bad":   text_len < 200 AND has images
      - "empty": text_len < 20

    Args:
        file_path: Path to the PDF file.

    Returns:
        DocumentAudit with per-page quality assessments.
    """
    file_path = Path(file_path)

    # Determine total page count
    try:
        from pdfmux.pdf_cache import get_doc

        doc = get_doc(file_path)
    except ImportError:
        import fitz

        doc = fitz.open(str(file_path))
    total_pages = len(doc)

    # Process in windows to bound memory on large documents
    from pdfmux.column_reorder import reorder_text_ab
    from pdfmux.headings import inject_headings

    page_audits: list[PageAudit] = []

    for start in range(0, total_pages, PAGE_WINDOW):
        end = min(start + PAGE_WINDOW, total_pages)
        page_range = list(range(start, end))

        chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True, pages=page_range)

        # Reuse cached doc for heading detection
        try:
            from pdfmux.pdf_cache import get_doc

            fitz_doc = get_doc(file_path)
        except ImportError:
            fitz_doc = fitz.open(str(file_path))

        for i, chunk in enumerate(chunks):
            page_num = start + i
            text = chunk.get("text", "")

            # Column-aware reading order (A/B comparison — safe, no-op if uncertain)
            if page_num < len(fitz_doc):
                text = reorder_text_ab(text, fitz_doc[page_num])

            # Inject heading markers via font-size analysis
            if page_num < len(fitz_doc):
                text = inject_headings(text, fitz_doc[page_num])

            text_len = len(text.strip())
            image_count = len(chunk.get("images", []))

            quality, reason = _classify_page(text_len, image_count)

            page_audits.append(
                PageAudit(
                    page_num=page_num,
                    text=text,
                    text_len=text_len,
                    image_count=image_count,
                    quality=quality,
                    reason=reason,
                )
            )

    audit = DocumentAudit(pages=page_audits, total_pages=total_pages)

    n_good = len(audit.good_pages)
    n_bad = len(audit.bad_pages)
    n_empty = len(audit.empty_pages)
    logger.info(
        f"Audit: {n_good} good, {n_bad} bad, {n_empty} empty out of {audit.total_pages} pages"
    )

    return audit


def audit_pages(pages: list[PageResult]) -> list[PageResult]:
    """Re-score a list of PageResults with proper quality classification.

    Takes raw PageResults from an extractor (where quality=GOOD by default)
    and applies the audit thresholds to set the true quality.

    Returns:
        New list of PageResult with updated quality and confidence.
    """
    audited = []
    for p in pages:
        text_len = p.char_count
        image_count = p.image_count
        quality_str, _ = _classify_page(text_len, image_count)
        quality = PageQuality(quality_str)
        confidence = score_page(p.text, image_count)

        audited.append(
            PageResult(
                page_num=p.page_num,
                text=p.text,
                confidence=confidence,
                quality=quality,
                extractor=p.extractor,
                image_count=p.image_count,
                ocr_applied=p.ocr_applied,
            )
        )
    return audited


def _classify_page(text_len: int, image_count: int) -> tuple[str, str]:
    """Classify a single page's extraction quality."""
    if text_len < EMPTY_TEXT_THRESHOLD:
        if image_count > 0:
            return "empty", f"no text ({text_len} chars) with {image_count} images"
        return "empty", f"no text ({text_len} chars)"

    if text_len < GOOD_TEXT_THRESHOLD and image_count > 0:
        return "bad", f"low text ({text_len} chars) with {image_count} images"

    if text_len >= GOOD_TEXT_THRESHOLD:
        return "good", f"{text_len} chars extracted"

    if text_len >= MINIMAL_TEXT_THRESHOLD and image_count == 0:
        return "good", f"{text_len} chars, no images"

    return "good", f"{text_len} chars, no images to OCR"
