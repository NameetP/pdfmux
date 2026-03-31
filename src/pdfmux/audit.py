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


def compute_document_confidence(
    pages: list[PageResult],
    *,
    ocr_page_count: int = 0,
    unrecovered_count: int = 0,
) -> tuple[float, list[str]]:
    """Content-weighted document confidence + warnings.

    Longer pages contribute more to the average — a 3000-char page
    matters more than a 50-char page.

    Returns:
        (confidence, warnings) tuple.
    """
    warnings: list[str] = []

    if not pages:
        warnings.append("Empty output — extraction may have failed")
        return 0.0, warnings

    # Content-weighted average
    total_chars = sum(max(1, p.char_count) for p in pages)
    if total_chars == 0:
        warnings.append("No text extracted from any page")
        return 0.0, warnings

    weighted_sum = sum(p.confidence * max(1, p.char_count) for p in pages)
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
            warnings.append(
                f"{len(empty_pages)} empty page(s) detected (pages: {page_nums})"
            )
        else:
            first_five = ", ".join(str(p.page_num + 1) for p in empty_pages[:5])
            warnings.append(
                f"{len(empty_pages)} empty pages detected "
                f"(first 5: {first_five}, ...)"
            )

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
