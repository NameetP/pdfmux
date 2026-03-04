"""Per-page quality auditing — the core of multi-pass extraction.

Fast-extracts every page individually, then scores each one to determine
which pages need re-extraction with OCR. This is the "audit" step in:

  fast extract → audit → selective OCR → merge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)

# --- Thresholds ---
# Derived from real-world pitch deck analysis (EPP Mahila Money, 47 pages).
# Pages with <200 chars + images consistently had text baked into images.
# Pages with 200+ chars were reliably extractable by PyMuPDF.
GOOD_TEXT_THRESHOLD = 200  # chars — above this, page is probably fine
MINIMAL_TEXT_THRESHOLD = 50  # chars — below this with images = definitely bad
EMPTY_TEXT_THRESHOLD = 20  # chars — below this = empty regardless


@dataclass(frozen=True)
class PageAudit:
    """Quality assessment for a single page."""

    page_num: int  # 0-indexed
    text: str  # Text from fast extraction
    text_len: int  # len(text.strip())
    image_count: int  # Number of images on this page
    quality: str  # "good" | "bad" | "empty"
    reason: str  # Human-readable explanation


@dataclass
class DocumentAudit:
    """Quality assessment for the entire document."""

    pages: list[PageAudit]
    total_pages: int

    @property
    def good_pages(self) -> list[int]:
        """Page numbers that passed quality audit."""
        return [p.page_num for p in self.pages if p.quality == "good"]

    @property
    def bad_pages(self) -> list[int]:
        """Page numbers with low text + images (text likely in images)."""
        return [p.page_num for p in self.pages if p.quality == "bad"]

    @property
    def empty_pages(self) -> list[int]:
        """Page numbers with no extractable text."""
        return [p.page_num for p in self.pages if p.quality == "empty"]

    @property
    def needs_ocr(self) -> bool:
        """Whether any pages need OCR re-extraction."""
        return len(self.bad_pages) + len(self.empty_pages) > 0


def audit_document(file_path: str | Path) -> DocumentAudit:
    """Fast-extract every page and score quality individually.

    Uses pymupdf4llm with page_chunks=True to get per-page data.
    Each page is classified:
      - "good":  text_len >= 200, OR text_len >= 50 with no images
      - "bad":   text_len < 200 AND has images (text likely in images)
      - "empty": text_len < 20 (regardless of images)

    Args:
        file_path: Path to the PDF file.

    Returns:
        DocumentAudit with per-page quality assessments.
    """
    file_path = Path(file_path)

    # Extract per-page data
    chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

    page_audits: list[PageAudit] = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        text_len = len(text.strip())
        image_count = len(chunk.get("images", []))

        quality, reason = _classify_page(text_len, image_count)

        page_audits.append(
            PageAudit(
                page_num=i,
                text=text,
                text_len=text_len,
                image_count=image_count,
                quality=quality,
                reason=reason,
            )
        )

    audit = DocumentAudit(pages=page_audits, total_pages=len(chunks))

    n_good = len(audit.good_pages)
    n_bad = len(audit.bad_pages)
    n_empty = len(audit.empty_pages)
    logger.info(
        f"Audit: {n_good} good, {n_bad} bad, {n_empty} empty "
        f"out of {audit.total_pages} pages"
    )

    return audit


def _classify_page(text_len: int, image_count: int) -> tuple[str, str]:
    """Classify a single page's extraction quality.

    Returns:
        Tuple of (quality, reason).
    """
    # Empty: barely any text at all
    if text_len < EMPTY_TEXT_THRESHOLD:
        if image_count > 0:
            return "empty", f"no text ({text_len} chars) with {image_count} images"
        return "empty", f"no text ({text_len} chars)"

    # Bad: some text but likely incomplete because images are present
    if text_len < GOOD_TEXT_THRESHOLD and image_count > 0:
        return "bad", f"low text ({text_len} chars) with {image_count} images"

    # Good with images: enough text that extraction probably worked
    if text_len >= GOOD_TEXT_THRESHOLD:
        return "good", f"{text_len} chars extracted"

    # Good without images: low text but no images, so nothing to OCR
    if text_len >= MINIMAL_TEXT_THRESHOLD and image_count == 0:
        return "good", f"{text_len} chars, no images"

    # Edge case: some text, no images, below minimal threshold
    # Not much we can do with OCR here — no images to OCR
    return "good", f"{text_len} chars, no images to OCR"
