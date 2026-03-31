"""Region OCR — surgical OCR of image regions within pages.

Instead of re-extracting entire pages with OCR (losing good digital text),
this module detects image regions without overlapping text and OCRs just
those regions, then merges the results back into the page text.

Pipeline rule:
    - "bad" pages  → region OCR (preserve existing good text)
    - "empty" pages → full-page OCR (nothing to preserve)
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.types import WeakRegion

logger = logging.getLogger(__name__)


def detect_weak_regions(
    file_path: str | Path,
    page_num: int,
) -> list[WeakRegion]:
    """Find image regions on a page that lack overlapping text.

    Algorithm:
    1. Get all images with their bounding boxes
    2. Get all text blocks with their bounding boxes
    3. For each image, check if any text block significantly overlaps
    4. Images without overlapping text → weak regions (need OCR)

    Args:
        file_path: Path to the PDF file.
        page_num: 0-indexed page number.

    Returns:
        List of WeakRegion objects for images needing OCR.
    """
    try:
        from pdfmux.pdf_cache import get_doc

        doc = get_doc(file_path)
    except ImportError:
        doc = fitz.open(str(file_path))
    if page_num >= len(doc):
        return []

    page = doc[page_num]
    regions: list[WeakRegion] = []

    # Get image bboxes
    image_bboxes = _get_image_bboxes(page)
    if not image_bboxes:
        return []

    # Get text block bboxes
    text_bboxes = _get_text_bboxes(page)

    for img_rect in image_bboxes:
        # Check overlap with text blocks
        has_text = _has_significant_text_overlap(img_rect, text_bboxes)
        if not has_text:
            bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
            regions.append(
                WeakRegion(
                    page_num=page_num,
                    bbox=bbox,
                    reason="image without overlapping text",
                )
            )

    logger.debug(
        f"Page {page_num}: found {len(regions)} weak regions out of {len(image_bboxes)} images"
    )
    return regions


def ocr_region(
    file_path: str | Path,
    region: WeakRegion,
    *,
    dpi: int = 200,
) -> str:
    """OCR a specific region of a page.

    Renders the region at the given DPI and runs OCR on the cropped image.

    Args:
        file_path: Path to the PDF file.
        region: WeakRegion with page number and bounding box.
        dpi: Resolution for rendering (default 200).

    Returns:
        Extracted text from the region, or empty string on failure.
    """
    try:
        from rapidocr import RapidOCR
    except ImportError:
        logger.debug("RapidOCR not available for region OCR")
        return ""

    try:
        from pdfmux.pdf_cache import get_doc

        doc = get_doc(file_path)
    except ImportError:
        doc = fitz.open(str(file_path))
    if region.page_num >= len(doc):
        return ""

    page = doc[region.page_num]
    clip_rect = fitz.Rect(region.bbox)

    # Render just the clipped region at target DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)

    # Convert to bytes for OCR
    img_bytes = pix.tobytes("png")

    # Run OCR on the cropped image
    try:
        engine = RapidOCR()
        result = engine(img_bytes)
        if result and result[0]:
            lines = [item[1] for item in result[0]]
            return "\n".join(lines)
    except Exception as e:
        logger.debug(f"Region OCR failed: {e}")

    return ""


def merge_region_text(
    page_text: str,
    regions: list[WeakRegion],
    region_texts: list[str],
) -> str:
    """Merge OCR'd region text into the existing page text.

    Appends OCR results at the end of the page text, separated by
    a marker. Region text is inserted in top-to-bottom order by
    the region's y-position.

    Args:
        page_text: Existing page text from fast extraction.
        regions: List of WeakRegion objects that were OCR'd.
        region_texts: Corresponding OCR text for each region.

    Returns:
        Merged text with OCR content appended.
    """
    if not regions or not region_texts:
        return page_text

    # Pair regions with their text, filter empty
    paired = [(r, t) for r, t in zip(regions, region_texts) if t.strip()]

    if not paired:
        return page_text

    # Sort by y-position (top of bbox) for reading order
    paired.sort(key=lambda x: x[0].bbox[1])

    # Build merged text
    parts = [page_text.rstrip()]
    for _region, text in paired:
        parts.append(text.strip())

    return "\n\n".join(parts)


def region_ocr_page(
    file_path: str | Path,
    page_num: int,
    page_text: str,
) -> tuple[str, int]:
    """Run region OCR on a single page and merge results.

    Convenience function that combines detect → OCR → merge.

    Args:
        file_path: Path to the PDF file.
        page_num: 0-indexed page number.
        page_text: Existing text from fast extraction.

    Returns:
        (merged_text, region_count) tuple.
    """
    regions = detect_weak_regions(file_path, page_num)
    if not regions:
        return page_text, 0

    region_texts = []
    for region in regions:
        text = ocr_region(file_path, region)
        region_texts.append(text)

    merged = merge_region_text(page_text, regions, region_texts)
    n_recovered = sum(1 for t in region_texts if t.strip())

    logger.info(
        f"Page {page_num}: region OCR recovered text from "
        f"{n_recovered}/{len(regions)} image regions"
    )

    return merged, n_recovered


def _get_image_bboxes(page: fitz.Page) -> list[fitz.Rect]:
    """Get bounding boxes for all images on the page."""
    bboxes = []
    image_list = page.get_images(full=True)

    for img_info in image_list:
        xref = img_info[0]
        try:
            rects = page.get_image_rects(xref)
            for rect in rects:
                if rect.is_empty or rect.is_infinite:
                    continue
                # Skip tiny images (icons, bullets)
                if rect.width < 50 or rect.height < 50:
                    continue
                bboxes.append(rect)
        except Exception:
            continue

    return bboxes


def _get_text_bboxes(page: fitz.Page) -> list[fitz.Rect]:
    """Get bounding boxes for all text blocks on the page."""
    blocks = page.get_text("blocks")
    bboxes = []

    for block in blocks:
        if block[6] != 0:  # type != text
            continue
        text = block[4].strip()
        if not text:
            continue
        rect = fitz.Rect(block[0], block[1], block[2], block[3])
        bboxes.append(rect)

    return bboxes


def _has_significant_text_overlap(
    img_rect: fitz.Rect,
    text_bboxes: list[fitz.Rect],
    *,
    min_overlap_ratio: float = 0.15,
) -> bool:
    """Check if an image region has significant overlap with text blocks.

    "Significant" means at least min_overlap_ratio of the image area
    is covered by text blocks.
    """
    img_area = img_rect.width * img_rect.height
    if img_area <= 0:
        return False

    total_overlap = 0.0
    for text_rect in text_bboxes:
        intersection = img_rect & text_rect
        if intersection.is_empty:
            continue
        total_overlap += intersection.width * intersection.height

    return (total_overlap / img_area) >= min_overlap_ratio
