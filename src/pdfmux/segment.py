"""Segment-level detection — identify regions on a page.

Instead of routing an entire page to one extractor, detect segments
(text blocks, tables, images, formulas) and route each to its specialist.

Used in PREMIUM mode for mixed-content pages where a single extractor
can't handle everything well.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Types of content segments on a PDF page."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"


@dataclass(frozen=True)
class Segment:
    """A detected region on a PDF page."""

    segment_type: SegmentType
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int  # 0-indexed
    text: str = ""
    confidence: float = 0.0
    area: float = 0.0  # bbox area in points²


def detect_segments(file_path: str | Path, page_num: int) -> list[Segment]:
    """Detect content segments on a single PDF page.

    Uses PyMuPDF's block analysis to identify text blocks, images,
    and infer tables from grid patterns.

    Args:
        file_path: Path to the PDF.
        page_num: 0-indexed page number.

    Returns:
        List of Segment objects, sorted top-to-bottom.
    """
    doc = fitz.open(str(file_path))
    if page_num >= len(doc):
        doc.close()
        return []

    page = doc[page_num]
    page_height = page.rect.height
    segments: list[Segment] = []

    # Get text blocks: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")

    for block in blocks:
        x0, y0, x1, y1 = block[:4]
        text = block[4] if len(block) > 4 else ""
        block_type = block[6] if len(block) > 6 else 0
        bbox = (x0, y0, x1, y1)
        area = (x1 - x0) * (y1 - y0)

        if block_type == 1:
            # Image block
            segments.append(
                Segment(
                    segment_type=SegmentType.IMAGE,
                    bbox=bbox,
                    page_num=page_num,
                    area=area,
                )
            )
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()

        # Classify text block
        seg_type = _classify_block(text, bbox, page_height)
        segments.append(
            Segment(
                segment_type=seg_type,
                bbox=bbox,
                page_num=page_num,
                text=text,
                area=area,
            )
        )

    # Detect tables from structured patterns
    table_segments = _detect_table_regions(page, page_num)
    if table_segments:
        # Merge: replace text segments that overlap with table regions
        segments = _merge_table_segments(segments, table_segments)

    doc.close()

    # Sort top-to-bottom, left-to-right
    segments.sort(key=lambda s: (s.bbox[1], s.bbox[0]))

    return segments


def detect_segments_all_pages(file_path: str | Path) -> dict[int, list[Segment]]:
    """Detect segments on all pages of a PDF.

    Returns:
        Dict mapping page_num → list of segments.
    """
    doc = fitz.open(str(file_path))
    page_count = len(doc)
    doc.close()

    result = {}
    for page_num in range(page_count):
        segs = detect_segments(file_path, page_num)
        if segs:
            result[page_num] = segs

    return result


def page_segment_summary(segments: list[Segment]) -> dict[str, int]:
    """Summarize segment types on a page.

    Returns:
        Dict of {segment_type: count}.
    """
    counts: dict[str, int] = {}
    for seg in segments:
        key = seg.segment_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def is_mixed_content(segments: list[Segment]) -> bool:
    """Check if a page has mixed content (multiple segment types).

    Returns True if the page has at least 2 different non-text segment types,
    or text + tables/images.
    """
    types = {s.segment_type for s in segments}
    types.discard(SegmentType.HEADER)
    types.discard(SegmentType.FOOTER)

    if len(types) >= 3:
        return True
    if SegmentType.TABLE in types and SegmentType.TEXT in types:
        return True
    if SegmentType.IMAGE in types and SegmentType.TEXT in types:
        return True
    return False


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _classify_block(
    text: str, bbox: tuple[float, float, float, float], page_height: float
) -> SegmentType:
    """Classify a text block into a segment type."""
    x0, y0, x1, y1 = bbox

    # Header: top 10% of page, short text
    if y0 < page_height * 0.10 and len(text) < 200:
        return SegmentType.HEADER

    # Footer: bottom 10% of page, short text
    if y1 > page_height * 0.90 and len(text) < 200:
        return SegmentType.FOOTER

    # Table: detect pipe-delimited or tab-separated patterns
    if _looks_like_table(text):
        return SegmentType.TABLE

    # Formula: detect math patterns
    if _looks_like_formula(text):
        return SegmentType.FORMULA

    return SegmentType.TEXT


def _looks_like_table(text: str) -> bool:
    """Heuristic: does this text block look like a table?"""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False

    # Check for pipe-delimited rows
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    if pipe_lines >= 2:
        return True

    # Check for tab-separated data
    tab_lines = sum(1 for line in lines if line.count("\t") >= 2)
    if tab_lines >= 2:
        return True

    # Check for consistent column alignment (multiple number columns)
    number_lines = sum(1 for line in lines if len(re.findall(r"\d+[.,]\d+|\$\d+|\d+%", line)) >= 2)
    if number_lines >= 3 and number_lines / len(lines) > 0.5:
        return True

    return False


def _looks_like_formula(text: str) -> bool:
    """Heuristic: does this text block contain math formulas?"""
    # LaTeX markers
    if re.search(r"\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\gamma", text):
        return True

    # Dense math symbols
    math_chars = sum(1 for c in text if c in "∑∫∏√∞≈≠≤≥±×÷∂∇")
    if math_chars > 3:
        return True

    # Equation patterns
    if re.search(r"[a-z]\s*=\s*[a-z0-9].*[+\-*/^]", text, re.IGNORECASE):
        # Only if it's short (not a paragraph with = sign)
        if len(text) < 200:
            return True

    return False


def _detect_table_regions(page: fitz.Page, page_num: int) -> list[Segment]:
    """Use PyMuPDF's table finder to detect table regions."""
    try:
        tables = page.find_tables()
        if not tables or not tables.tables:
            return []

        segments = []
        for table in tables.tables:
            bbox = (table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3])
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # Extract table text
            try:
                cells = table.extract()
                text = "\n".join(" | ".join(str(c) if c else "" for c in row) for row in cells)
            except Exception:
                text = ""

            segments.append(
                Segment(
                    segment_type=SegmentType.TABLE,
                    bbox=bbox,
                    page_num=page_num,
                    text=text,
                    area=area,
                )
            )

        return segments
    except Exception:
        return []


def _merge_table_segments(
    text_segments: list[Segment],
    table_segments: list[Segment],
) -> list[Segment]:
    """Replace text segments that overlap with detected table regions."""
    result = []

    for seg in text_segments:
        overlaps_table = False
        for table in table_segments:
            if _bbox_overlap(seg.bbox, table.bbox) > 0.5:
                overlaps_table = True
                break
        if not overlaps_table:
            result.append(seg)

    # Add table segments
    result.extend(table_segments)

    return result


def _bbox_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute overlap ratio between two bounding boxes.

    Returns fraction of box A that overlaps with box B (0.0 to 1.0).
    """
    x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap_area = x_overlap * y_overlap

    a_area = (a[2] - a[0]) * (a[3] - a[1])
    if a_area <= 0:
        return 0.0

    return overlap_area / a_area
