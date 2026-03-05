"""PDF type detection — classify a PDF to route it to the best extractor.

Opens the PDF with PyMuPDF. Inspects every page for:
- Text content (character count)
- Embedded images (count + coverage area)
- Line patterns (table detection)
- Text alignment patterns (table detection)
- Column structure (multi-column reading order)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.errors import FileError
from pdfmux.types import PageLayout


@dataclass
class PDFClassification:
    """Result of classifying a PDF document."""

    is_digital: bool = False
    is_scanned: bool = False
    is_mixed: bool = False
    is_graphical: bool = False  # Image-heavy — text in images that fast extraction misses
    has_tables: bool = False
    page_count: int = 0
    languages: list[str] = field(default_factory=list)
    confidence: float = 0.0
    digital_pages: list[int] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    graphical_pages: list[int] = field(default_factory=list)


def classify(file_path: str | Path) -> PDFClassification:
    """Classify a PDF to determine the best extraction strategy.

    Args:
        file_path: Path to the PDF file.

    Returns:
        PDFClassification with detection results.

    Raises:
        FileError: If the file doesn't exist or isn't a PDF.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileError(f"PDF not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise FileError(f"Not a PDF file: {file_path}")

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise FileError(f"Cannot open PDF: {file_path} — {e}") from e

    result = PDFClassification(page_count=len(doc))

    digital_pages = []
    scanned_pages = []
    graphical_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        text_len = len(text)
        image_count = len(images)

        # Classify into digital / scanned
        if text_len > 50:
            digital_pages.append(page_num)
        elif images:
            scanned_pages.append(page_num)
        else:
            digital_pages.append(page_num)

        # Detect graphical pages (pitch decks, infographics, slides)
        if (image_count >= 2 and text_len < 500) or (image_count >= 1 and text_len < 100):
            graphical_pages.append(page_num)

    result.digital_pages = digital_pages
    result.scanned_pages = scanned_pages
    result.graphical_pages = graphical_pages

    total = len(doc)
    if total == 0:
        result.confidence = 0.0
        doc.close()
        return result

    digital_ratio = len(digital_pages) / total

    if digital_ratio >= 0.8:
        result.is_digital = True
        result.confidence = min(0.95, digital_ratio)
    elif digital_ratio <= 0.2:
        result.is_scanned = True
        result.confidence = min(0.95, 1 - digital_ratio)
    else:
        result.is_mixed = True
        result.confidence = 0.7

    graphical_ratio = len(graphical_pages) / total
    if graphical_ratio > 0.25:
        result.is_graphical = True

    result.has_tables = _detect_tables(doc)

    doc.close()
    return result


def _detect_tables(doc: fitz.Document) -> bool:
    """Heuristic table detection using line analysis."""
    for page_num in range(min(len(doc), 5)):
        page = doc[page_num]
        drawings = page.get_drawings()

        horizontal_lines = 0
        vertical_lines = 0

        for drawing in drawings:
            for item in drawing.get("items", []):
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 50:
                        horizontal_lines += 1
                    elif abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) > 20:
                        vertical_lines += 1

        if horizontal_lines >= 3 and vertical_lines >= 2:
            return True

    for page_num in range(min(len(doc), 5)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split("\n")
        tab_lines = sum(1 for line in lines if "\t" in line or line.count("  ") >= 3)
        if tab_lines >= 3:
            return True

    return False


# ---------------------------------------------------------------------------
# Layout detection — multi-column reading order
# ---------------------------------------------------------------------------

# Minimum gap between column x-positions to consider them separate columns
_COLUMN_GAP_MIN = 50.0  # points (~0.7 inches)


def detect_layout(page: fitz.Page) -> PageLayout:
    """Detect column structure and reading order for a page.

    Algorithm:
    1. Extract text blocks with bboxes
    2. Cluster block x0 (left-edge) positions with gap detection
    3. 2+ clusters = multi-column
    4. Sort blocks column-by-column (left columns first, top-to-bottom within each)

    Args:
        page: A PyMuPDF page object.

    Returns:
        PageLayout with column count, boundaries, and reading order.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
    text_blocks = [(i, b) for i, b in enumerate(blocks) if b[6] == 0 and b[4].strip()]

    if not text_blocks:
        return PageLayout(columns=1, column_boundaries=(), reading_order=())

    # Cluster x0 positions
    x0_positions = sorted(set(b[0] for _, b in text_blocks))
    column_groups = _cluster_positions(x0_positions)

    if len(column_groups) < 2:
        # Single column — natural order (top to bottom)
        order = tuple(i for i, _ in sorted(text_blocks, key=lambda t: t[1][1]))
        page_width = page.rect.width
        return PageLayout(
            columns=1,
            column_boundaries=((0.0, page_width),),
            reading_order=order,
        )

    # Multi-column: assign blocks to columns, sort within each
    boundaries = _build_column_boundaries(column_groups, page.rect.width)
    reading_order = _build_reading_order(text_blocks, boundaries)

    return PageLayout(
        columns=len(boundaries),
        column_boundaries=tuple(boundaries),
        reading_order=tuple(reading_order),
    )


def _cluster_positions(positions: list[float]) -> list[list[float]]:
    """Cluster sorted x-positions into column groups by gap detection."""
    if not positions:
        return []

    clusters: list[list[float]] = [[positions[0]]]
    for pos in positions[1:]:
        if pos - clusters[-1][-1] > _COLUMN_GAP_MIN:
            clusters.append([pos])
        else:
            clusters[-1].append(pos)

    return clusters


def _build_column_boundaries(
    clusters: list[list[float]], page_width: float
) -> list[tuple[float, float]]:
    """Build (x_min, x_max) boundaries for each column cluster."""
    boundaries = []
    for i, cluster in enumerate(clusters):
        x_min = min(cluster)
        # x_max extends to start of next cluster (or page width)
        if i + 1 < len(clusters):
            x_max = min(clusters[i + 1]) - 1.0
        else:
            x_max = page_width
        boundaries.append((x_min, x_max))
    return boundaries


def _build_reading_order(
    text_blocks: list[tuple[int, tuple]],
    boundaries: list[tuple[float, float]],
) -> list[int]:
    """Assign blocks to columns and sort: left-to-right columns, top-to-bottom within."""
    columned: list[list[tuple[int, float]]] = [[] for _ in boundaries]

    for block_idx, block in text_blocks:
        x0 = block[0]
        y0 = block[1]
        # Find closest column
        best_col = 0
        best_dist = abs(x0 - boundaries[0][0])
        for col_idx, (col_min, col_max) in enumerate(boundaries):
            dist = abs(x0 - col_min)
            if dist < best_dist:
                best_dist = dist
                best_col = col_idx
        columned[best_col].append((block_idx, y0))

    # Sort within each column by y0 (top to bottom)
    order = []
    for col_blocks in columned:
        col_blocks.sort(key=lambda t: t[1])
        order.extend(idx for idx, _ in col_blocks)

    return order
