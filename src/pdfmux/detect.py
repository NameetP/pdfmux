"""PDF type detection — classify a PDF to route it to the best extractor.

Opens the PDF with PyMuPDF. Inspects every page for:
- Text content (character count)
- Embedded images (count + coverage area)
- Line patterns (table detection)
- Text alignment patterns (table detection)
- Column structure (multi-column reading order)
"""

from __future__ import annotations

import re
from collections import Counter
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
    empty_pages: list[int] = field(default_factory=list)


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
    empty_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        text_len = len(text)
        image_count = len(images)

        # Classify into digital / scanned / empty
        if text_len < 20 and image_count == 0:
            empty_pages.append(page_num)
        elif text_len > 50:
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
    result.empty_pages = empty_pages

    total = len(doc)
    if total == 0:
        result.confidence = 0.0
        doc.close()
        return result

    # Exclude empty pages from digital/scanned ratio calculation
    non_empty_total = total - len(empty_pages)
    if non_empty_total == 0:
        # All pages empty — classify as digital
        result.is_digital = True
        result.confidence = 0.5
        doc.close()
        return result

    digital_ratio = len(digital_pages) / non_empty_total

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


# --- Table detection constants ---
_TABLE_SAMPLE_PAGES = 20
_TABLE_SCORE_THRESHOLD = 2
_NUMBER_DENSE_THRESHOLD = 0.30
_ALIGNED_COLUMN_MIN_LINES = 4


def _detect_tables(doc: fitz.Document) -> bool:
    """Multi-signal table detection with strategic page sampling.

    Samples pages from front, middle, and back of document.
    Uses 4 signals scored additively:
        Signal 1: Drawn grid lines — score 2
        Signal 2: Number-dense lines (financial data) — score 2
        Signal 3: Aligned column positions via text blocks — score 2
        Signal 4: Tab/whitespace patterns — score 1

    Returns True if combined score >= _TABLE_SCORE_THRESHOLD.
    """
    total = len(doc)
    if total == 0:
        return False

    sample_pages = _get_sample_pages(total, _TABLE_SAMPLE_PAGES)
    total_score = 0

    for page_num in sample_pages:
        page = doc[page_num]
        page_score = 0
        page_score += _score_drawn_lines(page)
        page_score += _score_number_density(page)
        page_score += _score_column_alignment(page)
        page_score += _score_whitespace_patterns(page)
        page_score += _score_find_tables(page)
        total_score += page_score

        if total_score >= _TABLE_SCORE_THRESHOLD:
            return True

    return total_score >= _TABLE_SCORE_THRESHOLD


def _get_sample_pages(total: int, sample_size: int) -> list[int]:
    """Select pages from front, quarter, middle, three-quarter, and back.

    For large documents (>200 pages), uses wider sampling windows to
    catch tables that may appear only in specific sections.
    """
    if total <= sample_size:
        return list(range(total))

    # Scale sample size for very large documents
    effective_sample = sample_size if total <= 200 else sample_size + (total // 100)
    effective_sample = min(effective_sample, total)

    chunk = effective_sample // 5
    front = list(range(min(chunk, total)))
    q1_start = max(0, total // 4 - chunk // 2)
    q1 = list(range(q1_start, min(q1_start + chunk, total)))
    mid_start = max(0, total // 2 - chunk // 2)
    middle = list(range(mid_start, min(mid_start + chunk, total)))
    q3_start = max(0, 3 * total // 4 - chunk // 2)
    q3 = list(range(q3_start, min(q3_start + chunk, total)))
    back_start = max(0, total - chunk)
    back = list(range(back_start, total))

    return sorted(set(front + q1 + middle + q3 + back))


def _score_drawn_lines(page: fitz.Page) -> int:
    """Score based on drawn horizontal/vertical lines. Returns 0 or 2."""
    drawings = page.get_drawings()
    h_lines = 0
    v_lines = 0
    for drawing in drawings:
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 50:
                    h_lines += 1
                elif abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) > 20:
                    v_lines += 1
    return 2 if (h_lines >= 3 and v_lines >= 2) else 0


def _score_number_density(page: fitz.Page) -> int:
    """Score based on lines dominated by numbers (financial data). Returns 0 or 2."""
    text = page.get_text("text")
    lines = text.split("\n")
    number_dense_lines = 0

    for line in lines:
        stripped = line.strip()
        if len(stripped) < 20:
            continue
        non_space = stripped.replace(" ", "")
        if not non_space:
            continue
        numeric_chars = sum(1 for c in non_space if c in "0123456789$,%.()-")
        ratio = numeric_chars / len(non_space)
        if ratio >= _NUMBER_DENSE_THRESHOLD:
            number_dense_lines += 1

    return 2 if number_dense_lines >= 5 else 0


def _score_column_alignment(page: fitz.Page) -> int:
    """Score based on text blocks with aligned x-positions. Returns 0 or 2."""
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    if len(text_blocks) < 6:
        return 0

    x0_rounded = [round(b[0] / 5) * 5 for b in text_blocks]
    x0_counts = Counter(x0_rounded)
    aligned_columns = sum(1 for count in x0_counts.values() if count >= _ALIGNED_COLUMN_MIN_LINES)

    return 2 if aligned_columns >= 3 else 0


def _score_whitespace_patterns(page: fitz.Page) -> int:
    """Score based on whitespace-separated columns. Returns 0 or 1."""
    text = page.get_text("text")
    lines = text.split("\n")
    tab_lines = sum(
        1 for line in lines
        if len(line.strip()) > 20 and len(re.findall(r"  {3,}", line)) >= 3
    )
    return 1 if tab_lines >= 5 else 0


def _score_find_tables(page: fitz.Page) -> int:
    """Score using PyMuPDF's built-in find_tables() heuristic. Returns 0 or 2."""
    try:
        tables = page.find_tables()
        if tables.tables and len(tables.tables) >= 1:
            return 2
    except (AttributeError, Exception):
        pass
    return 0


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
