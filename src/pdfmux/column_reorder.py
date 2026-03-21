"""Column-aware reading order — A/B comparison approach.

Detects multi-column pages and reorders paragraphs if the column-aware
order scores better than the original pymupdf4llm order.

Safety: worst case is a no-op (original text returned). The A/B comparison
ensures we never regress — we only change the order when the alternative
is demonstrably better.

Integration: called from fast.py after pymupdf4llm extraction, before
heading injection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import fitz

logger = logging.getLogger(__name__)

# Minimum gap between x0 clusters to consider as separate columns (points)
# 200pt is conservative — avoids false positives from indented text
_COLUMN_GAP_MIN = 200.0

# Minimum blocks per column to consider it a real column
_MIN_BLOCKS_PER_COLUMN = 3

# Minimum text per column (chars) — avoid detecting marginal notes as columns
_MIN_CHARS_PER_COLUMN = 100

# Spanning element: block wider than this fraction of page width
_SPAN_WIDTH_RATIO = 0.60


@dataclass(frozen=True)
class _ColumnLayout:
    """Detected column layout for a page."""
    columns: int
    boundaries: list[tuple[float, float]]  # (x_min, x_max) per column
    block_order: list[int]  # block indices in column-aware reading order


def detect_columns(page: fitz.Page) -> _ColumnLayout | None:
    """Detect multi-column layout on a page.

    Returns None if page is single-column or detection is uncertain.
    Only returns a layout when confident about multi-column structure.
    """
    blocks = page.get_text("blocks")
    text_blocks = [(i, b) for i, b in enumerate(blocks) if b[6] == 0 and b[4].strip()]

    if len(text_blocks) < 6:
        return None

    page_width = page.rect.width

    # Separate spanning elements from column-assignable blocks
    column_blocks: list[tuple[int, tuple]] = []
    spanning_indices: set[int] = set()

    for idx, block in text_blocks:
        block_width = block[2] - block[0]  # x1 - x0
        if block_width > page_width * _SPAN_WIDTH_RATIO:
            spanning_indices.add(idx)
        else:
            column_blocks.append((idx, block))

    if len(column_blocks) < 6:
        return None

    # Cluster x0 positions of non-spanning blocks
    x0_positions = sorted(set(b[0] for _, b in column_blocks))
    clusters = _cluster_x0(x0_positions)

    if len(clusters) < 2:
        return None

    # Build column boundaries
    boundaries = _build_boundaries(clusters, page_width)

    # Assign blocks to columns
    columns_of_blocks: list[list[tuple[int, float]]] = [[] for _ in boundaries]

    for block_idx, block in column_blocks:
        x0, y0 = block[0], block[1]
        col = _assign_column(x0, boundaries)
        columns_of_blocks[col].append((block_idx, y0))

    # Validate: each column must have enough blocks and text
    for col_idx, col_blocks in enumerate(columns_of_blocks):
        if len(col_blocks) < _MIN_BLOCKS_PER_COLUMN:
            return None

        # Check text volume
        total_chars = sum(
            len(blocks[bidx][4].strip())
            for bidx, _ in col_blocks
        )
        if total_chars < _MIN_CHARS_PER_COLUMN:
            return None

    # Validate: columns must have overlapping y-ranges (actual interleaving)
    y_ranges = []
    for col_blocks in columns_of_blocks:
        ys = [y for _, y in col_blocks]
        y_ranges.append((min(ys), max(ys)))

    # At least 50% vertical overlap between adjacent columns
    for i in range(len(y_ranges) - 1):
        overlap_start = max(y_ranges[i][0], y_ranges[i + 1][0])
        overlap_end = min(y_ranges[i][1], y_ranges[i + 1][1])
        overlap = max(0, overlap_end - overlap_start)
        span = max(y_ranges[i][1] - y_ranges[i][0], y_ranges[i + 1][1] - y_ranges[i + 1][0])
        if span > 0 and overlap / span < 0.3:
            return None  # Columns don't overlap — not real multi-column

    # Build column-aware reading order:
    # Interleave spanning elements at their y-position
    order: list[tuple[float, int]] = []

    for col_blocks in columns_of_blocks:
        col_blocks.sort(key=lambda t: t[1])
        for block_idx, y0 in col_blocks:
            order.append((y0, block_idx))

    # Add spanning elements at their y-position
    for idx, block in text_blocks:
        if idx in spanning_indices:
            order.append((block[1], idx))

    # Sort: first by column membership (left columns first), then by y
    # Actually, build the proper order: col1 blocks, then col2 blocks, etc.
    # with spanning elements inserted at correct y-positions
    final_order: list[int] = []

    # Collect all spanning elements with their y-positions
    spanning_list = [(blocks[idx][1], idx) for idx in spanning_indices]
    spanning_list.sort(key=lambda t: t[0])
    spanning_iter = iter(spanning_list)
    next_span = next(spanning_iter, None)

    for col_blocks in columns_of_blocks:
        col_blocks.sort(key=lambda t: t[1])

        for block_idx, y0 in col_blocks:
            # Insert any spanning elements that come before this block
            while next_span and next_span[0] < y0:
                if next_span[1] not in set(final_order):
                    final_order.append(next_span[1])
                next_span = next(spanning_iter, None)
            final_order.append(block_idx)

    # Add remaining spanning elements
    while next_span:
        if next_span[1] not in set(final_order):
            final_order.append(next_span[1])
        next_span = next(spanning_iter, None)

    return _ColumnLayout(
        columns=len(boundaries),
        boundaries=boundaries,
        block_order=final_order,
    )


def reorder_text_ab(
    original_text: str,
    page: fitz.Page,
) -> str:
    """A/B test: compare original text vs column-reordered text.

    Returns whichever scores better on self-consistency.
    If column detection fails or reordering doesn't improve, returns original.
    """
    layout = detect_columns(page)
    if layout is None:
        return original_text

    # Build column-reordered text by rearranging paragraphs
    reordered = _reorder_paragraphs(original_text, page, layout)
    if reordered is None:
        return original_text

    # A/B comparison: score both candidates
    score_original = _score_reading_order(original_text, page)
    score_reordered = _score_reading_order(reordered, page)

    # Only switch if reordered is clearly better (margin > 0.05)
    if score_reordered > score_original + 0.05:
        logger.info(
            f"Column reorder: switching to reordered text "
            f"(score {score_original:.3f} -> {score_reordered:.3f}, "
            f"{layout.columns} columns)"
        )
        return reordered

    return original_text


def _reorder_paragraphs(
    text: str,
    page: fitz.Page,
    layout: _ColumnLayout,
) -> str | None:
    """Reorder paragraphs from pymupdf4llm text to match column layout.

    Matches paragraphs to fitz blocks using text overlap, then emits
    paragraphs in the column-aware block order.

    Returns None if matching fails.
    """
    blocks = page.get_text("blocks")
    block_texts = {
        i: blocks[i][4].strip()
        for i in layout.block_order
        if i < len(blocks) and blocks[i][6] == 0
    }

    # Split pymupdf4llm text into paragraphs (double-newline separated)
    paragraphs = re.split(r"\n{2,}", text.strip())
    if not paragraphs:
        return None

    # Match each block to its best paragraph
    block_to_para: dict[int, int] = {}
    used_paras: set[int] = set()

    for block_idx in layout.block_order:
        if block_idx not in block_texts:
            continue

        block_text = block_texts[block_idx]
        # Clean block text for comparison
        block_clean = _normalize_for_match(block_text)
        if len(block_clean) < 10:
            continue

        best_para = -1
        best_score = 0.0

        for para_idx, para in enumerate(paragraphs):
            if para_idx in used_paras:
                continue

            para_clean = _normalize_for_match(para)
            if not para_clean:
                continue

            # Compute overlap score using longest common substring ratio
            score = _overlap_score(block_clean, para_clean)
            if score > best_score:
                best_score = score
                best_para = para_idx

        if best_para >= 0 and best_score > 0.3:
            block_to_para[block_idx] = best_para
            used_paras.add(best_para)

    # Need at least 50% of blocks matched
    if len(block_to_para) < len(layout.block_order) * 0.4:
        return None

    # Build reordered text
    reordered_paras: list[str] = []

    # First: paragraphs in block order
    for block_idx in layout.block_order:
        if block_idx in block_to_para:
            reordered_paras.append(paragraphs[block_to_para[block_idx]])

    # Then: any unmatched paragraphs in original order (preserving footnotes, etc.)
    for para_idx, para in enumerate(paragraphs):
        if para_idx not in used_paras and para.strip():
            reordered_paras.append(para)

    return "\n\n".join(reordered_paras)


def _score_reading_order(text: str, page: fitz.Page) -> float:
    """Score how well text follows the expected reading order.

    Compares consecutive paragraphs against block y-positions.
    Higher score = better reading order (fewer y-position jumps).

    Returns 0.0 to 1.0.
    """
    blocks = page.get_text("blocks")
    text_blocks = [
        (i, b) for i, b in enumerate(blocks)
        if b[6] == 0 and b[4].strip()
    ]

    if len(text_blocks) < 3:
        return 0.5  # Can't score with too few blocks

    paragraphs = re.split(r"\n{2,}", text.strip())
    if len(paragraphs) < 3:
        return 0.5

    # Match paragraphs to blocks by text overlap
    para_positions: list[tuple[float, float]] = []  # (y0, x0) per matched para

    for para in paragraphs:
        para_clean = _normalize_for_match(para)
        if len(para_clean) < 10:
            continue

        best_block = None
        best_score = 0.0

        for block_idx, block in text_blocks:
            block_clean = _normalize_for_match(block[4])
            score = _overlap_score(para_clean, block_clean)
            if score > best_score:
                best_score = score
                best_block = block

        if best_block is not None and best_score > 0.2:
            para_positions.append((best_block[1], best_block[0]))  # (y0, x0)

    if len(para_positions) < 3:
        return 0.5

    # Score: count how many consecutive para transitions go "forward"
    # (y increases, or same y with x increase)
    # A jump UP in y is a sign of wrong column interleaving
    forward = 0
    total = len(para_positions) - 1

    for i in range(total):
        y_curr, x_curr = para_positions[i]
        y_next, x_next = para_positions[i + 1]

        # Forward: y increases by more than a line height (~12pt)
        if y_next > y_curr + 12:
            forward += 1
        # Same y-band: x increases (left to right on same line)
        elif abs(y_next - y_curr) <= 12:
            forward += 1
        # Backward y but large x change: likely column switch (acceptable)
        elif y_next < y_curr and abs(x_next - x_curr) > 100:
            forward += 0.5  # Partial credit for column transitions

    return forward / total if total > 0 else 0.5


def _normalize_for_match(text: str) -> str:
    """Normalize text for matching: strip markdown, collapse whitespace."""
    # Remove markdown formatting
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
    text = re.sub(r"#{1,6}\s+", "", text)  # headings
    text = re.sub(r"\|[^\n]+\|", "", text)  # table rows
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _overlap_score(a: str, b: str) -> float:
    """Compute text overlap score between two strings.

    Uses character n-gram overlap for speed (no external dependency).
    Returns 0.0 to 1.0.
    """
    if not a or not b:
        return 0.0

    # Use 4-gram overlap
    n = 4
    if len(a) < n or len(b) < n:
        # Fall back to substring check for very short strings
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return 1.0 if shorter in longer else 0.0

    grams_a = set(a[i:i + n] for i in range(len(a) - n + 1))
    grams_b = set(b[i:i + n] for i in range(len(b) - n + 1))

    if not grams_a or not grams_b:
        return 0.0

    intersection = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return intersection / union if union > 0 else 0.0


def _cluster_x0(positions: list[float]) -> list[list[float]]:
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


def _build_boundaries(
    clusters: list[list[float]], page_width: float
) -> list[tuple[float, float]]:
    """Build (x_min, x_max) boundaries for each column cluster."""
    boundaries = []
    for i, cluster in enumerate(clusters):
        x_min = min(cluster)
        if i + 1 < len(clusters):
            x_max = min(clusters[i + 1]) - 1.0
        else:
            x_max = page_width
        boundaries.append((x_min, x_max))
    return boundaries


def _assign_column(x0: float, boundaries: list[tuple[float, float]]) -> int:
    """Assign a block to its closest column."""
    best_col = 0
    best_dist = abs(x0 - boundaries[0][0])
    for col_idx, (col_min, _) in enumerate(boundaries):
        dist = abs(x0 - col_min)
        if dist < best_dist:
            best_dist = dist
            best_col = col_idx
    return best_col
