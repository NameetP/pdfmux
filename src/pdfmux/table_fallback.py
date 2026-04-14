"""Fallback table detection for borderless/whitespace-aligned tables.

When PyMuPDF's ``find_tables()`` returns nothing, this module scans the
page text for column-aligned patterns and reconstructs tables from
whitespace structure. Pure heuristic — no ML, no extra dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import fitz

from pdfmux.types import ExtractedTable

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_text_tables(page: fitz.Page, page_num: int) -> list[ExtractedTable]:
    """Detect borderless tables via whitespace column analysis.

    Only call this when ``page.find_tables()`` returns empty.

    Returns:
        List of ExtractedTable objects (may be empty).
    """
    text = page.get_text("text")
    if not text or len(text.strip()) < 20:
        return []

    lines = text.split("\n")
    tables: list[ExtractedTable] = []

    regions = _find_table_regions(lines)
    for start, end in regions:
        region_lines = lines[start:end]
        col_positions = _find_column_positions(region_lines)
        if len(col_positions) < 1:  # need at least 1 split → 2 columns
            continue

        rows = _split_into_columns(region_lines, col_positions)
        if len(rows) < 3:
            continue

        if not _has_numeric_column(rows):
            continue

        # First row is header
        headers = rows[0]
        data_rows = rows[1:]

        tables.append(
            ExtractedTable(
                page_num=page_num,
                headers=tuple(headers),
                rows=tuple(tuple(r) for r in data_rows),
                bbox=None,
            )
        )

    return tables


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

_MIN_TABLE_ROWS = 3
_MIN_GAP_WIDTH = 2  # minimum consecutive spaces to count as column gap


def _find_table_regions(lines: list[str]) -> list[tuple[int, int]]:
    """Find contiguous line ranges that look table-like.

    A line is "table-like" if it has 2+ internal whitespace gaps of
    ``_MIN_GAP_WIDTH`` spaces and is not too long (likely a paragraph).
    """
    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if in_region and i - start >= _MIN_TABLE_ROWS:
                regions.append((start, i))
            in_region = False
            continue

        gaps = _count_internal_gaps(line)
        if gaps >= 1 and len(stripped) < 200:
            if not in_region:
                start = i
                in_region = True
        else:
            if in_region and i - start >= _MIN_TABLE_ROWS:
                regions.append((start, i))
            in_region = False

    # Handle region at end of text
    if in_region and len(lines) - start >= _MIN_TABLE_ROWS:
        regions.append((start, len(lines)))

    return regions


def _count_internal_gaps(line: str) -> int:
    """Count runs of 2+ spaces inside the line (not leading/trailing)."""
    stripped = line.strip()
    if not stripped:
        return 0
    return len(re.findall(r" {2,}", stripped))


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def _find_column_positions(lines: list[str]) -> list[int]:
    """Find character positions where columns are separated.

    A position is a column boundary if 60%+ of non-empty lines have
    a space at that position while having non-space chars nearby.
    """
    if not lines:
        return []

    # Pad lines to uniform length
    max_len = max(len(line) for line in lines)
    if max_len < 4:
        return []

    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < _MIN_TABLE_ROWS:
        return []

    threshold = len(non_empty) * 0.75  # 75% of lines must agree

    # For each position, count how many lines have a multi-space gap there
    # (not just single spaces, which occur naturally between words)
    gap_counts = [0] * max_len
    char_counts = [0] * max_len

    for line in non_empty:
        padded = line.ljust(max_len)
        in_gap = False
        gap_start = 0
        for pos in range(len(padded)):
            if padded[pos] == " ":
                if not in_gap:
                    gap_start = pos
                    in_gap = True
            else:
                if in_gap and pos - gap_start >= _MIN_GAP_WIDTH:
                    # Mark all positions in this multi-space gap
                    for gp in range(gap_start, pos):
                        gap_counts[gp] += 1
                in_gap = False
                char_counts[pos] += 1

    # Column boundaries: positions where most lines have multi-space gaps,
    # bordered by positions where most lines have characters
    candidates: list[int] = []
    for pos in range(2, max_len - 2):
        if gap_counts[pos] >= threshold:
            # Must have chars on both sides (within 3 positions)
            has_left = any(char_counts[p] > threshold * 0.5 for p in range(max(0, pos - 3), pos))
            has_right = any(  # noqa: E501
                char_counts[p] > threshold * 0.5 for p in range(pos + 1, min(max_len, pos + 4))
            )
            if has_left and has_right:
                candidates.append(pos)

    # Merge adjacent positions into single column boundaries (take center)
    if not candidates:
        return []

    merged: list[int] = []
    group_start = candidates[0]
    prev = candidates[0]

    for pos in candidates[1:]:
        if pos - prev <= 2:
            prev = pos
        else:
            merged.append((group_start + prev) // 2)
            group_start = pos
            prev = pos
    merged.append((group_start + prev) // 2)

    return merged


# ---------------------------------------------------------------------------
# Column splitting
# ---------------------------------------------------------------------------


def _split_into_columns(
    lines: list[str],
    col_positions: list[int],
) -> list[list[str]]:
    """Split each line at column positions into cells."""
    rows: list[list[str]] = []
    boundaries = [0] + col_positions + [None]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        cells: list[str] = []
        for j in range(len(boundaries) - 1):
            start = boundaries[j]
            end = boundaries[j + 1]
            cell = line[start:end].strip() if end else line[start:].strip()
            cells.append(cell)

        # Skip rows that are all empty
        if not any(c for c in cells):
            continue

        rows.append(cells)

    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"[\d$%€£,.]+")


def _has_numeric_column(rows: Sequence[Sequence[str]]) -> bool:
    """Check if at least one column is predominantly numeric.

    A column is "numeric" if 50%+ of its non-empty cells match a
    numeric pattern (digits, currency symbols, percentages).
    """
    if not rows or len(rows) < 2:
        return False

    n_cols = len(rows[0])
    data_rows = rows[1:]  # skip header

    for col_idx in range(n_cols):
        numeric_count = 0
        non_empty = 0
        for row in data_rows:
            if col_idx < len(row) and row[col_idx].strip():
                non_empty += 1
                if _NUMERIC_RE.search(row[col_idx]):
                    numeric_count += 1
        if non_empty >= 2 and numeric_count / max(non_empty, 1) >= 0.5:
            return True

    return False
