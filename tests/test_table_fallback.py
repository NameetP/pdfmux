"""Tests for borderless table fallback detection."""

from __future__ import annotations

import fitz

from pdfmux.table_fallback import (
    _find_column_positions,
    _find_table_regions,
    _has_numeric_column,
    detect_text_tables,
)


def _make_page_with_raw_text(text: str) -> tuple[fitz.Page, int]:
    """Create a PDF page containing the given text.

    Returns (page, page_num=0).
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Insert text line by line to preserve spacing
    y = 72
    for line in text.split("\n"):
        page.insert_text((72, y), line, fontsize=10)
        y += 14
    return page, 0


class TestDetectTextTables:
    """Tests for the main detect_text_tables function."""

    def test_whitespace_aligned_table(self):
        """Lines with consistent column gaps should produce ExtractedTable."""
        text = (
            "Name          Age    Score\n"
            "Alice           25     92.5\n"
            "Bob             30     88.0\n"
            "Charlie         22     95.3\n"
            "Diana           28     91.7\n"
        )
        page, page_num = _make_page_with_raw_text(text)
        tables = detect_text_tables(page, page_num)
        assert len(tables) >= 1
        assert len(tables[0].headers) >= 2
        assert len(tables[0].rows) >= 3

    def test_no_table_returns_empty(self):
        """Normal paragraph text should not be detected as a table."""
        text = (
            "This is a normal paragraph of text that discusses\n"
            "various topics in a flowing manner. There are no\n"
            "columns or tabular structures in this text at all.\n"
            "It just continues as regular prose content.\n"
        )
        page, page_num = _make_page_with_raw_text(text)
        tables = detect_text_tables(page, page_num)
        assert tables == []

    def test_too_few_rows_skipped(self):
        """Fewer than 3 rows should not be detected as a table."""
        text = "Item      Value\nFoo       100\n"
        page, page_num = _make_page_with_raw_text(text)
        tables = detect_text_tables(page, page_num)
        assert tables == []

    def test_empty_page_returns_empty(self):
        doc = fitz.open()
        page = doc.new_page()
        tables = detect_text_tables(page, 0)
        assert tables == []


class TestFindColumnPositions:
    """Tests for column position detection."""

    def test_clear_columns(self):
        lines = [
            "Name          Age    Score",
            "Alice           25     92.5",
            "Bob             30     88.0",
            "Charlie         22     95.3",
        ]
        positions = _find_column_positions(lines)
        assert len(positions) >= 1  # at least one column split

    def test_no_columns_in_prose(self):
        lines = [
            "This is just regular text.",
            "Another line of regular text.",
            "And yet another line here.",
        ]
        positions = _find_column_positions(lines)
        assert len(positions) == 0


class TestFindTableRegions:
    """Tests for table region detection."""

    def test_detects_region(self):
        lines = [
            "Header     Col1    Col2",
            "Row1       100     200",
            "Row2       300     400",
            "Row3       500     600",
            "",
            "Regular paragraph text after the table.",
        ]
        regions = _find_table_regions(lines)
        assert len(regions) >= 1
        start, end = regions[0]
        assert end - start >= 3


class TestHasNumericColumn:
    """Tests for numeric column validation."""

    def test_numeric_column_detected(self):
        rows = [
            ["Name", "Score"],
            ["Alice", "92.5"],
            ["Bob", "88.0"],
            ["Charlie", "95.3"],
        ]
        assert _has_numeric_column(rows) is True

    def test_no_numeric_column(self):
        rows = [
            ["Name", "City"],
            ["Alice", "London"],
            ["Bob", "Paris"],
            ["Charlie", "Tokyo"],
        ]
        assert _has_numeric_column(rows) is False

    def test_currency_detected(self):
        rows = [
            ["Item", "Price"],
            ["Widget", "$10.99"],
            ["Gadget", "$24.50"],
            ["Gizmo", "$7.25"],
        ]
        assert _has_numeric_column(rows) is True
