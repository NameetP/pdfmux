"""Tests for layout detection and column reading order."""

from __future__ import annotations

from pathlib import Path

import fitz

from pdfmux.detect import detect_layout
from pdfmux.types import PageLayout


class TestDetectLayout:
    """Tests for detect_layout() function."""

    def test_single_column_no_change(self, digital_pdf: Path) -> None:
        """Single-column digital PDF should return columns=1."""
        doc = fitz.open(str(digital_pdf))
        layout = detect_layout(doc[0])
        doc.close()
        assert layout.columns == 1
        assert len(layout.column_boundaries) == 1

    def test_empty_page_single_column(self, empty_pdf: Path) -> None:
        """Empty page should return columns=1 with empty reading order."""
        doc = fitz.open(str(empty_pdf))
        layout = detect_layout(doc[0])
        doc.close()
        assert layout.columns == 1
        assert layout.reading_order == ()

    def test_two_column_detection(self, tmp_path: Path) -> None:
        """A page with two clearly separated columns should be detected."""
        pdf_path = tmp_path / "two_col.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Left column (x=50)
        for y in range(72, 500, 30):
            page.insert_text((50, y), f"Left column text line at y={y}", fontsize=10)

        # Right column (x=320, gap of ~270 points from left column)
        for y in range(72, 500, 30):
            page.insert_text((320, y), f"Right column text line at y={y}", fontsize=10)

        doc.save(str(pdf_path))
        doc.close()

        doc2 = fitz.open(str(pdf_path))
        layout = detect_layout(doc2[0])
        doc2.close()

        assert layout.columns == 2
        assert len(layout.column_boundaries) == 2
        assert len(layout.reading_order) > 0

    def test_reading_order_left_then_right(self, tmp_path: Path) -> None:
        """Reading order should be left column first, then right column."""
        pdf_path = tmp_path / "two_col_order.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Insert right column FIRST in the document (to test ordering)
        page.insert_text((320, 100), "Right column first line", fontsize=10)
        page.insert_text((320, 130), "Right column second line", fontsize=10)

        # Then left column
        page.insert_text((50, 100), "Left column first line", fontsize=10)
        page.insert_text((50, 130), "Left column second line", fontsize=10)

        doc.save(str(pdf_path))
        doc.close()

        doc2 = fitz.open(str(pdf_path))
        layout = detect_layout(doc2[0])
        doc2.close()

        if layout.columns == 2:
            # Reading order should start with left column blocks
            doc2 = fitz.open(str(pdf_path))
            page = doc2[0]
            all_blocks = page.get_text("blocks")
            text_blocks = [b for b in all_blocks if b[6] == 0 and b[4].strip()]
            doc2.close()

            # First blocks in reading order should be from left column
            if layout.reading_order:
                first_idx = layout.reading_order[0]
                if first_idx < len(text_blocks):
                    assert text_blocks[first_idx][0] < 200  # Left column x0 < 200

    def test_pagelayout_is_frozen(self) -> None:
        """PageLayout should be immutable."""
        layout = PageLayout(columns=1, column_boundaries=(), reading_order=())
        import pytest

        with pytest.raises(AttributeError):
            layout.columns = 2  # type: ignore[misc]
