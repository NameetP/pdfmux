"""Table cells must read the same no matter what pymupdf4llm did to PyMuPDF's global state.

Importing pymupdf4llm calls ``pymupdf.TOOLS.unset_quad_corrections(True)``; after that, and until
something calls ``pymupdf4llm.to_markdown()``, ``find_tables().extract()`` returns "37.50" as
"3750\\n." (a silent 100x). These tests reproduce exactly that window and assert every pdfmux
reader of table cells is immune. See src/pdfmux/table_cells.py.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pymupdf4llm  # noqa: F401 — imported on purpose: this is what changes the global state
import pytest

from pdfmux.extractors.fast import _extract_tables_fast
from pdfmux.segment import _detect_table_regions
from pdfmux.table_cells import table_cell_texts

ROWS = [
    ["Date", "Description", "Debit", "Credit", "Balance"],
    ["01/09/2026", "Card 0", "37.50", "", "962.50"],
    ["02/09/2026", "Deposit 1", "", "101.00", "1,063.50"],
    ["03/09/2026", "Card 2", "39.50", "", "1,024.00"],
]


def _ruled_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 40), "Statement of account", fontsize=12)
    for r, row in enumerate(ROWS):
        for c, cell in enumerate(row):
            page.insert_text((43 + c * 110, 73 + r * 18), cell, fontsize=9)
    for r in range(len(ROWS) + 1):
        page.draw_line((40, 60 + r * 18), (590, 60 + r * 18))
    for c in range(len(ROWS[0]) + 1):
        page.draw_line((40 + c * 110, 60), (40 + c * 110, 60 + len(ROWS) * 18))
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def broken_global_state():
    """Put PyMuPDF in the state pymupdf4llm leaves it in on import, and undo it afterwards."""
    fitz.TOOLS.unset_quad_corrections(True)
    saved_flags = fitz.table.FLAGS
    yield
    fitz.table.FLAGS = saved_flags


@pytest.fixture
def page(tmp_path: Path, broken_global_state):
    doc = fitz.open(_ruled_pdf(tmp_path / "s.pdf"))
    yield doc[0]
    doc.close()


def test_upstream_extract_is_corrupted_in_this_state(page: fitz.Page) -> None:
    """Sentinel. If this starts failing, PyMuPDF/pymupdf4llm changed behaviour; the helper is
    still correct, but the comments in table_cells.py should be revisited."""
    raw = page.find_tables().tables[0].extract()
    if raw[1][2] == "37.50":
        pytest.skip("upstream extract() no longer corrupts cells in this state")
    assert raw[1][2] != "37.50"


def test_helper_reads_exact_cells(page: fitz.Page) -> None:
    rows = table_cell_texts(page, page.find_tables().tables[0])
    assert rows == ROWS


def test_fast_extractor_tables_are_exact(page: fitz.Page) -> None:
    _, tables = _extract_tables_fast(page, 0, "")
    assert len(tables) == 1
    t = tables[0]
    assert list(t.headers) == ROWS[0]
    assert [list(r) for r in t.rows] == ROWS[1:]


def test_segment_table_text_is_exact(page: fitz.Page) -> None:
    segs = _detect_table_regions(page, 0)
    assert segs and "37.50" in segs[0].text and "Card 0" in segs[0].text
    assert "3750" not in segs[0].text
