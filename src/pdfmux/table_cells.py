"""Read the text of a PyMuPDF ``find_tables()`` table, cell by cell, independent of global state.

Why not ``table.extract()``: its output depends on PyMuPDF *process-wide* settings that another
library changes behind our back.

* Importing ``pymupdf4llm`` (a core pdfmux dependency, imported by ``import pdfmux``) calls
  ``pymupdf.TOOLS.unset_quad_corrections(True)``. From then on, ``extract()`` on a ruled table
  returns "37.50" as ``"3750\\n."`` and "Card 0" as ``"Card0"``; after whitespace cleanup that is
  ``3750``: a silent 100x on every amount.
* ``pymupdf4llm.to_markdown()`` later overwrites ``pymupdf.table.FLAGS``, which happens to make
  ``extract()`` correct again. So the same page gives right or wrong numbers depending only on
  whether something called ``to_markdown()`` earlier in the process.

Verified on PyMuPDF 1.27.1 / pymupdf4llm 0.3.4 (2026-10-02). ``page.get_textbox(cell_rect)`` returns
the laid-out text ("37.50", "Card 0") in every one of those states, so every pdfmux reader of
``find_tables()`` cells goes through this function. Pinned by tests/test_table_cells.py.
"""

from __future__ import annotations

import re
from typing import Any

import fitz

_WS = re.compile(r"\s+")


def table_cell_texts(page: fitz.Page, table: Any) -> list[list[str]]:
    """Return the table as rows of cleaned cell strings ("" for empty or merged-away cells)."""
    rows: list[list[str]] = []
    for row in table.rows:
        out: list[str] = []
        for bbox in row.cells:
            if bbox is None:
                out.append("")
                continue
            out.append(_WS.sub(" ", page.get_textbox(fitz.Rect(bbox))).strip())
        rows.append(out)
    return rows
