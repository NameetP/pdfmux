"""Remote extract_tables engine: parsing, stitching, typing, reconciliation, scans.

Fixture PDFs are drawn on the fly with ruled grids so find_tables() sees real tables, and the
truth for every cell is known exactly.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdfmux.remote.tables import (
    extract_tables,
    parse_amount,
    parse_date_column,
    result_to_dict,
)

# ---------------------------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------------------------

COL_W, ROW_H, X0, Y0 = 110, 18, 40, 60


def _draw_table(page: fitz.Page, rows: list[list[str]]) -> None:
    ncols = len(rows[0])
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = X0 + c * COL_W, Y0 + r * ROW_H
            page.insert_text((x + 3, y + 13), cell, fontsize=9)
    for r in range(len(rows) + 1):
        y = Y0 + r * ROW_H
        page.draw_line((X0, y), (X0 + ncols * COL_W, y))
    for c in range(ncols + 1):
        x = X0 + c * COL_W
        page.draw_line((x, Y0), (x, Y0 + len(rows) * ROW_H))


def make_pdf(tmp_path: Path, pages: list[list[list[str]]], name: str = "t.pdf") -> str:
    doc = fitz.open()
    for rows in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((40, 40), "Statement of account", fontsize=12)
        _draw_table(page, rows)
    p = tmp_path / name
    doc.save(p)
    doc.close()
    return str(p)


HEADER = ["Date", "Description", "Debit", "Credit", "Balance"]


def statement_rows(n: int, start: float = 1000.0) -> list[list[str]]:
    """A consistent running-balance statement: credits on odd rows, debits on even."""
    rows, bal = [], start
    for i in range(n):
        if i % 2:
            amt = 100.0 + i
            bal += amt
            rows.append(
                [f"{(i % 28) + 1:02d}/09/2026", f"Deposit {i}", "", f"{amt:,.2f}", f"{bal:,.2f}"]
            )
        else:
            amt = 37.5 + i
            bal -= amt
            rows.append(
                [f"{(i % 28) + 1:02d}/09/2026", f"Card {i}", f"{amt:,.2f}", "", f"{bal:,.2f}"]
            )
    return rows


# ---------------------------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234.50", 1234.5),
        ("(1,234.50)", -1234.5),
        ("1,234.50 DR", -1234.5),
        ("1,234.50 CR", 1234.5),
        ("1,234.50-", -1234.5),
        ("-1234.5", -1234.5),
        ("AED 1,234.50", 1234.5),
        ("$12.00", 12.0),
        ("£1,000", 1000.0),
        ("1.234,50", 1234.5),  # EU
        ("12,5", 12.5),  # EU decimal
        ("1.234.567", 1234567.0),  # EU grouping
        ("1,23,456.78", 123456.78),  # Indian grouping
        ("₹ 1,23,456", 123456.0),
        ("0.00", 0.0),
    ],
)
def test_parse_amount(raw: str, expected: float) -> None:
    assert parse_amount(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "Deposit", "12/09/2026", "N/A", "abc123", "1-2-3", "usd12"])
def test_parse_amount_rejects_non_amounts(raw: str) -> None:
    assert parse_amount(raw) is None


def test_dates_resolve_order_from_the_column_or_refuse() -> None:
    dmy = parse_date_column(["13/09/2026", "01/10/2026"])
    assert (
        dmy is not None
        and dmy[0].isoformat() == "2026-09-13"
        and dmy[1].isoformat() == "2026-10-01"
    )
    mdy = parse_date_column(["09/13/2026", "10/01/2026"])
    assert mdy is not None and mdy[1].isoformat() == "2026-10-01"
    # Every value fits both orders: refuse rather than guess.
    assert parse_date_column(["03/04/2026", "05/06/2026"]) is None
    text = parse_date_column(["5 Sep 2026", "12-Oct-26"])
    assert text is not None and text[1].isoformat() == "2026-10-12"
    assert parse_date_column(["hello", "world"]) is None


# ---------------------------------------------------------------------------------------------
# End to end on real PDFs
# ---------------------------------------------------------------------------------------------


def test_single_page_statement_types_and_reconciles(tmp_path: Path) -> None:
    # 15 rows: days 13-15 are what tell the parser the dates are day-first.
    rows = statement_rows(15)
    r = extract_tables(make_pdf(tmp_path, [[HEADER, *rows]]))
    assert len(r.tables) == 1
    t = r.tables[0]
    assert t.header == HEADER
    assert t.column_types == ["date", "text", "number", "number", "number"]
    assert len(t.values) == 15
    assert t.values[0][0] == "2026-09-01"
    assert isinstance(t.values[0][4], float)
    rec = t.reconciliation
    assert rec is not None and rec["rows_checked"] == 14 and rec["rows_verified"] == 14


def test_multi_page_table_is_stitched_and_repeated_headers_dropped(tmp_path: Path) -> None:
    rows = statement_rows(30)
    pdf = make_pdf(tmp_path, [[HEADER, *rows[:15]], [HEADER, *rows[15:]]])
    r = extract_tables(pdf)
    assert len(r.tables) == 1, [(t.page_start, t.page_end, t.header) for t in r.tables]
    t = r.tables[0]
    assert (t.page_start, t.page_end) == (1, 2)
    assert len(t.values) == 30
    assert all(v[0] != "Date" for v in t.values)
    assert t.reconciliation["rows_verified"] == 29


def test_a_dropped_row_breaks_reconciliation_at_that_row(tmp_path: Path) -> None:
    rows = statement_rows(10)
    del rows[5]  # the "skipped row" pdfmux exists to catch
    t = extract_tables(make_pdf(tmp_path, [[HEADER, *rows]])).tables[0]
    rec = t.reconciliation
    assert rec["rows_verified"] == rec["rows_checked"] - 1
    assert rec["first_breaks"] == [6]


def test_tables_without_a_balance_have_no_reconciliation(tmp_path: Path) -> None:
    rows = [["Item", "Qty", "Price"], ["Widget", "2", "10.00"], ["Gadget", "1", "25.50"]]
    t = extract_tables(make_pdf(tmp_path, [rows])).tables[0]
    assert t.column_types == ["text", "number", "number"]
    assert t.reconciliation is None


def test_scanned_page_is_reported_not_silently_empty(tmp_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(50, 50, 300, 300), pixmap=pix)
    p = tmp_path / "scan.pdf"
    doc.save(p)
    r = extract_tables(str(p))
    assert r.scanned_pages == [1]
    assert r.tables == []


def test_page_cap_truncates_and_says_so(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path, [[["A", "B"], ["1", "2"], ["3", "4"]] for _ in range(4)])
    r = extract_tables(pdf, max_pages=2)
    assert r.pages == 2 and r.truncated is True


def test_result_dict_is_json_ready(tmp_path: Path) -> None:
    import json

    d = result_to_dict(extract_tables(make_pdf(tmp_path, [[HEADER, *statement_rows(4)]])))
    json.dumps(d)
    assert d["tables"][0]["columns"][4] == {"name": "Balance", "type": "number"}
