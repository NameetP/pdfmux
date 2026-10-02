"""Deterministic table extraction for the remote ``extract_tables`` tool (ChatGPT plugin).

Why a separate path from the main pipeline: the router may fall through to paid extractors
(Mistral OCR, LLM) on table documents. This path never does. It uses PyMuPDF's
``find_tables()`` plus pdfmux's whitespace fallback, with no models and no network, so a call
costs only CPU and the plugin's spend is bounded by construction.

What it adds on top of raw cells, because the target documents are bank statements, invoices
and reports where "the numbers are right" is the whole job:

* **Stitching.** A table that continues across pages becomes one table; repeated header rows on
  continuation pages are dropped.
* **Typing.** A column whose cells are (almost all) amounts becomes numbers: ``(1,234.50)``,
  ``1,234.50 DR``, ``-1234.5``, ``1.234,50`` (EU) and ``1,23,456.78`` (Indian grouping) all parse.
  Dates become ISO dates only when the column's format is unambiguous; ``03/04/2026`` stays text
  if nothing in the column says whether it is day-first or month-first.
* **Reconciliation.** If a table has a balance column plus debit/credit (or a signed amount)
  column, each row is checked: previous balance ± movement = this balance. That is the
  "nothing skipped" proof: a dropped row breaks the chain at exactly that point.
* **Honesty about scans.** Pages with no text layer are reported as scanned instead of silently
  yielding an empty spreadsheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import fitz

from pdfmux.table_cells import table_cell_texts
from pdfmux.table_fallback import detect_text_tables

MIN_SCANNED_CHARS = 25
NUMERIC_SHARE = 0.8
DATE_SHARE = 0.8
TOLERANCE = 0.011

# ---------------------------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------------------------

# ISO codes are matched UPPERCASE only ("abc123" must not read as currency "abc" + 123); the
# word forms Rs/Dhs are matched in any case.
_CURRENCY = re.compile(
    r"^(?:[A-Z]{3}|US\$|\$|€|£|¥|₹|(?i:rs)\.?|(?i:dhs)\.?|د\.إ)\s*|\s*(?:[A-Z]{3}|\$|€|£|¥|₹)$"
)
_DIRECTION = re.compile(r"(?i)\s*\b(DR|CR|Dr\.?|Cr\.?)$")


def parse_amount(raw: str) -> float | None:
    """Parse a money-like cell to a float, or None if it isn't one.

    Debit markers (DR, trailing minus, parentheses) make the value negative; CR keeps it positive.
    Grouping is resolved from the separators actually present, not from a locale guess.
    """
    s = raw.strip().replace("−", "-").replace("\xa0", " ")
    if not s or len(s) > 40:
        return None
    negative = False
    m = _DIRECTION.search(s)
    if m:
        negative = m.group(1).upper().startswith("DR")
        s = s[: m.start()].strip()
    for _ in range(2):
        s = _CURRENCY.sub("", s).strip()
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    if s.endswith("-"):
        negative, s = True, s[:-1].strip()
    if s.startswith("-"):
        negative, s = True, s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    s = s.replace(" ", "")
    if not re.fullmatch(r"[0-9][0-9.,']*", s):
        return None
    s = s.replace("'", "")
    if "," in s and "." in s:
        # Whichever separator comes last is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        # "1,234" / "1,23,456" are grouping; "12,5" / "1234,56" are EU decimals.
        if len(tail) == 3 and re.fullmatch(r"[0-9]{1,3}(,[0-9]{2,3})*", s):
            s = s.replace(",", "")
        elif len(tail) in (1, 2) and "," not in head:
            s = head + "." + tail
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1:
        # "1.234.567" is EU grouping.
        s = s.replace(".", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


_MONTHS = {
    m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)
}
_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$")
_TEXT_DATE = re.compile(r"^(\d{1,2})[ \-]([A-Za-z]{3,9})[ \-,]*(\d{2,4})$")


def _year(y: str) -> int:
    n = int(y)
    return n + 2000 if n < 100 else n


def _date_or_none(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date_column(cells: list[str]) -> list[date | None] | None:
    """Parse a whole column as dates, or return None if it isn't one / its order is ambiguous."""
    filled = [c.strip() for c in cells if c.strip()]
    if not filled:
        return None
    numeric = [_NUMERIC_DATE.match(c) for c in filled]
    if all(numeric):
        firsts = [int(m.group(1)) for m in numeric if m]
        seconds = [int(m.group(2)) for m in numeric if m]
        if any(f > 12 for f in firsts):
            order = "dmy"
        elif any(s > 12 for s in seconds):
            order = "mdy"
        else:
            return None  # every value fits both orders: refuse to guess
    else:
        order = None

    out: list[date | None] = []
    hits = 0
    for c in cells:
        c = c.strip()
        d: date | None = None
        if c:
            if m := _ISO.match(c):
                d = _date_or_none(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif (m := _NUMERIC_DATE.match(c)) and order:
                a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
                d = _date_or_none(y, b, a) if order == "dmy" else _date_or_none(y, a, b)
            elif m := _TEXT_DATE.match(c):
                mon = _MONTHS.get(m.group(2)[:3].lower())
                if mon:
                    d = _date_or_none(_year(m.group(3)), mon, int(m.group(1)))
            if d:
                hits += 1
        out.append(d)
    return out if hits / len(filled) >= DATE_SHARE else None


# ---------------------------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------------------------


@dataclass
class Table:
    page_start: int  # 1-based
    page_end: int
    header: list[str]
    rows: list[list[str]]
    column_types: list[str] = field(default_factory=list)  # "number" | "date" | "text"
    values: list[list[Any]] = field(
        default_factory=list
    )  # typed cells (float, ISO date str, str, None)
    reconciliation: dict[str, Any] | None = None

    @property
    def width(self) -> int:
        return len(self.header)


@dataclass
class TablesResult:
    pages: int
    tables: list[Table]
    scanned_pages: list[int]
    truncated: bool


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------


def _clean(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _is_garbage(rows: list[list[str]]) -> bool:
    """PyMuPDF 1.27 find_tables() sometimes returns single-cell or mostly-empty 'tables'."""
    if len(rows) < 2:
        return True
    width = max(len(r) for r in rows)
    if width < 2:
        return True
    cells = [c for r in rows for c in r]
    empty = sum(1 for c in cells if not c)
    return empty / max(len(cells), 1) > 0.6


def _looks_like_header(row: list[str]) -> bool:
    filled = [c for c in row if c]
    if not filled:
        return False
    return sum(1 for c in filled if parse_amount(c) is None) / len(filled) >= 0.75


def _page_tables(page: fitz.Page, page_no: int) -> list[Table]:
    out: list[Table] = []
    try:
        found = page.find_tables().tables
    except Exception:
        found = []
    for t in found:
        rows = table_cell_texts(page, t)
        rows = [r for r in rows if any(r)]
        if _is_garbage(rows):
            continue
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        if _looks_like_header(rows[0]):
            header, body = rows[0], rows[1:]
        else:
            header, body = [f"Column {i + 1}" for i in range(width)], rows
        if body:
            out.append(Table(page_no, page_no, header, body))
    if out:
        return out
    for et in detect_text_tables(page, page_no - 1):
        header = [_clean(h) for h in et.headers]
        body = [[_clean(c) for c in r] for r in et.rows]
        width = max([len(header)] + [len(r) for r in body]) if body else len(header)
        if width < 2 or not body:
            continue
        header = header + [f"Column {i + 1}" for i in range(len(header), width)]
        body = [r + [""] * (width - len(r)) for r in body]
        out.append(Table(page_no, page_no, header, body))
    return out


def _norm(row: list[str]) -> list[str]:
    return [c.lower() for c in row]


def stitch(tables: list[Table]) -> list[Table]:
    """Merge a table into the previous one when it continues it on the next page."""
    merged: list[Table] = []
    for t in tables:
        prev = merged[-1] if merged else None
        continues = (
            prev is not None
            and t.page_start == prev.page_end + 1
            and t.width == prev.width
            and (
                _norm(t.header) == _norm(prev.header)  # header repeated on the new page
                or all(h.startswith("Column ") for h in t.header)  # continuation without a header
            )
        )
        if continues and prev is not None:
            body = t.rows
            if (
                all(h.startswith("Column ") for h in t.header)
                and body
                and _norm(body[0]) == _norm(prev.header)
            ):
                body = body[1:]
            if all(h.startswith("Column ") for h in t.header) and not all(
                h.startswith("Column ") for h in prev.header
            ):
                pass  # keep prev's real header
            prev.rows.extend(body)
            prev.page_end = t.page_end
        else:
            merged.append(t)
    # Drop header rows repeated mid-table (some statements reprint the header every N rows).
    for t in merged:
        hdr = _norm(t.header)
        t.rows = [r for r in t.rows if _norm(r) != hdr]
    return merged


def type_columns(t: Table) -> None:
    cols = list(zip(*t.rows)) if t.rows else [() for _ in t.header]
    types: list[str] = []
    typed_cols: list[list[Any]] = []
    for col in cols:
        cells = list(col)
        filled = [c for c in cells if c]
        amounts = [parse_amount(c) for c in cells]
        if (
            filled
            and sum(1 for c, a in zip(cells, amounts) if c and a is not None) / len(filled)
            >= NUMERIC_SHARE
        ):
            types.append("number")
            # A filled cell that didn't parse stays as its text, so nothing is silently dropped.
            typed_cols.append([a if a is not None else (c or None) for c, a in zip(cells, amounts)])
            continue
        dates = parse_date_column(cells)
        if dates is not None:
            types.append("date")
            typed_cols.append([d.isoformat() if d else (c or None) for c, d in zip(cells, dates)])
            continue
        types.append("text")
        typed_cols.append([c or None for c in cells])
    t.column_types = types
    t.values = [list(r) for r in zip(*typed_cols)] if typed_cols and t.rows else []


# ---------------------------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------------------------

_BAL = re.compile(r"(?i)\bbal(ance)?\b")
_DEBIT = re.compile(r"(?i)\b(debit|debits|withdrawal|withdrawals|paid out|money out|dr)\b")
_CREDIT = re.compile(r"(?i)\b(credit|credits|deposit|deposits|paid in|money in|cr)\b")
_AMOUNT = re.compile(r"(?i)\b(amount|amt|transaction amount)\b")


def _col(t: Table, pattern: re.Pattern[str]) -> int | None:
    for i, (h, ty) in enumerate(zip(t.header, t.column_types)):
        if ty == "number" and pattern.search(h):
            return i
    return None


def reconcile(t: Table) -> dict[str, Any] | None:
    """Check previous balance ± movement = balance on every row. None if the table has no
    balance column or no movement column to check against."""
    bal = _col(t, _BAL)
    if bal is None:
        return None
    debit, credit, amount = _col(t, _DEBIT), _col(t, _CREDIT), _col(t, _AMOUNT)
    if debit == bal:
        debit = None
    if credit == bal:
        credit = None
    if debit is None and credit is None and amount is None:
        return None

    def movement(row: list[Any], sign: int) -> float | None:
        def num(i: int | None) -> float:
            v = row[i] if i is not None else None
            return float(v) if isinstance(v, int | float) else 0.0

        if debit is not None or credit is not None:
            return sign * (num(credit) - abs(num(debit)))
        v = row[amount] if amount is not None else None
        return sign * float(v) if isinstance(v, int | float) else None

    rows = t.values
    best: dict[str, Any] | None = None
    # Some statements list newest first; try both directions and both sign conventions.
    for order in ("asc", "desc"):
        seq = rows if order == "asc" else list(reversed(rows))
        for sign in (1, -1):
            checked = verified = 0
            breaks: list[int] = []
            prev_bal: float | None = None
            for idx, row in enumerate(seq):
                b = row[bal]
                if not isinstance(b, int | float):
                    continue
                mv = movement(row, sign)
                if prev_bal is not None and mv is not None:
                    checked += 1
                    if abs(prev_bal + mv - b) <= TOLERANCE:
                        verified += 1
                    else:
                        breaks.append(idx + 1 if order == "asc" else len(seq) - idx)
                prev_bal = b
            if checked and (best is None or verified > best["rows_verified"]):
                best = {
                    "rows_checked": checked,
                    "rows_verified": verified,
                    "first_breaks": sorted(breaks)[:5],
                    "balance_column": t.header[bal],
                }
    return best


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def extract_tables(path: str, max_pages: int = 30) -> TablesResult:
    doc = fitz.open(path)
    try:
        if doc.needs_pass:
            raise ValueError("This PDF is password-protected.")
        total = doc.page_count
        n = min(total, max_pages)
        raw: list[Table] = []
        scanned: list[int] = []
        for i in range(n):
            page = doc[i]
            if len(page.get_text("text").strip()) < MIN_SCANNED_CHARS and page.get_images(
                full=False
            ):
                scanned.append(i + 1)
                continue
            raw.extend(_page_tables(page, i + 1))
        tables = stitch(raw)
        for t in tables:
            type_columns(t)
            t.reconciliation = reconcile(t)
        return TablesResult(pages=n, tables=tables, scanned_pages=scanned, truncated=total > n)
    finally:
        doc.close()


def result_to_dict(r: TablesResult) -> dict[str, Any]:
    return {
        "pages": r.pages,
        "truncated": r.truncated,
        "scanned_pages": r.scanned_pages,
        "tables": [
            {
                "page_start": t.page_start,
                "page_end": t.page_end,
                "columns": [{"name": h, "type": ty} for h, ty in zip(t.header, t.column_types)],
                "rows": t.values,
                "row_count": len(t.values),
                "reconciliation": t.reconciliation,
            }
            for t in r.tables
        ],
    }
