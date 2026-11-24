"""Flag OCR readings that a person should check against the original page.

OCR output is searchable text, not a record. The rules here never change the
extracted text. They rank what a reviewer should check first; an unflagged
line is not a verified line.

Checks, all on OCR'd pages:

1. **Low-confidence lines.** RapidOCR scores every line. On degraded scans
   (5-8 pt print, heavy JPEG compression, 100-200 DPI) a 0.92 cutoff caught
   63% of misread lines and flagged 1 of 625 correctly read lines (12 synthetic
   lab-report pages, 755 lines). ``PDFMUX_OCR_MIN_LINE_SCORE`` overrides it.
2. **Dates that cannot be right.** A single wrong digit is often read with high
   confidence (a report dated 2026 came back as 2096), so dates are checked on
   every OCR'd page, scored or not: impossible, before 1900, more than 10 years
   ahead, or later than today where the nearest label before the date is a
   past event ("collected", "DOB", "reported") or there is no label. "next",
   "due", "valid until", "expiry", "appointment" make a future date expected.
3. **Ambiguous dates.** 03/04/2026 is 3 April or March 4. The day/month order
   is read from the whole document: any date like 25/12/2026 settles it as
   day-first, 12/25/2026 as month-first. With no clue, or with both orders in
   one document, each swappable date is flagged.
4. **Suspect numbers.** In a line with a measurement unit, a decimal point read
   as a colon (``8:02 mmol/L``) or a letter inside a number (``1O.2``, ``13.l``)
   is flagged. Opt-in (``PDFMUX_OCR_RANGE_CHECK=1``): a value more than 5x
   outside the reference range printed on its own line (a dropped decimal:
   ``Haemoglobin 132 g/dL 13-17``).
5. **Table cells.** In a table on an OCR'd page, a cell whose type breaks its
   column (a word in a number column) is flagged, and a row whose cells all fit
   once moved one column over is flagged as shifted.

Every flag carries a stable ``id`` and, for RapidOCR lines, the OCR ``line_index``
and ``bbox`` (PDF points, top-left origin of the visible page), so it can be
located on the page image even when the line did not survive into the text body.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

DEFAULT_MIN_LINE_SCORE = 0.92
_SCORE_ENV = "PDFMUX_OCR_MIN_LINE_SCORE"
_RANGE_ENV = "PDFMUX_OCR_RANGE_CHECK"

LOW_CONFIDENCE = "low_confidence"
IMPLAUSIBLE_DATE = "implausible_date"
AMBIGUOUS_DATE = "ambiguous_date"
SUSPECT_NUMBER = "suspect_number"
TABLE_CELL = "table_cell"

REVIEW_NOTE = (
    "Review flags rank what to check first against the original page. "
    "An unflagged line is not a verified line, and extracted text is never corrected."
)

# Dates further ahead than this are flagged even on forward-looking lines.
_MAX_YEARS_AHEAD = 10
_MIN_YEAR = 1900

# Labels that make a future date expected.
_FORWARD_WORDS = re.compile(
    r"\b(next|due|valid|validity|expir\w*|exp|until|till|upto|renew\w*|"
    r"follow[- ]?up|appointment|appt|review|recall|schedul\w*|booked|planned|"
    r"maturity|effective)\b",
    re.IGNORECASE,
)
# Labels that make a future date a misread: the event has already happened.
_PAST_WORDS = re.compile(
    r"\b(collect\w*|sampl\w*|drawn|received|report\w*|tested|test date|issued|"
    r"printed|generated|registered|admi\w*|discharg\w*|visit\w*|dob|d\.o\.b|born|"
    r"birth|dated|signed|verified|authori[sz]ed|approved|performed|done)\b",
    re.IGNORECASE,
)

_MONTHS = {
    name.lower(): i
    for i in range(1, 13)
    for name in (calendar.month_name[i], calendar.month_abbr[i])
}
_MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))

_SEP = r"[/.\-]"
# 12/03/2026, 12-3-2026, 12.03.2026 (day/month order read from the document)
_NUM_DMY = re.compile(rf"(?<![\d/.\-])(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}})(?![\d/.\-]*\d)")
# 2026-03-12, 2026/03/12
_NUM_YMD = re.compile(rf"(?<![\d/.\-])(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?![\d/.\-]*\d)")
# 03/2026, 3-2026 (month/year, as on expiry dates)
_NUM_MY = re.compile(rf"(?<![\d/.\-])(\d{{1,2}}){_SEP}(\d{{4}})(?![\d/.\-]*\d)")
# 12 March 2026, 12-Mar-2026, 12th Mar, 2026
_TXT_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-/.,]*({_MONTH_RE})\.?[\s\-/.,]*(\d{{4}})\b", re.IGNORECASE
)
# March 12, 2026
_TXT_MDY = re.compile(
    rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)
# March 2026, Mar-2026
_TXT_MY = re.compile(rf"\b({_MONTH_RE})\.?[\s\-/,]*(\d{{4}})\b", re.IGNORECASE)

# Measurement units that make a line a result line.
_UNITS = (
    r"g/dl|mg/dl|ng/dl|[uµμ]g/dl|g/l|mg/l|mmol/l|[uµμ]mol/l|nmol/l|pmol/l|meq/l|"
    r"[uµμ]?iu/m?l|miu/m?l|u/l|ng/ml|pg/ml|[uµμ]g/ml|mcg|[uµμ]g|fl|pg|%|mm/hr?|"
    r"mmhg|bpm|/[uµμ]l|/cumm|/hpf|cells|x\s?10\S*|10\^\S*|lakhs?|kg|cm|mg|ml"
)
_UNIT_RE = re.compile(rf"(?<![a-z]){_UNITS}(?![a-z])", re.IGNORECASE)
_UNIT_CELL = re.compile(rf"^\s*(?:{_UNITS})\s*$", re.IGNORECASE)
_TIME_WORDS = re.compile(
    r"\b(time|timing|at|hrs?|hours?|am|pm|a\.m|p\.m|clock|min|mins|minutes|"
    r"titre|titer|ratio|dilution)\b",
    re.IGNORECASE,
)
_COLON_NUM = re.compile(r"(?<![\d:])(\d{1,4}):(\d{1,3})(?![\d:])")
_LETTER_NUM = re.compile(
    r"(?<![A-Za-z\d])(\d+[OolI]+\d*(?:\.\d+)?|\d+\.\d*[OolI]+\d*|[OolI]\.\d+)(?![A-Za-z])"
)
_RANGE = re.compile(r"(?P<lo>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<hi>\d+(?:\.\d+)?)", re.IGNORECASE)
_VALUE_AFTER_LABEL = re.compile(r"[A-Za-z][^\d]*?(?<![\d.])(?P<v>\d+(?:\.\d+)?)(?![\d.:/])")

# Numbers as tables print them: 13.2, <0.9, 4.9 H, (356), $1,234.50, 12%, 45†, 7¹
_NUM_CELL = re.compile(
    r"^\s*[<>≤≥]?\s*[-+−(]?\s*[$€£₹¥]?\s*\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d+)?\s*\)?\s*%?"
    r"\s*[HLhl*↑↓†‡§¹²³⁴⁵⁶⁷⁸⁹⁰]*\s*$|^\s*[<>≤≥]?\s*[-+−(]?\s*[$€£₹¥]?\s*\d+(?:[.,]\d+)?"
    r"\s*\)?\s*%?\s*[HLhl*↑↓†‡§¹²³⁴⁵⁶⁷⁸⁹⁰]*\s*$"
)
# Punctuation and currency fragments left behind by cell splitting: not evidence of anything.
_FRAGMENT = re.compile(r"^[\s$€£₹¥%()\-–—.,:;*†‡§]+$")
_RANGE_CELL = re.compile(r"^\s*[<>≤≥]?\s*\d+(?:[.,]\d+)?\s*(?:(?:-|–|—|to)\s*\d+(?:[.,]\d+)?)?\s*$")


@dataclass(frozen=True)
class OCRLine:
    """One line as the OCR engine read it, before any Markdown cleanup."""

    index: int  # reading order on the page, 0-based
    text: str
    score: float | None = None
    bbox: tuple[float, float, float, float] | None = None  # PDF points, top-left origin

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "score": None if self.score is None else round(self.score, 3),
            "bbox": None if self.bbox is None else [round(v, 1) for v in self.bbox],
        }


@dataclass(frozen=True)
class ReviewFlag:
    """One OCR reading to check against the original page."""

    page_num: int  # 0-indexed
    kind: str
    text: str  # the OCR line (or table row) as read
    reason: str  # human-readable why
    score: float | None = None  # OCR line score, when the engine gives one
    line_index: int | None = None  # OCRLine.index on this page
    bbox: tuple[float, float, float, float] | None = None
    anchor: dict | None = None  # e.g. {"table": 0, "row": 2, "col": 1}
    occurrence: int = 0  # disambiguates identical flags on one page
    id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            key = "|".join(
                (
                    str(self.page_num),
                    self.kind,
                    " ".join(self.text.split()).lower(),
                    repr(sorted((self.anchor or {}).items())),
                    str(self.occurrence),
                )
            )
            object.__setattr__(self, "id", hashlib.sha1(key.encode()).hexdigest()[:12])

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "page": self.page_num + 1,
            "kind": self.kind,
            "text": self.text,
            "reason": self.reason,
            "score": None if self.score is None else round(self.score, 3),
            "line_index": self.line_index,
            "bbox": None if self.bbox is None else [round(v, 1) for v in self.bbox],
        }
        if self.anchor:
            out["anchor"] = dict(self.anchor)
        return out


def min_line_score() -> float:
    """Line-score cutoff; ``PDFMUX_OCR_MIN_LINE_SCORE`` overrides 0.92 (0 disables)."""
    raw = os.environ.get(_SCORE_ENV)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Ignoring non-numeric %s=%r", _SCORE_ENV, raw)
        else:
            if 0.0 <= value <= 1.0:
                return value
            logger.warning("Ignoring %s=%r (must be between 0 and 1)", _SCORE_ENV, raw)
    return DEFAULT_MIN_LINE_SCORE


def range_check_enabled() -> bool:
    """Opt-in reference-range check (``PDFMUX_OCR_RANGE_CHECK=1``)."""
    return os.environ.get(_RANGE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def _valid(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def detect_date_order(texts: Iterable[str]) -> str | None:
    """``"dmy"``, ``"mdy"``, ``"mixed"`` or ``None`` from unambiguous numeric dates."""
    dmy = mdy = False
    for text in texts:
        for m in _NUM_DMY.finditer(text):
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12 and 1 <= b <= 12:
                dmy = True
            elif b > 12 and 1 <= a <= 12:
                mdy = True
    if dmy and mdy:
        return "mixed"
    return "dmy" if dmy else "mdy" if mdy else None


@dataclass(frozen=True)
class _DateSpan:
    text: str
    readings: tuple[date | None, ...]
    year: int
    start: int
    swappable: bool  # numeric d/m where both orders give different valid dates


def _candidates(line: str, order: str | None = None) -> list[_DateSpan]:
    found: list[_DateSpan] = []
    taken: list[tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        s, e = m.span()
        return all(e <= a or s >= b for a, b in taken)

    def add(m: re.Match, readings: list[date | None], year: int, swappable: bool = False) -> None:
        taken.append(m.span())
        found.append(_DateSpan(m.group(0), tuple(readings), year, m.start(), swappable))

    for m in _NUM_YMD.finditer(line):
        y, mo, d = (int(g) for g in m.groups())
        add(m, [_valid(y, mo, d)], y)
    for m in _NUM_DMY.finditer(line):
        if not free(m):
            continue
        a, b, y = (int(g) for g in m.groups())
        day_first, month_first = _valid(y, b, a), _valid(y, a, b)
        if order == "dmy":
            readings = [day_first]
        elif order == "mdy":
            readings = [month_first]
        else:
            readings = [day_first, month_first]
        swappable = a != b and day_first is not None and month_first is not None
        add(m, readings, y, swappable)
    for m in _NUM_MY.finditer(line):
        if not free(m):
            continue
        mo, y = (int(g) for g in m.groups())
        if 1 <= mo <= 12:  # "Batch 13/2026" is a code, not a date
            add(m, [_valid(y, mo, 1)], y)
    for m in _TXT_DMY.finditer(line):
        if not free(m):
            continue
        d, mon, y = m.groups()
        add(m, [_valid(int(y), _MONTHS[mon.lower()], int(d))], int(y))
    for m in _TXT_MDY.finditer(line):
        if not free(m):
            continue
        mon, d, y = m.groups()
        add(m, [_valid(int(y), _MONTHS[mon.lower()], int(d))], int(y))
    for m in _TXT_MY.finditer(line):
        if not free(m):
            continue
        mon, y = m.groups()
        add(m, [_valid(int(y), _MONTHS[mon.lower()], 1)], int(y))
    return found


def _future_expected(line: str, start: int) -> bool:
    """True when the nearest label before ``start`` says this date is meant to be ahead."""
    before = line[:start]
    fwd = [m.end() for m in _FORWARD_WORDS.finditer(before)]
    past = [m.end() for m in _PAST_WORDS.finditer(before)]
    if not fwd:
        return False
    return not past or max(fwd) > max(past)


def date_problems(line: str, today: date | None = None, order: str | None = None) -> list[str]:
    """Reasons the dates in ``line`` cannot be right; empty when they look fine.

    ``order`` is the document's day/month order from :func:`detect_date_order`.
    """
    today = today or date.today()
    problems: list[str] = []
    for span in _candidates(line, order if order in ("dmy", "mdy") else None):
        if span.year < _MIN_YEAR:
            problems.append(f'"{span.text}" is before {_MIN_YEAR}')
            continue
        if span.year > today.year + _MAX_YEARS_AHEAD:
            problems.append(f'"{span.text}" is more than {_MAX_YEARS_AHEAD} years in the future')
            continue
        valid = [r for r in span.readings if r is not None]
        if not valid:
            problems.append(f'"{span.text}" is not a real calendar date')
            continue
        if all(r > today for r in valid) and not _future_expected(line, span.start):
            problems.append(f'"{span.text}" is later than today')
    return problems


def _fmt(d: date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def ambiguous_dates(line: str, order: str | None) -> list[str]:
    """Reasons for dates whose day and month could swap, given the document's order."""
    if order in ("dmy", "mdy"):
        return []
    out = []
    for span in _candidates(line):
        if not span.swappable:
            continue
        day_first, month_first = span.readings
        assert day_first is not None and month_first is not None
        why = (
            "this document mixes day-first and month-first dates"
            if order == "mixed"
            else "nothing else in the document settles day-first or month-first"
        )
        out.append(f'"{span.text}" could be {_fmt(day_first)} or {_fmt(month_first)}; {why}')
    return out


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _label_before(text: str, start: int) -> str:
    """The words before a value on its line or row: the analyte or field name."""
    label = re.sub(r"[\s:|\-–—]+$", "", text[:start]).strip()
    return label if re.search(r"[A-Za-z]{2}", label) else ""


def _number_findings(text: str, *, range_check: bool | None = None) -> list[tuple[int, str, str]]:
    """(start, matched text, message) for each suspect number on a result line or row."""
    if not _UNIT_RE.search(text):
        return []
    found: list[tuple[int, str, str]] = []
    if not _TIME_WORDS.search(text):
        for m in _COLON_NUM.finditer(text):
            if m.group(1) == "1" and int(m.group(2)) >= 2:
                continue  # 1:80 is a titre, not a decimal
            found.append(
                (m.start(), m.group(0), f'"{m.group(0)}" may be a decimal point read as a colon')
            )
    for m in _LETTER_NUM.finditer(text):
        found.append(
            (
                m.start(),
                m.group(0),
                f'"{m.group(0)}" mixes letters into a number (O for 0, l or I for 1)',
            )
        )
    if range_check if range_check is not None else range_check_enabled():
        found.extend(_range_findings(text))
    return found


def number_problems(line: str, *, range_check: bool | None = None) -> list[str]:
    """Numbers on a result line that look misread, each prefixed with its label.

    Never rewrites anything.
    """
    out = []
    for start, _, message in _number_findings(line, range_check=range_check):
        label = _label_before(line, start)
        out.append(f"{label}: {message}" if label and not message.startswith(label) else message)
    return out


def _range_findings(line: str) -> list[tuple[int, str, str]]:
    vm = _VALUE_AFTER_LABEL.search(line)
    if not vm:
        return []
    rm = next(_RANGE.finditer(line, vm.end()), None)
    if not rm:
        return []
    value, lo, hi = float(vm.group("v")), float(rm.group("lo")), float(rm.group("hi"))
    if hi <= 0 or lo > hi:
        return []
    label = _label_before(line, vm.start("v")) or "value"
    if value > hi * 5 or (lo > 0 and value < lo / 5):
        return [
            (
                vm.start("v"),
                vm.group("v"),
                f"{label} {vm.group('v')} is more than 5x outside its reference range "
                f"{rm.group(0)} (a dropped or extra decimal point?)",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _cell_type(cell: str) -> str | None:
    c = cell.strip()
    if not c or _FRAGMENT.match(c):
        return None
    if _NUM_CELL.match(c):
        return "number"
    if _RANGE_CELL.match(c):
        return "range"
    if _UNIT_CELL.match(c):
        return "unit"
    return "text"


def _column_types(rows: Sequence[Sequence[str]], ncols: int) -> list[str | None]:
    types: list[str | None] = []
    for col in range(ncols):
        seen = [t for r in rows if col < len(r) and (t := _cell_type(r[col]))]
        best = None
        if len(seen) >= 3:
            top = max(set(seen), key=seen.count)
            if seen.count(top) / len(seen) >= 0.6:
                best = top
        types.append(best)
    return types


def _fits(row: Sequence[str], types: Sequence[str | None], shift: int) -> int:
    """How many typed columns ``row`` satisfies when moved ``shift`` columns."""
    ok = 0
    for col, want in enumerate(types):
        src = col - shift
        if want is None or not (0 <= src < len(row)):
            continue
        got = _cell_type(row[src])
        if got == want or (want == "range" and got == "number"):
            ok += 1
    return ok


def review_table(
    page_num: int,
    table_index: int,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[ReviewFlag, ...]:
    """Cells that break their column's type, and rows that look shifted by one column."""
    if len(rows) < 3:
        return ()
    ncols = max(len(headers), *(len(r) for r in rows))
    types = _column_types(rows, ncols)
    if sum(t in ("number", "range", "unit") for t in types) == 0:
        return ()
    flags: list[ReviewFlag] = []
    for r, row in enumerate(rows):
        row_text = " | ".join(c.strip() for c in row)
        bad = [
            c
            for c, want in enumerate(types)
            if want in ("number", "range", "unit")
            and c < len(row)
            and (got := _cell_type(row[c])) is not None
            and got != want
            and not (want == "range" and got == "number")
        ]
        if not bad:
            continue
        base = _fits(row, types, 0)
        shifted = [s for s in (1, -1) if _fits(row, types, s) > base]
        if len(bad) >= 2 and shifted:
            direction = "right" if shifted[0] == 1 else "left"
            flags.append(
                ReviewFlag(
                    page_num,
                    TABLE_CELL,
                    row_text,
                    f"row {r + 1} of table {table_index + 1} fits its columns only if moved "
                    f"one column {direction}: a cell may have been dropped or split",
                    bbox=bbox,
                    anchor={"table": table_index, "row": r},
                )
            )
            continue
        for c in bad:
            head = (
                headers[c].strip() if c < len(headers) and headers[c].strip() else f"column {c + 1}"
            )
            flags.append(
                ReviewFlag(
                    page_num,
                    TABLE_CELL,
                    row_text,
                    f'"{row[c].strip()}" in the {types[c]} column "{head}" '
                    f"(row {r + 1} of table {table_index + 1})",
                    bbox=bbox,
                    anchor={"table": table_index, "row": r, "col": c},
                )
            )
    return tuple(flags)


def pipe_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Markdown pipe tables in ``text`` as (headers, rows)."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    block: list[str] = []

    def flush() -> None:
        rows = [
            [c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in block
            if not re.fullmatch(r"\s*\|?[\s:|\-]+\|?\s*", ln)
        ]
        if len(rows) >= 2:
            tables.append((rows[0], rows[1:]))
        block.clear()

    for ln in text.splitlines():
        if ln.strip().startswith("|"):
            block.append(ln)
        elif block:
            flush()
    if block:
        flush()
    return tables


# ---------------------------------------------------------------------------
# Visual rows and scanned tables
# ---------------------------------------------------------------------------
#
# OCR engines return one box per run of text. In a columned lab report every cell
# is its own box ("Glucose", "8:02", "mmol/L", "3.9 - 5.5"), so a check that needs
# a unit on the same line never fires, and no table exists to check. The boxes'
# positions rebuild both: boxes sharing a baseline band form a visual row, and a
# run of rows whose boxes line up in the same x-bands forms a table.


def visual_rows(lines: Sequence[OCRLine]) -> list[list[OCRLine]]:
    """Group boxed OCR lines into visual rows, top to bottom, cells left to right."""
    boxed = [ln for ln in lines if ln.bbox is not None and ln.text.strip()]
    boxed.sort(key=lambda ln: ((ln.bbox[1] + ln.bbox[3]) / 2, ln.bbox[0]))  # type: ignore[index]
    rows: list[list[OCRLine]] = []
    band: tuple[float, float] | None = None
    for ln in boxed:
        top, bottom = ln.bbox[1], ln.bbox[3]  # type: ignore[index]
        mid = (top + bottom) / 2
        if band is not None and band[0] <= mid <= band[1]:
            rows[-1].append(ln)
            band = (min(band[0], top), max(band[1], bottom))
        else:
            rows.append([ln])
            band = (top, bottom)
    for row in rows:
        row.sort(key=lambda ln: ln.bbox[0])  # type: ignore[index]
    return rows


@dataclass(frozen=True)
class OCRTable:
    """A table rebuilt from OCR box positions. Cells keep their own score and bbox."""

    index: int
    columns: tuple[tuple[float, float], ...]  # x-extent of each column, PDF points
    rows: tuple[tuple[OCRLine | None, ...], ...]  # grid; None = empty cell
    header: bool  # first row is a header (no numbers in it)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        cells = [c for r in self.rows for c in r if c is not None and c.bbox]
        return (
            min(c.bbox[0] for c in cells),  # type: ignore[index]
            min(c.bbox[1] for c in cells),  # type: ignore[index]
            max(c.bbox[2] for c in cells),  # type: ignore[index]
            max(c.bbox[3] for c in cells),  # type: ignore[index]
        )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "bbox": [round(v, 1) for v in self.bbox],
            "header": self.header,
            "rows": [[None if c is None else c.to_dict() for c in r] for r in self.rows],
        }


def _column_bands(rows: Sequence[Sequence[OCRLine]], pad: float) -> list[tuple[float, float]]:
    spans = sorted((c.bbox[0], c.bbox[2]) for r in rows for c in r)  # type: ignore[index]
    bands: list[list[float]] = []
    for x0, x1 in spans:
        if bands and x0 <= bands[-1][1] + pad:
            bands[-1][1] = max(bands[-1][1], x1)
        else:
            bands.append([x0, x1])
    return [(a, b) for a, b in bands]


def _grid(row: Sequence[OCRLine], bands: Sequence[tuple[float, float]]) -> list[OCRLine | None]:
    cells: list[OCRLine | None] = [None] * len(bands)
    for c in row:
        x0, x1 = c.bbox[0], c.bbox[2]  # type: ignore[index]
        overlaps = [max(0.0, min(x1, b1) - max(x0, b0)) for b0, b1 in bands]
        col = max(range(len(bands)), key=lambda i: overlaps[i])
        if cells[col] is None:
            cells[col] = c
        else:  # two boxes in one cell: keep reading order, join text
            prev = cells[col]
            assert prev is not None
            cells[col] = OCRLine(
                prev.index,
                f"{prev.text} {c.text}",
                min(s for s in (prev.score, c.score) if s is not None)
                if prev.score is not None or c.score is not None
                else None,
                (
                    min(prev.bbox[0], c.bbox[0]),  # type: ignore[index]
                    min(prev.bbox[1], c.bbox[1]),  # type: ignore[index]
                    max(prev.bbox[2], c.bbox[2]),  # type: ignore[index]
                    max(prev.bbox[3], c.bbox[3]),  # type: ignore[index]
                ),
            )
    return cells


def find_ocr_tables(lines: Sequence[OCRLine]) -> list[OCRTable]:
    """Tables rebuilt from OCR box positions: 3+ rows of 3+ cells sharing column bands."""
    rows = visual_rows(lines)
    if not rows:
        return []
    heights = sorted(
        c.bbox[3] - c.bbox[1]  # type: ignore[index]
        for r in rows
        for c in r
    )
    h = heights[len(heights) // 2] or 10.0
    tables: list[OCRTable] = []
    run: list[list[OCRLine]] = []

    def row_top(r: Sequence[OCRLine]) -> float:
        return min(c.bbox[1] for c in r)  # type: ignore[index]

    def row_bottom(r: Sequence[OCRLine]) -> float:
        return max(c.bbox[3] for c in r)  # type: ignore[index]

    def close() -> None:
        if len(run) >= 3:
            bands = _column_bands(run, pad=h * 0.6)
            if len(bands) >= 3:
                grid = [_grid(r, bands) for r in run]
                filled = sum(1 for g in grid if sum(c is not None for c in g) >= len(bands) - 1)
                if filled / len(grid) >= 0.6:
                    first = [c.text for c in grid[0] if c is not None]
                    header = not any(_cell_type(t) in ("number", "range") for t in first)
                    tables.append(
                        OCRTable(
                            len(tables),
                            tuple(bands),
                            tuple(tuple(g) for g in grid),
                            header,
                        )
                    )
        run.clear()

    for r in rows:
        if len(r) >= 3 and (not run or row_top(r) - row_bottom(run[-1]) <= h * 2.5):
            run.append(r)
        else:
            close()
            if len(r) >= 3:
                run.append(r)
    close()
    return tables


def _review_ocr_table(page_num: int, table: OCRTable) -> list[ReviewFlag]:
    """Column checks on a scanned table, anchored to the exact cell box."""
    texts = [[c.text if c is not None else "" for c in r] for r in table.rows]
    headers, body = (texts[0], texts[1:]) if table.header else ([], texts)
    offset = 1 if table.header else 0
    out = []
    for f in review_table(page_num, table.index, headers, body):
        anchor = dict(f.anchor or {})
        anchor.pop("table", None)
        r = anchor["row"] + offset
        cells = [c for c in table.rows[r] if c is not None]
        target = table.rows[r][anchor["col"]] if "col" in anchor else None
        if target is not None:
            bbox, line_index, score = target.bbox, target.index, target.score
        else:  # shifted row: the whole row's box
            bbox = (
                min(c.bbox[0] for c in cells),  # type: ignore[index]
                min(c.bbox[1] for c in cells),  # type: ignore[index]
                max(c.bbox[2] for c in cells),  # type: ignore[index]
                max(c.bbox[3] for c in cells),  # type: ignore[index]
            )
            line_index, score = cells[0].index, None
        out.append(
            ReviewFlag(
                page_num,
                TABLE_CELL,
                f.text,
                f.reason.replace(f"table {table.index + 1}", f"scanned table {table.index + 1}"),
                score=score,
                line_index=line_index,
                bbox=bbox,
                anchor={
                    "ocr_table": table.index,
                    "row": r,
                    **({"col": anchor["col"]} if "col" in anchor else {}),
                },
            )
        )
    return out


def _row_number_flags(
    page_num: int, row: Sequence[OCRLine], seen: set[tuple[int, str]]
) -> list[ReviewFlag]:
    """Number checks across a visual row, so a value and its unit in separate boxes meet."""
    if len(row) < 2:
        return []
    text, starts = "", []
    for c in row:
        starts.append(len(text))
        text += c.text.strip() + "   "
    out = []
    for start, matched, message in _number_findings(text):
        cell = max(i for i, s0 in enumerate(starts) if s0 <= start)
        line = row[cell]
        if (line.index, matched) in seen:
            continue  # the line-level check already caught it
        seen.add((line.index, matched))
        label = _label_before(text, starts[cell]) if cell else _label_before(text, start)
        reason = f"{label}: {message}" if label and not message.startswith(label) else message
        out.append(
            ReviewFlag(
                page_num,
                SUSPECT_NUMBER,
                " | ".join(c.text.strip() for c in row),
                reason,
                score=line.score,
                line_index=line.index,
                bbox=line.bbox,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Pages and documents
# ---------------------------------------------------------------------------


def _as_lines(lines: Iterable) -> list[OCRLine]:
    out: list[OCRLine] = []
    for i, item in enumerate(lines):
        if isinstance(item, OCRLine):
            out.append(item)
        else:
            text, score = item
            out.append(OCRLine(i, text, score))
    return out


def _number(flags: list[ReviewFlag]) -> tuple[ReviewFlag, ...]:
    """Give identical flags on a page distinct occurrence numbers, so ids stay unique."""
    seen: dict[tuple, int] = {}
    out = []
    for f in flags:
        key = (f.kind, " ".join(f.text.split()).lower(), repr(sorted((f.anchor or {}).items())))
        n = seen.get(key, 0)
        seen[key] = n + 1
        if n:
            f = ReviewFlag(
                f.page_num, f.kind, f.text, f.reason, f.score, f.line_index, f.bbox, f.anchor, n
            )
        out.append(f)
    return tuple(out)


def review_page(
    page_num: int,
    lines: Iterable,
    *,
    today: date | None = None,
    min_score: float | None = None,
    date_order: str | None = None,
    check_ambiguous: bool = False,
) -> tuple[ReviewFlag, ...]:
    """Flags for one OCR'd page.

    ``lines`` is a sequence of :class:`OCRLine` or ``(text, score)`` pairs.
    ``date_order`` and ``check_ambiguous`` come from the document-level pass.
    """
    cutoff = min_line_score() if min_score is None else min_score
    flags: list[ReviewFlag] = []
    seen_numbers: set[tuple[int, str]] = set()
    lines = _as_lines(lines)
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        anchor = dict(score=line.score, line_index=line.index, bbox=line.bbox)
        for problem in date_problems(text, today, date_order):
            flags.append(
                ReviewFlag(
                    page_num, IMPLAUSIBLE_DATE, text, f"date may be misread: {problem}", **anchor
                )
            )
        if check_ambiguous:
            for problem in ambiguous_dates(text, date_order):
                flags.append(ReviewFlag(page_num, AMBIGUOUS_DATE, text, problem, **anchor))
        for start, matched, message in _number_findings(text):
            label = _label_before(text, start)
            reason = f"{label}: {message}" if label and not message.startswith(label) else message
            flags.append(ReviewFlag(page_num, SUSPECT_NUMBER, text, reason, **anchor))
            seen_numbers.add((line.index, matched))
        if line.score is not None and line.score < cutoff:
            flags.append(
                ReviewFlag(
                    page_num,
                    LOW_CONFIDENCE,
                    text,
                    f"low OCR confidence ({line.score:.2f} < {cutoff:.2f})",
                    **anchor,
                )
            )
    # Boxed lines (RapidOCR): rebuild visual rows and scanned tables from positions.
    if any(ln.bbox is not None for ln in lines):
        for row in visual_rows(lines):
            flags.extend(_row_number_flags(page_num, row, seen_numbers))
        for table in find_ocr_tables(lines):
            flags.extend(_review_ocr_table(page_num, table))
    return _number(flags)


def review_text(page_num: int, text: str, *, today: date | None = None) -> tuple[ReviewFlag, ...]:
    """Content checks for OCR engines that give text without line scores."""
    return review_page(page_num, ((line, None) for line in text.splitlines()), today=today)


def review_document(
    pages: Sequence,
    ocr_page_nums: Iterable[int] = (),
    *,
    today: date | None = None,
) -> dict[int, tuple[ReviewFlag, ...]]:
    """Whole-document review: flags keyed by page number.

    Everything runs on OCR'd pages only, tables included; digital pages get no flags.

    ``pages`` are :class:`pdfmux.types.PageResult`-like objects (``page_num``,
    ``text``, ``ocr_applied``, optional ``ocr_lines`` and ``tables``). The day/month
    order is read from every page, OCR'd or not.
    """
    ocr_set = set(ocr_page_nums)
    order = detect_date_order(
        [p.text for p in pages]
        + [ln.text for p in pages for ln in (getattr(p, "ocr_lines", ()) or ())]
    )
    out: dict[int, tuple[ReviewFlag, ...]] = {}
    for p in pages:
        is_ocr = bool(getattr(p, "ocr_applied", False) or p.page_num in ocr_set)
        flags: list[ReviewFlag] = []
        if is_ocr:
            lines = getattr(p, "ocr_lines", ()) or [
                OCRLine(i, t) for i, t in enumerate(p.text.splitlines())
            ]
            flags.extend(
                review_page(p.page_num, lines, today=today, date_order=order, check_ambiguous=True)
            )
        if not is_ocr:
            continue
        # Tables only on OCR'd pages: on 31 real digital PDFs (10-Ks, FDA reports) the
        # column-type check raised 313 flags on accounting layouts and footnote marks,
        # none of them extraction errors. A column shift is an OCR failure mode.
        tables = [
            (list(t.headers), [list(r) for r in t.rows], t.bbox)
            for t in (getattr(p, "tables", ()) or ())
        ] or [(h, r, None) for h, r in pipe_tables(p.text)]
        for i, (headers, rows, bbox) in enumerate(tables):
            flags.extend(review_table(p.page_num, i, headers, rows, bbox))
        out[p.page_num] = _number(flags)
    return out


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

_MAX_LISTED = 10
_EXAMPLES_PER_PAGE = 3


def _clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarise(flags: list[ReviewFlag], one: str, many: str) -> list[str]:
    by_page: dict[int, list[ReviewFlag]] = {}
    for f in flags:
        by_page.setdefault(f.page_num, []).append(f)
    out = []
    for page_num in sorted(by_page):
        page_flags = sorted(
            by_page[page_num], key=lambda f: f.score if f.score is not None else 1.0
        )
        examples = "; ".join(f'"{_clip(f.text, 40)}"' for f in page_flags[:_EXAMPLES_PER_PAGE])
        more = len(page_flags) - _EXAMPLES_PER_PAGE
        suffix = f" and {more} more" if more > 0 else ""
        noun = one if len(page_flags) == 1 else many
        out.append(
            f"Page {page_num + 1}: {len(page_flags)} {noun} — check the original "
            f"({examples}{suffix})"
        )
    return out


def flags_to_warnings(flags: Sequence[ReviewFlag]) -> list[str]:
    """Short warnings for the CLI, MCP and Markdown note. JSON carries every flag."""
    if not flags:
        return []
    warnings: list[str] = []

    listed = [f for f in flags if f.kind in (IMPLAUSIBLE_DATE, SUSPECT_NUMBER)]
    for f in listed[:_MAX_LISTED]:
        what = f.reason.removeprefix("date may be misread: ")
        warnings.append(
            f'Page {f.page_num + 1}: OCR {what} — check the original ("{_clip(f.text)}")'
        )
    if len(listed) > _MAX_LISTED:
        warnings.append(
            f"{len(listed) - _MAX_LISTED} more OCR dates or numbers to check "
            "(see review_flags in JSON)"
        )

    ambiguous = [f for f in flags if f.kind == AMBIGUOUS_DATE]
    if ambiguous:
        pages = sorted({f.page_num + 1 for f in ambiguous})
        shown = ", ".join(str(p) for p in pages[:10]) + (", ..." if len(pages) > 10 else "")
        warnings.append(
            f"{len(ambiguous)} OCR date(s) could be read day-first or month-first "
            f"(pages {shown}) — check the original"
        )

    warnings += _summarise(
        [f for f in flags if f.kind == TABLE_CELL],
        "table cell or row out of place",
        "table cells or rows out of place",
    )
    warnings += _summarise(
        [f for f in flags if f.kind == LOW_CONFIDENCE],
        "OCR line read with low confidence",
        "OCR lines read with low confidence",
    )
    return warnings
