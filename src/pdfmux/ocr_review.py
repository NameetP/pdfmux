"""Flag OCR readings that a person should check against the original page.

OCR output is searchable text, not a record. It fails in two ways that look
identical in the extracted text:

1. **Low-confidence lines.** RapidOCR scores every line it reads. On degraded
   scans (5-8 pt print, heavy JPEG compression, 100-200 DPI) a 0.92 cutoff
   caught 63% of misread lines and flagged 1 of 625 correctly read lines
   (12 synthetic lab-report pages, 755 lines). ``PDFMUX_OCR_MIN_LINE_SCORE``
   overrides the cutoff.
2. **Confident misreads.** A single wrong digit is often read with high
   confidence: a report dated 2026 came back as 2096. No score catches that,
   so dates are sanity-checked on every OCR'd page, scored or not: impossible
   (month 13, 31 February), before 1900, or more than 10 years ahead. A date
   later than today is judged by the nearest label before it on the line:
   "next", "due", "valid until", "expiry", "appointment" make a future date
   expected; "collected", "DOB", "reported", "issued" (or no label) make it
   a likely misread.

Flags never change the extracted text. They are reported as warnings and, in
JSON output, as structured ``review_flags`` entries.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

DEFAULT_MIN_LINE_SCORE = 0.92
_SCORE_ENV = "PDFMUX_OCR_MIN_LINE_SCORE"

LOW_CONFIDENCE = "low_confidence"
IMPLAUSIBLE_DATE = "implausible_date"

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
# 12/03/2026, 12-3-2026, 12.03.2026 (day/month order unknown)
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


@dataclass(frozen=True)
class ReviewFlag:
    """One OCR reading to check against the original page."""

    page_num: int  # 0-indexed
    kind: str  # LOW_CONFIDENCE or IMPLAUSIBLE_DATE
    text: str  # the OCR line as extracted
    reason: str  # human-readable why
    score: float | None = None  # OCR line score, when the engine gives one

    def to_dict(self) -> dict:
        return {
            "page": self.page_num + 1,
            "kind": self.kind,
            "text": self.text,
            "reason": self.reason,
            "score": None if self.score is None else round(self.score, 3),
        }


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


def _valid(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _candidates(line: str) -> list[tuple[str, list[date | None], int, int]]:
    """(matched text, possible readings, year, start) for every date-shaped span in ``line``."""
    found: list[tuple[str, list[date | None], int, int]] = []
    taken: list[tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        s, e = m.span()
        return all(e <= a or s >= b for a, b in taken)

    def add(m: re.Match, readings: list[date | None], year: int) -> None:
        taken.append(m.span())
        found.append((m.group(0), readings, year, m.start()))

    for m in _NUM_YMD.finditer(line):
        y, mo, d = (int(g) for g in m.groups())
        add(m, [_valid(y, mo, d)], y)
    for m in _NUM_DMY.finditer(line):
        if not free(m):
            continue
        a, b, y = (int(g) for g in m.groups())
        # Day/month order is unknown: keep both readings.
        add(m, [_valid(y, b, a), _valid(y, a, b)], y)
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


def date_problems(line: str, today: date | None = None) -> list[str]:
    """Reasons the dates in ``line`` cannot be right; empty when they look fine."""
    today = today or date.today()
    problems: list[str] = []
    for span, readings, year, start in _candidates(line):
        if year < _MIN_YEAR:
            problems.append(f'"{span}" is before {_MIN_YEAR}')
            continue
        if year > today.year + _MAX_YEARS_AHEAD:
            problems.append(f'"{span}" is more than {_MAX_YEARS_AHEAD} years in the future')
            continue
        valid = [r for r in readings if r is not None]
        if not valid:
            problems.append(f'"{span}" is not a real calendar date')
            continue
        if all(r > today for r in valid) and not _future_expected(line, start):
            problems.append(f'"{span}" is later than today')
    return problems


def review_page(
    page_num: int,
    lines: Iterable[tuple[str, float | None]],
    *,
    today: date | None = None,
    min_score: float | None = None,
) -> tuple[ReviewFlag, ...]:
    """Flags for one OCR'd page. ``lines`` pairs each line with its score (or None)."""
    cutoff = min_line_score() if min_score is None else min_score
    flags: list[ReviewFlag] = []
    for raw_text, score in lines:
        text = raw_text.strip()
        if not text:
            continue
        for problem in date_problems(text, today):
            reason = f"date may be misread: {problem}"
            flags.append(ReviewFlag(page_num, IMPLAUSIBLE_DATE, text, reason, score))
        if score is not None and score < cutoff:
            flags.append(
                ReviewFlag(
                    page_num,
                    LOW_CONFIDENCE,
                    text,
                    f"low OCR confidence ({score:.2f} < {cutoff:.2f})",
                    score,
                )
            )
    return tuple(flags)


def review_text(page_num: int, text: str, *, today: date | None = None) -> tuple[ReviewFlag, ...]:
    """Date checks only, for OCR engines that give text without line scores."""
    return review_page(page_num, ((line, None) for line in text.splitlines()), today=today)


_MAX_DATE_WARNINGS = 10
_EXAMPLES_PER_PAGE = 3


def _clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def flags_to_warnings(flags: Sequence[ReviewFlag]) -> list[str]:
    """Short warnings for the CLI, MCP and Markdown note. JSON carries every flag."""
    if not flags:
        return []
    warnings: list[str] = []

    dates = [f for f in flags if f.kind == IMPLAUSIBLE_DATE]
    for f in dates[:_MAX_DATE_WARNINGS]:
        warnings.append(
            f"Page {f.page_num + 1}: OCR {f.reason.removeprefix('date may be misread: ')} "
            f'— check the original ("{_clip(f.text)}")'
        )
    if len(dates) > _MAX_DATE_WARNINGS:
        warnings.append(
            f"{len(dates) - _MAX_DATE_WARNINGS} more OCR dates to check (see review_flags in JSON)"
        )

    by_page: dict[int, list[ReviewFlag]] = {}
    for f in flags:
        if f.kind == LOW_CONFIDENCE:
            by_page.setdefault(f.page_num, []).append(f)
    for page_num in sorted(by_page):
        page_flags = sorted(by_page[page_num], key=lambda f: f.score or 0.0)
        examples = "; ".join(f'"{_clip(f.text, 40)}"' for f in page_flags[:_EXAMPLES_PER_PAGE])
        more = len(page_flags) - _EXAMPLES_PER_PAGE
        suffix = f" and {more} more" if more > 0 else ""
        noun = "line" if len(page_flags) == 1 else "lines"
        warnings.append(
            f"Page {page_num + 1}: {len(page_flags)} OCR {noun} read with low confidence "
            f"— check the original ({examples}{suffix})"
        )
    return warnings
