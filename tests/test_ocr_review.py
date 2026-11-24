"""Tests for OCR review flags (pdfmux.ocr_review).

Regression: a scanned lab report dated 2026 came back from OCR as 2096 with
no signal. Flags must point a reviewer at that line without changing the text.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pdfmux.ocr_review import (
    DEFAULT_MIN_LINE_SCORE,
    IMPLAUSIBLE_DATE,
    LOW_CONFIDENCE,
    ReviewFlag,
    date_problems,
    flags_to_warnings,
    min_line_score,
    review_page,
    review_text,
)

TODAY = date(2026, 11, 24)


class TestDateProblems:
    @pytest.mark.parametrize(
        "line",
        [
            "Collected on 12/03/2096",  # the reported case
            "Report date 2096-03-12",
            "12 Mar 2096",
            "March 12, 2096",
            "Expiry 12/2099",  # too far ahead even with a forward label
            "Next review 12/01/2045",
        ],
    )
    def test_far_future_is_always_flagged(self, line):
        problems = date_problems(line, TODAY)
        assert problems and "more than 10 years" in problems[0]

    @pytest.mark.parametrize(
        "line",
        [
            "Collected on 14/12/2026",
            "Reported 2027-01-05",
            "DOB 03/04/2027",
            "Sample received 5 Jan 2027",
            "12/03/2027",  # no label: a report's dates are normally in the past
            "Referred to Dr Rao on 12/03/2027",  # "to" is not a forward label
        ],
    )
    def test_future_past_event_is_flagged(self, line):
        problems = date_problems(line, TODAY)
        assert problems and "later than today" in problems[0]

    @pytest.mark.parametrize(
        "line",
        [
            "Next review 12/01/2027",
            "Valid until March 2030",
            "Expiry 12/2030",
            "Follow-up appointment 05/02/2027",
            "Due date 2027-06-01",
            "Collected 12/03/2026  Next due 12/03/2027",  # nearest label wins
        ],
    )
    def test_expected_future_dates_pass(self, line):
        assert date_problems(line, TODAY) == []

    def test_nearest_label_decides(self):
        # The forward label belongs to the first date; "collected" governs the second.
        problems = date_problems("Next due 12/03/2027 collected 14/12/2026", TODAY)
        assert problems == ['"14/12/2026" is later than today']

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("DOB 31/02/1985", "not a real calendar date"),
            ("Report 31/13/2025", "not a real calendar date"),
            ("Dated 1885-04-02", "before 1900"),
        ],
    )
    def test_impossible_dates(self, line, expected):
        problems = date_problems(line, TODAY)
        assert problems and expected in problems[0]

    @pytest.mark.parametrize(
        "line",
        [
            "Report date 12/03/2026",
            "Collected 24/11/2026",  # today
            "HbA1c 6.2 % ref 4.0-5.6",
            "Batch 13/2026",  # a code, not a date
            "Card 4532/2026/11",
            "Haemoglobin 13.2 g/dL",
            "Invoice 2096",  # a bare number is not a date
            "Phone 04-123-4567",
            "",
        ],
    )
    def test_ordinary_lines_pass(self, line):
        assert date_problems(line, TODAY) == []

    def test_ambiguous_day_month_is_flagged_only_if_both_readings_are_future(self):
        # 05/12/2026: 5 Dec (future) or 12 May (past) -> one reading is plausible.
        assert date_problems("Collected 05/12/2026", TODAY) == []
        # 14/12/2026: only 14 Dec is valid, and it is future.
        assert date_problems("Collected 14/12/2026", TODAY)

    def test_default_today_is_used(self):
        assert date_problems("Collected 12/03/2199") != []


class TestReviewPage:
    def test_low_confidence_line_is_flagged(self):
        flags = review_page(2, [("HBA1C 6.11 mlU/L", 0.71), ("TSH 2.5 mIU/L", 0.99)], today=TODAY)
        assert [(f.kind, f.text, f.page_num) for f in flags] == [
            (LOW_CONFIDENCE, "HBA1C 6.11 mlU/L", 2)
        ]
        assert flags[0].score == 0.71

    def test_confident_misread_date_is_still_flagged(self):
        flags = review_page(0, [("Collected on 12/03/2096", 0.99)], today=TODAY)
        assert [f.kind for f in flags] == [IMPLAUSIBLE_DATE]
        assert flags[0].score == 0.99

    def test_line_can_carry_both_flags(self):
        flags = review_page(0, [("Collected on 12/03/2096", 0.5)], today=TODAY)
        assert {f.kind for f in flags} == {IMPLAUSIBLE_DATE, LOW_CONFIDENCE}

    def test_cutoff_is_exclusive_and_overridable(self):
        line = [("TSH 2.5", DEFAULT_MIN_LINE_SCORE)]
        assert review_page(0, line, today=TODAY) == ()
        assert review_page(0, line, today=TODAY, min_score=0.95)

    def test_zero_cutoff_disables_score_flags(self):
        assert review_page(0, [("TSH 2.5", 0.1)], today=TODAY, min_score=0.0) == ()

    def test_blank_lines_and_missing_scores(self):
        assert review_page(0, [("   ", 0.1), ("TSH 2.5", None)], today=TODAY) == ()

    def test_review_text_checks_dates_only(self):
        flags = review_text(4, "Name A\nCollected 12/03/2096\nTSH 2.5", today=TODAY)
        assert [(f.kind, f.text, f.page_num, f.score) for f in flags] == [
            (IMPLAUSIBLE_DATE, "Collected 12/03/2096", 4, None)
        ]

    def test_to_dict_is_json_ready_and_one_indexed(self):
        flag = review_page(0, [("Collected 12/03/2096", 0.98765)], today=TODAY)[0]
        d = flag.to_dict()
        assert d["page"] == 1 and d["kind"] == IMPLAUSIBLE_DATE and d["score"] == 0.988
        json.dumps(d)


class TestMinLineScoreEnv:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("PDFMUX_OCR_MIN_LINE_SCORE", raising=False)
        assert min_line_score() == DEFAULT_MIN_LINE_SCORE

    def test_override(self, monkeypatch):
        monkeypatch.setenv("PDFMUX_OCR_MIN_LINE_SCORE", "0.8")
        assert min_line_score() == 0.8
        assert review_page(0, [("TSH 2.5", 0.85)], today=TODAY) == ()

    @pytest.mark.parametrize("bad", ["high", "1.5", "-0.1"])
    def test_invalid_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("PDFMUX_OCR_MIN_LINE_SCORE", bad)
        assert min_line_score() == DEFAULT_MIN_LINE_SCORE


class TestWarnings:
    def test_empty(self):
        assert flags_to_warnings([]) == []

    def test_date_warning_names_page_line_and_action(self):
        flags = review_page(2, [("Collected on 12/03/2096", 0.99)], today=TODAY)
        (w,) = flags_to_warnings(flags)
        assert w.startswith("Page 3: OCR ")
        assert "12/03/2096" in w and "check the original" in w

    def test_low_confidence_is_summarised_per_page(self):
        lines = [(f"line {i}", 0.5 + i / 100) for i in range(6)]
        (w,) = flags_to_warnings(review_page(0, lines, today=TODAY))
        assert w.startswith("Page 1: 6 OCR lines read with low confidence")
        assert '"line 0"' in w and "and 3 more" in w

    def test_single_low_confidence_line_is_singular(self):
        (w,) = flags_to_warnings(review_page(0, [("TSH 2.5", 0.4)], today=TODAY))
        assert "1 OCR line read" in w and "more" not in w

    def test_date_warnings_are_capped(self):
        flags = [
            ReviewFlag(
                0, IMPLAUSIBLE_DATE, f"Collected 0{i % 9 + 1}/03/2096", "date may be misread: x"
            )
            for i in range(14)
        ]
        warnings = flags_to_warnings(flags)
        assert len(warnings) == 11
        assert warnings[-1].startswith("4 more OCR dates to check")

    def test_long_lines_are_clipped(self):
        (w,) = flags_to_warnings(review_page(0, [("x" * 300, 0.1)], today=TODAY))
        assert len(w) < 200


class TestPipelineIntegration:
    """The flags reach every output without changing the extracted text."""

    def _scan(self, tmp_path: Path) -> Path:
        import fitz

        src = fitz.open()
        page = src.new_page(width=595, height=842)
        page.insert_text((60, 100), "CITY LAB  Haematology report", fontsize=14)
        page.insert_text((60, 140), "Collected on 12/03/2096", fontsize=14)
        page.insert_text((60, 180), "Haemoglobin 13.2 g/dL", fontsize=14)
        pix = page.get_pixmap(dpi=200)
        scan = fitz.open()
        sp = scan.new_page(width=595, height=842)
        sp.insert_image(sp.rect, pixmap=pix)  # image only: no text layer
        out = tmp_path / "scan.pdf"
        scan.save(str(out))
        return out

    def test_scanned_report_flags_the_date(self, tmp_path):
        pytest.importorskip("rapidocr")
        from pdfmux.pipeline import process

        result = process(self._scan(tmp_path), output_format="json")
        data = json.loads(result.text)
        assert "2096" in data["content"]  # never silently corrected
        kinds = {f["kind"] for f in data["review_flags"]}
        assert IMPLAUSIBLE_DATE in kinds
        assert any("2096" in w and "check the original" in w for w in data["warnings"])
        assert result.review_flags == data["review_flags"]

    def test_ensure_review_flags_backstop(self):
        from pdfmux.pipeline import _ensure_review_flags
        from pdfmux.types import PageQuality, PageResult

        def page(n, text, ocr):
            return PageResult(n, text, 0.9, PageQuality.GOOD, "x", ocr_applied=ocr)

        pages = [
            page(0, "Collected 12/03/2096", True),  # OCR'd, unreviewed -> flagged
            page(1, "Collected 12/03/2096", False),  # native text is not OCR
            page(2, "Collected 12/03/2096", False),  # listed in ocr_pages -> flagged
        ]
        out = _ensure_review_flags(pages, [2])
        assert [bool(p.review_flags) for p in out] == [True, False, True]
        assert [p.text for p in out] == [p.text for p in pages]

    def test_json_omits_key_when_clean(self):
        from pdfmux.formatters.json_fmt import format_json

        assert "review_flags" not in json.loads(format_json("hello"))
        flagged = json.loads(format_json("hello", review_flags=[{"page": 1}]))
        assert flagged["review_flags"] == [{"page": 1}]
