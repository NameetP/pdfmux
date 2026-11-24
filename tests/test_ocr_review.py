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
        assert warnings[-1].startswith("4 more OCR dates or numbers to check")

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


class TestCacheIsVersioned:
    """A result cached by an older pdfmux must not hide this version's flags."""

    def test_key_changes_with_version(self, monkeypatch):
        from pdfmux import result_cache

        kw = dict(pdf_hash="abc", quality="standard", output_format="json", schema=None)
        monkeypatch.setattr(result_cache, "_pdfmux_version", lambda: "1.8.9")
        old = result_cache._build_key(**kw)
        monkeypatch.setattr(result_cache, "_pdfmux_version", lambda: "1.8.10")
        new = result_cache._build_key(**kw)
        assert old[0] != new[0] and old[1] != new[1]

    def test_key_is_stable_within_a_version(self):
        from pdfmux import result_cache

        kw = dict(pdf_hash="abc", quality="standard", output_format="json", schema=None)
        assert result_cache._build_key(**kw) == result_cache._build_key(**kw)

    def test_review_flags_survive_the_cache_round_trip(self):
        from pdfmux.detect import PDFClassification
        from pdfmux.pipeline import ConversionResult
        from pdfmux.result_cache import _dict_to_result, _result_to_dict

        flags = [{"page": 1, "kind": IMPLAUSIBLE_DATE, "text": "x", "reason": "y", "score": None}]
        original = ConversionResult(
            text="t",
            format="json",
            confidence=0.8,
            extractor_used="rapidocr",
            page_count=1,
            warnings=[],
            classification=PDFClassification(),
            review_flags=flags,
        )
        assert _dict_to_result(_result_to_dict(original)).review_flags == flags


class TestStreaming:
    def test_streamed_ocr_page_carries_flags(self, tmp_path):
        pytest.importorskip("rapidocr")
        from pdfmux.streaming import process_streaming

        events = list(process_streaming(TestPipelineIntegration()._scan(tmp_path)))
        ocr_pages = [e for e in events if e.type == "page" and e.data.get("ocr_applied")]
        assert ocr_pages, "scan should have been OCR'd"
        flags = ocr_pages[0].data["review_flags"]
        assert any(f["kind"] == IMPLAUSIBLE_DATE and "2096" in f["text"] for f in flags)
        assert "2096" in ocr_pages[0].data["text"]


# ---------------------------------------------------------------------------
# 1.9.0: anchors, ids, ambiguous dates, numbers, tables, key-value provenance
# ---------------------------------------------------------------------------

from pdfmux.ocr_review import (  # noqa: E402
    AMBIGUOUS_DATE,
    REVIEW_NOTE,
    SUSPECT_NUMBER,
    TABLE_CELL,
    OCRLine,
    ambiguous_dates,
    detect_date_order,
    number_problems,
    pipe_tables,
    review_document,
    review_table,
)


class TestAnchorsAndIds:
    def test_flag_carries_line_index_and_bbox(self):
        line = OCRLine(7, "Collected on 12/03/2096", 0.99, (10.0, 20.0, 200.0, 32.0))
        (flag,) = review_page(0, [line], today=TODAY)
        d = flag.to_dict()
        assert d["line_index"] == 7 and d["bbox"] == [10.0, 20.0, 200.0, 32.0]

    def test_ids_are_stable_and_distinct(self):
        lines = [OCRLine(0, "Collected 12/03/2096", 0.99), OCRLine(1, "TSH 2.5", 0.4)]
        a = review_page(0, lines, today=TODAY)
        b = review_page(0, lines, today=TODAY)
        assert [f.id for f in a] == [f.id for f in b]
        assert len({f.id for f in a}) == len(a) == 2
        assert all(len(f.id) == 12 for f in a)

    def test_id_depends_on_page_and_kind(self):
        a = review_page(0, [("Collected 12/03/2096", 0.5)], today=TODAY)
        b = review_page(1, [("Collected 12/03/2096", 0.5)], today=TODAY)
        assert {f.id for f in a}.isdisjoint({f.id for f in b})
        assert len({f.id for f in a}) == 2  # date + low confidence on one line

    def test_identical_lines_get_distinct_ids(self):
        flags = review_page(0, [("TSH 2.5", 0.4), ("TSH 2.5", 0.4)], today=TODAY)
        assert len({f.id for f in flags}) == 2

    def test_ocr_line_to_dict(self):
        assert OCRLine(0, "x", 0.98765, (1.234, 2.0, 3.0, 4.0)).to_dict() == {
            "index": 0,
            "text": "x",
            "score": 0.988,
            "bbox": [1.2, 2.0, 3.0, 4.0],
        }

    def test_review_note_says_unflagged_is_not_verified(self):
        assert "not a verified line" in REVIEW_NOTE and "never corrected" in REVIEW_NOTE


class TestDateOrder:
    @pytest.mark.parametrize(
        "texts,order",
        [
            (["Collected 25/12/2025"], "dmy"),
            (["Collected 12/25/2025"], "mdy"),
            (["03/04/2026"], None),
            (["25/12/2025", "12/25/2025"], "mixed"),
            (["2026-03-04", "12 March 2026"], None),
        ],
    )
    def test_detect(self, texts, order):
        assert detect_date_order(texts) == order

    def test_ambiguous_without_clue(self):
        (reason,) = ambiguous_dates("Date: 03/04/2026", None)
        assert "3 April 2026 or 4 March 2026" in reason

    @pytest.mark.parametrize("line", ["Date: 25/12/2026", "Date: 03/03/2026", "2026-03-04"])
    def test_not_ambiguous(self, line):
        assert ambiguous_dates(line, None) == []

    def test_settled_order_suppresses_ambiguity(self):
        assert ambiguous_dates("Date: 03/04/2026", "dmy") == []
        assert ambiguous_dates("Date: 03/04/2026", "mdy") == []

    def test_mixed_order_says_so(self):
        (reason,) = ambiguous_dates("Date: 03/04/2026", "mixed")
        assert "mixes day-first and month-first" in reason

    def test_settled_order_sharpens_future_check(self):
        # 05/12/2026 is 5 Dec (future) day-first, 12 May (past) month-first.
        assert date_problems("Collected 05/12/2026", TODAY, "dmy")
        assert date_problems("Collected 05/12/2026", TODAY, "mdy") == []
        assert date_problems("Collected 05/12/2026", TODAY, None) == []

    def test_wrong_order_date_is_impossible(self):
        problems = date_problems("Collected 05/13/2026", TODAY, "dmy")
        assert problems and "not a real calendar date" in problems[0]

    def test_document_pass_uses_clue_from_another_page(self):
        from pdfmux.types import PageQuality, PageResult

        pages = [
            PageResult(0, "DOB 25/12/1980", 0.9, PageQuality.GOOD, "fast"),  # digital, clue
            PageResult(1, "Date: 03/04/2026", 0.9, PageQuality.GOOD, "x", ocr_applied=True),
        ]
        assert review_document(pages, today=TODAY) == {1: ()}

    def test_document_pass_flags_ambiguous_without_clue(self):
        from pdfmux.types import PageQuality, PageResult

        pages = [PageResult(0, "Date: 03/04/2026", 0.9, PageQuality.GOOD, "x", ocr_applied=True)]
        (flag,) = review_document(pages, today=TODAY)[0]
        assert flag.kind == AMBIGUOUS_DATE and flag.line_index == 0


class TestNumbers:
    @pytest.mark.parametrize(
        "line,needle",
        [
            ("Glucose (fasting) 8:02 mmol/L 3.9-5.5", '"8:02" may be a decimal point'),
            ("Haemoglobin 1O.2 g/dL", '"1O.2" mixes letters'),
            ("Haemoglobin 13.l g/dL", '"13.l" mixes letters'),
            ("Creatinine I.02 mg/dL", '"I.02" mixes letters'),
        ],
    )
    def test_flagged(self, line, needle):
        problems = number_problems(line, range_check=False)
        assert problems and needle in problems[0]

    @pytest.mark.parametrize(
        "line",
        [
            "Glucose 8.02 mmol/L 3.9-5.5",
            "Collected at 08:30",  # time, no unit
            "Sample time 08:30 ml",  # time label wins even with a unit
            "ANA titre 1:80",
            "ANA 1:160 IU/mL",  # titre shape
            "HbA1c 6.2 %",
            "Vitamin B12 450 pg/mL",
            "Ref no 12:30",  # no unit: not a result line
            "WBC 7.1 x10^3/uL",
        ],
    )
    def test_clean(self, line):
        assert number_problems(line, range_check=False) == []

    def test_range_check_is_opt_in(self, monkeypatch):
        line = "Haemoglobin 132 g/dL 13-17"
        monkeypatch.delenv("PDFMUX_OCR_RANGE_CHECK", raising=False)
        assert number_problems(line) == []
        monkeypatch.setenv("PDFMUX_OCR_RANGE_CHECK", "1")
        (problem,) = number_problems(line)
        assert "Haemoglobin 132" in problem and "13-17" in problem

    @pytest.mark.parametrize(
        "line",
        ["Haemoglobin 13.2 g/dL 13-17", "Ferritin 900 ng/mL 30-400", "Glucose 5.1 mmol/L"],
    )
    def test_range_check_quiet_on_plausible_values(self, line):
        assert number_problems(line, range_check=True) == []

    def test_range_check_low_side(self):
        assert number_problems("Haemoglobin 1.3 g/dL 13-17", range_check=True)

    def test_number_flag_from_page_review(self):
        (flag,) = review_page(0, [OCRLine(3, "Glucose 8:02 mmol/L", 0.99)], today=TODAY)
        assert flag.kind == SUSPECT_NUMBER and flag.line_index == 3
        assert "8:02" in flag.text  # never rewritten to 8.02


H = ["Test", "Result", "Unit", "Range"]
GOOD = [
    ["Haemoglobin", "13.2", "g/dL", "13-17"],
    ["WBC", "7.1", "10^3/uL", "4-11"],
    ["Platelets", "250", "10^3/uL", "150-400"],
    ["MCH", "29.1", "pg", "27-32"],
]


class TestTables:
    def test_clean_table_has_no_flags(self):
        assert review_table(0, 0, H, GOOD) == ()

    def test_word_in_value_column(self):
        rows = [*GOOD, ["RBC", "see note", "10^6/uL", "4.5-5.5"]]
        (flag,) = review_table(0, 0, H, rows)
        assert flag.kind == TABLE_CELL
        assert flag.anchor == {"table": 0, "row": 4, "col": 1}
        assert '"see note" in the number column "Result"' in flag.reason

    def test_shifted_row_is_one_flag(self):
        rows = [*GOOD, ["MCV", "fL", "80-100", ""]]  # value dropped, row slid left
        (flag,) = review_table(0, 0, H, rows)
        assert flag.anchor == {"table": 0, "row": 4}
        assert "moved one column right" in flag.reason

    def test_number_in_unit_column(self):
        rows = [*GOOD, ["MCHC", "g/dL", "33.0", "32-36"]]  # swapped value and unit
        flags = review_table(0, 0, H, rows)
        assert {f.anchor["col"] for f in flags} == {1, 2}

    def test_only_affected_cells_are_flagged(self):
        rows = [*GOOD, ["RBC", "see note", "10^6/uL", "4.5-5.5"], *GOOD]
        flags = review_table(0, 0, H, rows)
        assert [f.anchor["row"] for f in flags] == [4]

    def test_small_or_text_tables_are_skipped(self):
        assert review_table(0, 0, H, GOOD[:2]) == ()
        words = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
        assert review_table(0, 0, ["x", "y"], words) == ()

    def test_flags_like_h_and_l_are_numbers(self):
        rows = [*GOOD, ["LDL", "4.9 H", "mmol/L", "0-3.0"], ["HDL", "<0.9", "mmol/L", ">1.0"]]
        assert review_table(0, 0, H, rows) == ()

    def test_pipe_tables_parse(self):
        text = "Intro\n\n| Test | Result |\n|---|---|\n| Hb | 13.2 |\n| WBC | 7.1 |\n\nAfter"
        assert pipe_tables(text) == [(["Test", "Result"], [["Hb", "13.2"], ["WBC", "7.1"]])]

    def _table_page(self, ocr: bool):
        from pdfmux.types import ExtractedTable, PageQuality, PageResult

        rows = [*GOOD, ["RBC", "see note", "10^6/uL", "4.5-5.5"]]
        table = ExtractedTable(0, tuple(H), tuple(tuple(r) for r in rows), bbox=(1, 2, 3, 4))
        return PageResult(0, "t", 0.9, PageQuality.GOOD, "x", ocr_applied=ocr, tables=(table,))

    def test_document_pass_checks_ocr_tables(self):
        (flag,) = review_document([self._table_page(True)], today=TODAY)[0]
        assert flag.kind == TABLE_CELL and flag.bbox == (1, 2, 3, 4)

    def test_digital_tables_are_not_checked(self):
        # Measured: 313 false flags on 31 digital filings before this rule.
        assert review_document([self._table_page(False)], today=TODAY) == {}

    def test_pipe_table_on_ocr_page(self):
        from pdfmux.types import PageQuality, PageResult

        text = "| Test | Result | Unit | Range |\n|---|---|---|---|\n" + "\n".join(
            "| " + " | ".join(r) + " |" for r in [*GOOD, ["MCV", "fL", "80-100", ""]]
        )
        page = PageResult(0, text, 0.9, PageQuality.GOOD, "x", ocr_applied=True)
        (flag,) = review_document([page], today=TODAY)[0]
        assert flag.anchor == {"table": 0, "row": 4} and "moved one column right" in flag.reason

    @pytest.mark.parametrize(
        "cell",
        ["(356)", "$1,234.50", "12%", "45†", "7¹", "1 234", "−3.2", "( 14 )", "<0.9", "4.9 H"],
    )
    def test_accounting_and_footnote_numbers(self, cell):
        from pdfmux.ocr_review import _cell_type

        assert _cell_type(cell) == "number"

    @pytest.mark.parametrize("cell", ["$", ")%", "%", "—", "†", "(", "  "])
    def test_fragments_are_ignored(self, cell):
        from pdfmux.ocr_review import _cell_type

        assert _cell_type(cell) is None

    def test_financial_table_is_clean(self):
        headers = ["Item", "2024", "2023", "Change"]
        rows = [
            ["Revenue", "$ 391,035", "$ 383,285", "2 %"],
            ["Cost", "(210,352)", "(214,137)", "(2)%"],
            ["Margin", "180,683", "169,148", "7%"],
            ["Tax", "29,749", "16,741", "78 %"],
        ]
        assert review_table(0, 0, headers, rows) == ()


class TestKeyValueProvenance:
    def _page(self, lines, flags=()):
        from pdfmux.types import PageQuality, PageResult

        return PageResult(
            0,
            "\n".join(ln.text for ln in lines),
            0.9,
            PageQuality.GOOD,
            "rapidocr",
            ocr_applied=True,
            ocr_lines=tuple(lines),
            review_flags=tuple(flags),
        )

    def test_low_confidence_field_links_its_flag(self):
        from pdfmux.pipeline import _kv_provenance
        from pdfmux.types import KeyValuePair

        lines = [OCRLine(0, "Policy number: AB-55I20", 0.61), OCRLine(1, "Name: A Person", 0.99)]
        flags = review_page(0, lines, today=TODAY)
        out = _kv_provenance(KeyValuePair("Policy number", "AB-55I20", 0), self._page(lines, flags))
        assert out["confidence"] == 0.61 and out["line_index"] == [0]
        assert out["flags"] == [flags[0].id]

    def test_clean_field_has_high_score_and_no_flags(self):
        from pdfmux.pipeline import _kv_provenance
        from pdfmux.types import KeyValuePair

        lines = [OCRLine(0, "Policy number: AB-55120", 0.99), OCRLine(1, "TSH 2.5", 0.4)]
        flags = review_page(0, lines, today=TODAY)
        out = _kv_provenance(KeyValuePair("Policy number", "AB-55120", 0), self._page(lines, flags))
        assert out == {"flags": [], "confidence": 0.99, "line_index": [0]}

    def test_digital_page_gets_nothing(self):
        from pdfmux.pipeline import _kv_provenance
        from pdfmux.types import KeyValuePair, PageQuality, PageResult

        page = PageResult(0, "Policy number: X", 0.9, PageQuality.GOOD, "fast")
        assert _kv_provenance(KeyValuePair("Policy number", "X", 0), page) == {}


class TestEndToEndAnchors:
    def test_every_flag_resolves_to_an_ocr_line(self, tmp_path):
        pytest.importorskip("rapidocr")
        from pdfmux.pipeline import process

        result = process(TestPipelineIntegration()._scan(tmp_path), output_format="json")
        data = json.loads(result.text)
        lines = {ln["index"]: ln for ln in data["pages"][0]["ocr_lines"]}
        assert data["review_flags"]
        for flag in data["review_flags"]:
            line = lines[flag["line_index"]]
            assert line["text"] == flag["text"] and flag["bbox"] == line["bbox"]
            x0, y0, x1, y1 = flag["bbox"]
            assert 0 <= x0 < x1 <= 595 and 0 <= y0 < y1 <= 842  # on the page, in points
        assert "not a verified line" in data["review_note"]


# ---------------------------------------------------------------------------
# 1.9.1: visual rows and scanned tables rebuilt from OCR box positions
# ---------------------------------------------------------------------------

from pdfmux.ocr_review import find_ocr_tables, visual_rows  # noqa: E402

COLS = (50.0, 220.0, 320.0, 420.0)


def _boxed(rows, top=120.0, step=24.0, score=0.99):
    """OCR lines laid out like a columned lab report: one box per cell."""
    out, i = [], 0
    for r, cells in enumerate(rows):
        y = top + r * step
        for c, text in enumerate(cells):
            if text is None:
                continue
            x = COLS[c]
            out.append(OCRLine(i, text, score, (x, y, x + 8 + 6 * len(text), y + 14)))
            i += 1
    return out


LAB = [
    ("Test", "Result", "Unit", "Reference range"),
    ("Haemoglobin", "13.2", "g/dL", "13.0 - 17.0"),
    ("Glucose (fasting)", "5.1", "mmol/L", "3.9 - 5.5"),
    ("HbA1c", "6.2", "%", "4.0 - 5.6"),
    ("TSH", "2.41", "mIU/L", "0.4 - 4.0"),
    ("Ferritin", "210", "ng/mL", "30 - 400"),
]


def _with(row, col, text):
    rows = [list(r) for r in LAB]
    rows[row][col] = text
    return rows


class TestVisualRows:
    def test_groups_by_baseline_and_sorts_left_to_right(self):
        lines = list(reversed(_boxed(LAB[:2])))
        rows = visual_rows(lines)
        assert [[c.text for c in r] for r in rows] == [list(LAB[0]), list(LAB[1])]

    def test_unboxed_lines_are_ignored(self):
        assert visual_rows([OCRLine(0, "no box", 0.9)]) == []

    def test_slightly_misaligned_boxes_share_a_row(self):
        a = OCRLine(0, "Glucose", 0.99, (50, 100, 100, 114))
        b = OCRLine(1, "5.1", 0.99, (220, 103, 240, 116))  # 3pt lower
        assert len(visual_rows([a, b])) == 1


class TestOcrTables:
    def test_lab_table_is_rebuilt_with_header(self):
        (t,) = find_ocr_tables(_boxed(LAB))
        assert t.header and len(t.rows) == 6 and len(t.columns) == 4
        assert t.rows[2][1].text == "5.1" and t.rows[2][1].score == 0.99
        d = t.to_dict()
        assert d["rows"][2][1]["bbox"] and d["index"] == 0

    def test_clean_table_has_no_flags(self):
        assert review_page(0, _boxed(LAB), today=TODAY) == ()

    def test_word_in_value_column_is_flagged_on_its_cell(self):
        lines = _boxed(_with(4, 1, "see note"))
        (flag,) = review_page(0, lines, today=TODAY)
        cell = next(ln for ln in lines if ln.text == "see note")
        assert flag.kind == TABLE_CELL
        assert flag.anchor == {"ocr_table": 0, "row": 4, "col": 1}
        assert flag.bbox == cell.bbox and flag.line_index == cell.index
        assert "scanned table 1" in flag.reason

    def test_misread_unit_is_flagged(self):
        # Real RapidOCR output for g/dL on a 200 DPI scan, scored 0.952.
        (flag,) = review_page(0, _boxed(_with(1, 2, "7p/6")), today=TODAY)
        assert '"7p/6" in the unit column "Unit"' in flag.reason

    def test_two_columns_or_two_rows_are_not_tables(self):
        assert find_ocr_tables(_boxed([r[:2] for r in LAB])) == []
        assert find_ocr_tables(_boxed(LAB[:2])) == []

    def test_large_vertical_gap_splits_tables(self):
        lines = _boxed(LAB[:4]) + _boxed(LAB[:4], top=600.0)
        lines = [OCRLine(i, ln.text, ln.score, ln.bbox) for i, ln in enumerate(lines)]
        assert len(find_ocr_tables(lines)) == 2

    def test_headerless_table(self):
        (t,) = find_ocr_tables(_boxed(LAB[1:]))
        assert not t.header

    def test_two_boxes_in_one_cell_are_joined(self):
        lines = _boxed(LAB)
        extra = OCRLine(99, "(fasting)", 0.8, (100, 168, 140, 182))  # same cell as "Glucose..."
        (t,) = find_ocr_tables([*lines, extra])
        cell = t.rows[2][0]
        assert "(fasting)" in cell.text and cell.score == 0.8


class TestRowNumbers:
    def test_colon_decimal_across_boxes_carries_the_label(self):
        lines = _boxed(_with(2, 1, "8:02"))
        flags = review_page(0, lines, today=TODAY)
        num = [f for f in flags if f.kind == SUSPECT_NUMBER]
        assert len(num) == 1
        assert num[0].reason.startswith('Glucose (fasting): "8:02" may be a decimal point')
        cell = next(ln for ln in lines if ln.text == "8:02")
        assert num[0].line_index == cell.index and num[0].bbox == cell.bbox
        assert "8:02" in num[0].text  # never rewritten

    def test_same_line_finding_is_not_duplicated(self):
        line = OCRLine(0, "Glucose 8:02 mmol/L", 0.99, (50, 100, 200, 114))
        other = OCRLine(1, "3.9 - 5.5", 0.99, (300, 100, 360, 114))
        num = [f for f in review_page(0, [line, other], today=TODAY) if f.kind == SUSPECT_NUMBER]
        assert len(num) == 1 and num[0].reason.startswith("Glucose:")

    def test_no_unit_in_row_no_flag(self):
        a = OCRLine(0, "Ref no", 0.99, (50, 100, 90, 114))
        b = OCRLine(1, "12:30", 0.99, (220, 100, 250, 114))
        assert review_page(0, [a, b], today=TODAY) == ()

    def test_number_problems_labels_line_findings(self):
        assert number_problems("Glucose 8:02 mmol/L", range_check=False)[0].startswith(
            'Glucose: "8:02"'
        )


class TestScannedLabEndToEnd:
    def _lab_scan(self, tmp_path, rows):
        import fitz

        src = fitz.open()
        page = src.new_page(width=595, height=842)
        for i, r in enumerate(rows):
            for x, c in zip(COLS, r, strict=False):
                page.insert_text((x, 130 + i * 24), c, fontsize=11)
        pix = page.get_pixmap(dpi=200)
        scan = fitz.open()
        sp = scan.new_page(width=595, height=842)
        sp.insert_image(sp.rect, pixmap=pix)
        out = tmp_path / "lab.pdf"
        scan.save(str(out))
        return out

    def test_faults_flagged_correct_cells_not(self, tmp_path):
        pytest.importorskip("rapidocr")
        from pdfmux.pipeline import process

        rows = _with(2, 1, "8:02")
        rows[4][1] = "see note"
        data = json.loads(process(self._lab_scan(tmp_path, rows), output_format="json").text)
        flags = data["review_flags"]
        assert any(
            f["kind"] == SUSPECT_NUMBER and f["reason"].startswith("Glucose (fasting):")
            for f in flags
        )
        cells = {
            (f["anchor"]["row"], f["anchor"].get("col")) for f in flags if f["kind"] == TABLE_CELL
        }
        assert (4, 1) in cells and (2, 1) in cells
        # Correct value cells (rows 1, 3, 5 of the Result column) are never flagged.
        assert not cells & {(1, 1), (3, 1), (5, 1)}
        assert data["pages"][0]["ocr_tables"][0]["rows"][2][1]["text"] == "8:02"
