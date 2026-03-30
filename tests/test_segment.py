"""Tests for segment-level detection."""

from __future__ import annotations

from pdfmux.segment import (
    Segment,
    SegmentType,
    _bbox_overlap,
    _looks_like_formula,
    _looks_like_table,
    is_mixed_content,
    page_segment_summary,
)

# ---------------------------------------------------------------------------
# Segment classification
# ---------------------------------------------------------------------------


class TestLooksLikeTable:
    def test_pipe_delimited(self):
        text = "Name | Age | City\nAlice | 30 | NYC\nBob | 25 | LA"
        assert _looks_like_table(text) is True

    def test_tab_separated(self):
        text = "Name\tAge\tCity\nAlice\t30\tNYC\nBob\t25\tLA"
        assert _looks_like_table(text) is True

    def test_number_columns(self):
        text = "Q1 $1,200 45.2%\nQ2 $1,500 52.1%\nQ3 $1,800 61.3%\nQ4 $2,100 70.0%"
        assert _looks_like_table(text) is True

    def test_plain_paragraph(self):
        text = "This is a normal paragraph of text without any table structure."
        assert _looks_like_table(text) is False

    def test_single_line(self):
        assert _looks_like_table("just one line") is False


class TestLooksLikeFormula:
    def test_latex(self):
        assert _looks_like_formula(r"E = mc^2 where \frac{1}{2}mv^2") is True

    def test_math_symbols(self):
        assert _looks_like_formula("∑ x = ∫ f(x)dx ± ∞ ≈ ∂y/∂x") is True

    def test_equation(self):
        assert _looks_like_formula("y = mx + b") is True

    def test_plain_text(self):
        assert _looks_like_formula("The company reported strong earnings.") is False

    def test_long_text_with_equals(self):
        # A long paragraph with = sign should NOT be classified as formula
        text = "The value of x = 5 in this context means that " + "word " * 50
        assert _looks_like_formula(text) is False


# ---------------------------------------------------------------------------
# BBox overlap
# ---------------------------------------------------------------------------


class TestBboxOverlap:
    def test_identical(self):
        bbox = (0, 0, 100, 100)
        assert _bbox_overlap(bbox, bbox) == 1.0

    def test_no_overlap(self):
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        assert _bbox_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        overlap = _bbox_overlap(a, b)
        assert 0.2 < overlap < 0.3  # 50*50 / 100*100 = 0.25

    def test_contained(self):
        a = (25, 25, 75, 75)
        b = (0, 0, 100, 100)
        assert _bbox_overlap(a, b) == 1.0  # A fully inside B

    def test_zero_area(self):
        a = (50, 50, 50, 50)  # zero-area box
        b = (0, 0, 100, 100)
        assert _bbox_overlap(a, b) == 0.0


# ---------------------------------------------------------------------------
# Page analysis
# ---------------------------------------------------------------------------


class TestPageSegmentSummary:
    def test_empty(self):
        assert page_segment_summary([]) == {}

    def test_mixed(self):
        segments = [
            Segment(SegmentType.TEXT, (0, 0, 100, 50), 0, "text"),
            Segment(SegmentType.TEXT, (0, 50, 100, 100), 0, "more text"),
            Segment(SegmentType.TABLE, (0, 100, 100, 200), 0, "table"),
            Segment(SegmentType.IMAGE, (0, 200, 100, 300), 0),
        ]
        summary = page_segment_summary(segments)
        assert summary["text"] == 2
        assert summary["table"] == 1
        assert summary["image"] == 1


class TestIsMixedContent:
    def test_text_only(self):
        segments = [
            Segment(SegmentType.TEXT, (0, 0, 100, 50), 0, "text"),
            Segment(SegmentType.TEXT, (0, 50, 100, 100), 0, "more"),
        ]
        assert is_mixed_content(segments) is False

    def test_text_plus_table(self):
        segments = [
            Segment(SegmentType.TEXT, (0, 0, 100, 50), 0, "text"),
            Segment(SegmentType.TABLE, (0, 50, 100, 150), 0, "table"),
        ]
        assert is_mixed_content(segments) is True

    def test_text_plus_image(self):
        segments = [
            Segment(SegmentType.TEXT, (0, 0, 100, 50), 0, "text"),
            Segment(SegmentType.IMAGE, (0, 50, 100, 150), 0),
        ]
        assert is_mixed_content(segments) is True

    def test_three_types(self):
        segments = [
            Segment(SegmentType.TEXT, (0, 0, 100, 50), 0, "text"),
            Segment(SegmentType.TABLE, (0, 50, 100, 100), 0, "table"),
            Segment(SegmentType.IMAGE, (0, 100, 100, 200), 0),
        ]
        assert is_mixed_content(segments) is True

    def test_header_footer_ignored(self):
        segments = [
            Segment(SegmentType.HEADER, (0, 0, 100, 20), 0, "header"),
            Segment(SegmentType.TEXT, (0, 20, 100, 80), 0, "body"),
            Segment(SegmentType.FOOTER, (0, 80, 100, 100), 0, "footer"),
        ]
        assert is_mixed_content(segments) is False


class TestSegmentDataclass:
    def test_creation(self):
        seg = Segment(
            segment_type=SegmentType.TABLE,
            bbox=(10, 20, 300, 400),
            page_num=0,
            text="| A | B |",
            area=110200.0,
        )
        assert seg.segment_type == SegmentType.TABLE
        assert seg.page_num == 0
        assert seg.area == 110200.0

    def test_frozen(self):
        seg = Segment(SegmentType.TEXT, (0, 0, 1, 1), 0)
        try:
            seg.text = "changed"
            assert False, "Should be frozen"
        except AttributeError:
            pass
