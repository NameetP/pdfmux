"""Tests for eval metrics."""

from __future__ import annotations

from pdfmux.eval.metrics import (
    hallucination_rate,
    structure_preservation,
    table_f1,
    text_accuracy,
)

# ---------------------------------------------------------------------------
# text_accuracy
# ---------------------------------------------------------------------------


class TestTextAccuracy:
    def test_identical(self):
        assert text_accuracy("hello world", "hello world") == 1.0

    def test_empty_both(self):
        assert text_accuracy("", "") == 1.0

    def test_empty_extracted(self):
        assert text_accuracy("", "hello world") == 0.0

    def test_empty_ground_truth(self):
        assert text_accuracy("hello world", "") == 0.0

    def test_partial_match(self):
        score = text_accuracy("hello world", "hello earth")
        assert 0.3 < score < 0.9

    def test_whitespace_normalization(self):
        score = text_accuracy("hello   world\n\n", "hello world")
        assert score > 0.9

    def test_case_insensitive(self):
        score = text_accuracy("Hello World", "hello world")
        assert score == 1.0

    def test_long_text_uses_token_overlap(self):
        long_text = "word " * 2000
        score = text_accuracy(long_text, long_text)
        assert score == 1.0


# ---------------------------------------------------------------------------
# structure_preservation
# ---------------------------------------------------------------------------


class TestStructurePreservation:
    def test_identical_structure(self):
        text = "# Heading\n\n- item 1\n- item 2\n\nParagraph."
        assert structure_preservation(text, text) == 1.0

    def test_missing_headings(self):
        extracted = "Some text\n\n- item 1\n- item 2"
        ground_truth = "# Heading\n\nSome text\n\n- item 1\n- item 2"
        score = structure_preservation(extracted, ground_truth)
        assert 0.5 < score < 1.0

    def test_extra_structure(self):
        extracted = "# Heading\n\n## Sub\n\nText"
        ground_truth = "Text"
        score = structure_preservation(extracted, ground_truth)
        assert 0.3 < score < 1.0

    def test_both_empty(self):
        assert structure_preservation("", "") == 1.0

    def test_tables_detected(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        counts = structure_preservation(text, text)
        assert counts == 1.0


# ---------------------------------------------------------------------------
# table_f1
# ---------------------------------------------------------------------------


class TestTableF1:
    def test_identical_table(self):
        table = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
        assert table_f1(table, table) == 1.0

    def test_no_tables_both(self):
        assert table_f1("plain text", "plain text") == 1.0

    def test_missing_table(self):
        assert table_f1("plain text", "| A | B |\n|---|---|\n| 1 | 2 |") == 0.0

    def test_partial_match(self):
        extracted = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
        ground_truth = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        score = table_f1(extracted, ground_truth)
        assert 0.4 < score < 1.0

    def test_cell_normalization(self):
        extracted = "| ALICE | 30 |\n|---|---|"
        ground_truth = "| alice | 30 |\n|---|---|"
        score = table_f1(extracted, ground_truth)
        assert score == 1.0


# ---------------------------------------------------------------------------
# hallucination_rate
# ---------------------------------------------------------------------------


class TestHallucinationRate:
    def test_no_hallucination(self):
        source = "The quick brown fox jumps over the lazy dog"
        extracted = "The quick brown fox"
        rate = hallucination_rate(extracted, source)
        assert rate == 0.0

    def test_all_hallucinated(self):
        source = "apple banana cherry"
        extracted = "xylophone watermelon tangerine"
        rate = hallucination_rate(extracted, source)
        assert rate > 0.5

    def test_empty_extraction(self):
        assert hallucination_rate("", "some source text") == 0.0

    def test_partial_hallucination(self):
        source = "The document discusses financial performance"
        extracted = "The document discusses financial performance and unicorn growth"
        rate = hallucination_rate(extracted, source)
        assert 0.0 < rate < 1.0

    def test_short_words_ignored(self):
        # Short words (<=3 chars) are filtered out
        source = "hello world"
        extracted = "hello world the and"
        rate = hallucination_rate(extracted, source)
        assert rate == 0.0
