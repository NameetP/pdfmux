"""Tests for dynamic quality scoring."""

from __future__ import annotations

from pdfmux.router.scorer import (
    _completeness_signal,
    _consistency_signal,
    _structure_signal,
    _text_coherence,
    score_llm_output,
)

# ---------------------------------------------------------------------------
# Overall scorer
# ---------------------------------------------------------------------------


class TestScoreLLMOutput:
    def test_empty_text_is_zero(self):
        assert score_llm_output("") == 0.0
        assert score_llm_output("   ") == 0.0
        assert score_llm_output("ab") == 0.0

    def test_good_text_scores_high(self):
        text = """# Annual Report 2025

## Financial Summary

Revenue increased by 15% year-over-year to $2.4 billion.
Operating expenses were reduced through efficiency improvements.

| Metric | 2024 | 2025 |
|--------|------|------|
| Revenue | $2.1B | $2.4B |
| Profit | $400M | $520M |

Key highlights:
- Strong growth in cloud services
- Expansion into new markets
- 30% reduction in customer churn
"""
        score = score_llm_output(text)
        assert score > 0.70

    def test_garbled_text_scores_low(self):
        text = "asd fgh jkl qwe rty uio zxc vbn m12 345 !@# $%^ &*("
        score = score_llm_output(text)
        assert score < 0.60

    def test_with_fast_text_reference(self):
        llm = "The company reported revenue of $2.4 billion in 2025."
        fast = "The company reported revenue of $2.4 billion in 2025."
        score = score_llm_output(llm, fast_text=fast)
        assert score > 0.70

    def test_without_fast_text(self):
        llm = "The company reported revenue of $2.4 billion in 2025."
        score = score_llm_output(llm, fast_text=None)
        assert score > 0.50

    def test_score_capped_at_one(self):
        text = "# Title\n\n" + "Word " * 500 + "\n\n- Item 1\n- Item 2"
        score = score_llm_output(text)
        assert score <= 1.0

    def test_score_never_negative(self):
        text = "\ufffd\ufffd\ufffd" * 20
        score = score_llm_output(text)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Text coherence
# ---------------------------------------------------------------------------


class TestTextCoherence:
    def test_clean_english(self):
        score = _text_coherence("The quick brown fox jumps over the lazy dog.")
        assert score > 0.80

    def test_low_alpha_ratio(self):
        score = _text_coherence("12345 67890 !@#$% ^&*()")
        assert score <= 0.70

    def test_mojibake(self):
        text = (
            "Hello \ufffd\ufffd\ufffd world \ufffd\ufffd\ufffd more \ufffd\ufffd\ufffd text \ufffd"
        )
        score = _text_coherence(text)
        assert score < 0.80

    def test_repetitive_text(self):
        text = " ".join(["hello"] * 50 + ["world"] * 5)
        score = _text_coherence(text)
        assert score < 0.90

    def test_very_short_words(self):
        text = "a b c d e f g h i j k l m n o p q r s t"
        score = _text_coherence(text)
        assert score < 0.90


# ---------------------------------------------------------------------------
# Structure signal
# ---------------------------------------------------------------------------


class TestStructureSignal:
    def test_rich_structure(self):
        text = "# Heading\n\n- Item 1\n- Item 2\n\n| A | B |\n|---|---|\n\n**bold**"
        score = _structure_signal(text)
        assert score > 0.80

    def test_plain_text(self):
        text = "Just a plain paragraph with no markdown elements at all."
        score = _structure_signal(text)
        assert 0.30 < score < 0.70

    def test_headings_only(self):
        text = "# Title\n\nSome text here."
        score = _structure_signal(text)
        assert score > 0.50

    def test_empty(self):
        score = _structure_signal("")
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Completeness signal
# ---------------------------------------------------------------------------


class TestCompletenessSignal:
    def test_full_page(self):
        text = "Word " * 300  # ~1500 chars
        score = _completeness_signal(text)
        assert score > 0.80

    def test_sparse_page(self):
        text = "Short text."
        score = _completeness_signal(text)
        assert score < 0.50

    def test_empty_page(self):
        assert _completeness_signal("") == 0.0

    def test_half_page(self):
        text = "Word " * 150  # ~750 chars
        score = _completeness_signal(text)
        assert 0.60 < score < 1.0


# ---------------------------------------------------------------------------
# Consistency signal
# ---------------------------------------------------------------------------


class TestConsistencySignal:
    def test_identical_texts(self):
        text = "The annual report shows strong growth in all segments."
        score = _consistency_signal(text, text)
        assert score > 0.90

    def test_completely_different(self):
        llm = "Quantum physics explores subatomic particles."
        fast = "The financial report discusses revenue trends."
        score = _consistency_signal(llm, fast)
        assert score < 0.30

    def test_partial_overlap(self):
        llm = "Revenue grew by 15% to reach $2.4 billion in total sales for the fiscal year."
        fast = "Revenue grew by 15% to reach $2.4 billion in total sales for the annual period."
        score = _consistency_signal(llm, fast)
        assert 0.30 < score < 0.95

    def test_empty_fast_text(self):
        score = _consistency_signal("Some LLM output", "")
        assert score == 0.5  # neutral

    def test_empty_llm_text(self):
        score = _consistency_signal("", "Some fast text")
        assert score == 0.5  # neutral
