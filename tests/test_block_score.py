"""Tests for block-level quality scoring."""

from __future__ import annotations

from pdfmux.audit import score_block


class TestScoreBlock:
    """Tests for score_block() function."""

    def test_good_block(self) -> None:
        """Normal English text should score high."""
        text = (
            "This is a perfectly normal paragraph of text that contains meaningful English words."
        )
        score = score_block(text)
        assert score >= 0.7

    def test_bad_block_mostly_numbers(self) -> None:
        """Block of mostly numbers/symbols should score lower."""
        text = "12345 67890 #$%^& ()[][] 98765 43210 $$$"
        score = score_block(text)
        assert score <= 0.7

    def test_empty_block(self) -> None:
        """Empty or near-empty block should score 0."""
        assert score_block("") == 0.0
        assert score_block("   ") == 0.0
        assert score_block("ab") == 0.0  # < 5 chars
