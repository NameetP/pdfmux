"""Tests for postprocess.py — text cleanup functions."""

from __future__ import annotations

from pdfmux.postprocess import _fix_spaced_text, clean_text


class TestCleanText:
    """Tests for the clean_text function."""

    def test_removes_control_characters(self) -> None:
        """Control characters (except newlines/tabs) should be stripped."""
        text = "Hello\x00World\x07Test\x1fEnd"
        result = clean_text(text)
        assert "Hello" in result
        assert "World" in result
        assert "\x00" not in result
        assert "\x07" not in result

    def test_preserves_newlines_and_tabs(self) -> None:
        """Newlines and tabs should be preserved."""
        text = "Line 1\nLine 2\n\tIndented"
        result = clean_text(text)
        assert "\n" in result
        assert "\t" in result

    def test_collapses_excessive_blank_lines(self) -> None:
        """4+ consecutive newlines should collapse to 3."""
        text = "Para 1\n\n\n\n\n\nPara 2"
        result = clean_text(text)
        assert "\n\n\n\n" not in result
        assert "Para 1" in result
        assert "Para 2" in result

    def test_fixes_broken_hyphenation(self) -> None:
        """Broken words across lines should be joined."""
        text = "The docu-\nment was created"
        result = clean_text(text)
        assert "document" in result

    def test_strips_trailing_whitespace(self) -> None:
        """Trailing whitespace on lines should be removed."""
        text = "Line with trailing spaces   \nAnother line  "
        result = clean_text(text)
        lines = result.split("\n")
        for line in lines:
            assert line == line.rstrip()

    def test_strips_document_edges(self) -> None:
        """Leading and trailing whitespace on the whole doc should be stripped."""
        text = "   \n\nHello World\n\n   "
        result = clean_text(text)
        assert result == "Hello World"

    def test_empty_input(self) -> None:
        """Empty string should return empty string."""
        assert clean_text("") == ""

    def test_clean_text_already_clean(self) -> None:
        """Clean text should pass through unchanged."""
        text = "This is a clean paragraph.\n\nAnother paragraph here."
        result = clean_text(text)
        assert result == text


class TestFixSpacedText:
    """Tests for spaced-out text repair."""

    def test_fixes_spaced_out_text(self) -> None:
        """Spaced-out characters should be joined."""
        text = "W i t h  o v e r  1 7  y e a r s"
        result = _fix_spaced_text(text)
        assert "With" in result
        assert "over" in result

    def test_preserves_normal_text(self) -> None:
        """Normal text should not be modified."""
        text = "This is a normal sentence with regular spacing."
        result = _fix_spaced_text(text)
        assert result == text

    def test_preserves_short_lines(self) -> None:
        """Lines with fewer than 5 words should be left alone."""
        text = "H i"
        result = _fix_spaced_text(text)
        assert result == text

    def test_preserves_empty_lines(self) -> None:
        """Empty lines should pass through."""
        text = "Line one\n\nLine three"
        result = _fix_spaced_text(text)
        assert result == text

    def test_preserves_leading_whitespace(self) -> None:
        """Leading whitespace on spaced lines should be preserved."""
        text = "    W i t h  o v e r  1 7  y e a r s  o f  e x p"
        result = _fix_spaced_text(text)
        assert result.startswith("    ")

    def test_multiline_mixed(self) -> None:
        """Mix of normal and spaced lines should fix only spaced ones."""
        spaced = "A n o t h e r  s p a c e d  o u t  l i n e  h e r e"
        text = f"Normal line here\n{spaced}\nBack to normal"
        result = _fix_spaced_text(text)
        assert "Normal line here" in result
        assert "Back to normal" in result
        # The spaced line should be compacted
        lines = result.split("\n")
        assert len(lines[1]) < len("A n o t h e r  s p a c e d  o u t  l i n e  h e r e")
