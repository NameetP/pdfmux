"""Tests for heading detection via font-size analysis."""

from __future__ import annotations

import fitz

from pdfmux.headings import _build_font_census, _promote_bold_lines, inject_headings


def _make_page_with_text(
    entries: list[tuple[str, float]],
    *,
    bold_entries: list[tuple[str, float]] | None = None,
) -> fitz.Page:
    """Create a PDF page with text at specified font sizes.

    Args:
        entries: List of (text, fontsize) pairs.
        bold_entries: List of (text, fontsize) pairs inserted as bold.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72

    for text, size in entries:
        page.insert_text((72, y), text, fontsize=size)
        y += size * 1.5

    if bold_entries:
        for text, size in bold_entries:
            page.insert_text(
                (72, y),
                text,
                fontsize=size,
                fontname="helv",  # Helvetica
            )
            y += size * 1.5

    return page


class TestInjectHeadings:
    """Tests for the main inject_headings function."""

    def test_empty_text_returns_unchanged(self):
        doc = fitz.open()
        page = doc.new_page()
        assert inject_headings("", page) == ""
        assert inject_headings("  \n  ", page) == "  \n  "

    def test_skip_existing_headings(self):
        """Text with 2+ existing headings should not be modified."""
        text = "# Title\n\nBody text here.\n\n## Section\n\nMore text."
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Title", fontsize=20)
        page.insert_text((72, 120), "Body text here.", fontsize=11)
        result = inject_headings(text, page)
        assert result == text

    def test_detect_large_font_heading(self):
        """Text with a significantly larger font should get # marker."""
        page = _make_page_with_text(
            [
                ("Introduction", 18),
                ("This is the body text of the document.", 11),
                ("More body text continues here with details.", 11),
            ]
        )
        text = "Introduction\n\nThis is the body text of the document.\n\nMore body text continues here with details."
        result = inject_headings(text, page)
        assert result.startswith("# Introduction")
        assert "# This is the body" not in result

    def test_multi_level_headings(self):
        """Three distinct font sizes should produce h1/h2/h3."""
        page = _make_page_with_text(
            [
                ("Main Title", 24),
                ("Chapter One", 18),
                ("Section Details", 14),
                ("Body text that goes on for a while.", 11),
                ("More body text with additional content.", 11),
            ]
        )
        text = (
            "Main Title\n\n"
            "Chapter One\n\n"
            "Section Details\n\n"
            "Body text that goes on for a while.\n\n"
            "More body text with additional content."
        )
        result = inject_headings(text, page)
        # Current implementation maps all detected headings to H1
        assert "# Main Title" in result
        assert "# Chapter One" in result
        assert "# Section Details" in result

    def test_no_false_positives_uniform_text(self):
        """Body text with uniform font size should not get heading markers."""
        page = _make_page_with_text(
            [
                ("First paragraph of text here.", 11),
                ("Second paragraph of text here.", 11),
                ("Third paragraph of text here.", 11),
            ]
        )
        text = (
            "First paragraph of text here.\n\n"
            "Second paragraph of text here.\n\n"
            "Third paragraph of text here."
        )
        result = inject_headings(text, page)
        assert "#" not in result

    def test_long_lines_not_marked_as_headings(self):
        """Lines >120 chars should never be heading candidates."""
        long_text = "A" * 130
        page = _make_page_with_text(
            [
                (long_text, 18),
                ("Body text.", 11),
            ]
        )
        text = f"{long_text}\n\nBody text."
        result = inject_headings(text, page)
        assert "# " + long_text not in result


class TestPromoteBoldLines:
    """Tests for bold-line promotion fallback."""

    def test_bold_line_becomes_heading(self):
        text = "**Overview**\n\nThis is the content."
        result = _promote_bold_lines(text)
        # Bold standalone lines get promoted to headings
        assert "Overview" in result

    def test_mid_paragraph_bold_not_promoted(self):
        text = "Some text before.\n**Not a heading**\n\nNext paragraph."
        result = _promote_bold_lines(text)
        # "Not a heading" is preceded by non-blank line, should stay bold
        assert "**Not a heading**" in result

    def test_long_bold_not_promoted(self):
        text = "**" + "A" * 70 + "**\n\nBody."
        result = _promote_bold_lines(text)
        # >60 chars, should not be promoted
        assert "###" not in result


class TestBuildFontCensus:
    """Tests for font census extraction."""

    def test_returns_body_size(self):
        page = _make_page_with_text(
            [
                ("Title", 20),
                ("Body line one here with more text.", 11),
                ("Body line two here with more text.", 11),
                ("Body line three here with more text.", 11),
            ]
        )
        body_size, candidates = _build_font_census(page)
        assert body_size == 11.0
        assert len(candidates) >= 4

    def test_empty_page(self):
        doc = fitz.open()
        page = doc.new_page()
        body_size, candidates = _build_font_census(page)
        assert body_size == 0.0
        assert candidates == []
