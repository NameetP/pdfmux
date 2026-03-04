"""Tests for section-aware chunking + token estimation."""

from __future__ import annotations

import pytest

from pdfmux.chunking import chunk_by_sections, estimate_tokens


class TestEstimateTokens:
    """Tests for estimate_tokens()."""

    def test_basic_text(self) -> None:
        """Normal text should return chars/4."""
        assert estimate_tokens("hello world") == 2  # 11 chars // 4

    def test_longer_text(self) -> None:
        """Longer text should scale proportionally."""
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_empty_string(self) -> None:
        """Empty string should return 1 (minimum)."""
        assert estimate_tokens("") == 1

    def test_whitespace_only(self) -> None:
        """Whitespace-only should return 1 (minimum after strip)."""
        assert estimate_tokens("   \n\t  ") == 1

    def test_single_char(self) -> None:
        """Single character should return 1 (minimum)."""
        assert estimate_tokens("a") == 1


class TestChunkBySections:
    """Tests for chunk_by_sections()."""

    def test_empty_input(self) -> None:
        """Empty text should return empty list."""
        assert chunk_by_sections("") == []
        assert chunk_by_sections("   ") == []

    def test_single_heading(self) -> None:
        """Single heading should produce one chunk."""
        text = "# Introduction\n\nThis is the intro text."
        chunks = chunk_by_sections(text)
        assert len(chunks) == 1
        assert chunks[0].title == "Introduction"
        assert "intro text" in chunks[0].text
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1
        assert chunks[0].tokens > 0

    def test_multiple_headings(self) -> None:
        """Multiple headings should split into multiple chunks."""
        text = (
            "# Chapter 1\n\nFirst chapter content.\n\n"
            "## Section 1.1\n\nSection content here.\n\n"
            "# Chapter 2\n\nSecond chapter content."
        )
        chunks = chunk_by_sections(text)
        assert len(chunks) == 3
        assert chunks[0].title == "Chapter 1"
        assert chunks[1].title == "Section 1.1"
        assert chunks[2].title == "Chapter 2"

    def test_no_headings_page_fallback(self) -> None:
        """Text without headings should fall back to page-based chunks."""
        text = "Page one content.\n\n---\n\nPage two content.\n\n---\n\nPage three content."
        chunks = chunk_by_sections(text)
        assert len(chunks) == 3
        assert chunks[0].title == "Page 1"
        assert chunks[1].title == "Page 2"
        assert chunks[2].title == "Page 3"

    def test_no_headings_single_page(self) -> None:
        """Single page without headings should produce one chunk."""
        text = "Just some plain text without any headings or page separators."
        chunks = chunk_by_sections(text)
        assert len(chunks) == 1
        assert chunks[0].title == "Page 1"
        assert "plain text" in chunks[0].text

    def test_page_tracking_across_separators(self) -> None:
        """Sections spanning page boundaries should track page_start and page_end."""
        text = (
            "# Introduction\n\nStart of intro.\n\n---\n\n"
            "Continuation of intro.\n\n"
            "# Methods\n\nMethods text."
        )
        chunks = chunk_by_sections(text)
        assert len(chunks) == 2
        # Introduction starts on page 1, continues onto page 2
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2
        # Methods is on page 2
        assert chunks[1].page_start == 2
        assert chunks[1].page_end == 2

    def test_token_estimation_in_chunks(self) -> None:
        """Each chunk should have a positive token estimate."""
        text = "# Title\n\n" + "word " * 100
        chunks = chunk_by_sections(text)
        assert len(chunks) == 1
        assert chunks[0].tokens > 0
        # ~500 chars / 4 ≈ 125 tokens
        assert chunks[0].tokens > 50

    def test_confidence_passthrough(self) -> None:
        """Document confidence should be inherited by all chunks."""
        text = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
        chunks = chunk_by_sections(text, confidence=0.85)
        assert all(c.confidence == 0.85 for c in chunks)

    def test_confidence_default(self) -> None:
        """Default confidence should be 1.0."""
        text = "# Section\n\nContent."
        chunks = chunk_by_sections(text)
        assert chunks[0].confidence == 1.0

    def test_chunk_is_frozen(self) -> None:
        """Chunk dataclass should be immutable."""
        text = "# Title\n\nContent."
        chunks = chunk_by_sections(text)
        with pytest.raises(AttributeError):
            chunks[0].title = "new title"  # type: ignore[misc]

    def test_empty_pages_skipped(self) -> None:
        """Empty pages in page-based fallback should be skipped."""
        text = "Page one content.\n\n---\n\n\n\n---\n\nPage three content."
        chunks = chunk_by_sections(text)
        assert len(chunks) == 2
        assert chunks[0].title == "Page 1"
        assert chunks[1].title == "Page 3"

    def test_heading_levels(self) -> None:
        """All heading levels (h1-h6) should be detected."""
        text = "# H1\n\nContent 1.\n\n## H2\n\nContent 2.\n\n### H3\n\nContent 3."
        chunks = chunk_by_sections(text)
        assert len(chunks) == 3
        assert chunks[0].title == "H1"
        assert chunks[1].title == "H2"
        assert chunks[2].title == "H3"
