"""Tests for agentic multi-pass extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pdfmux.agentic import (
    _estimate_cost,
    _find_page,
    _get_fallback_extractors,
    agentic_improve,
)
from pdfmux.types import PageQuality, PageResult


def _make_page(
    page_num: int, confidence: float, text: str = "Some text", extractor: str = "pymupdf"
) -> PageResult:
    return PageResult(
        page_num=page_num,
        text=text,
        confidence=confidence,
        quality=PageQuality.GOOD if confidence > 0.5 else PageQuality.BAD,
        extractor=extractor,
    )


class TestAgenticImprove:
    def test_all_pages_above_threshold(self):
        """No re-extraction when all pages are good."""
        pages = [_make_page(0, 0.95), _make_page(1, 0.90), _make_page(2, 0.85)]
        improved, name, passes = agentic_improve(pages, Path("test.pdf"), "pymupdf")
        assert passes == 1
        assert improved == pages
        assert name == "pymupdf"

    def test_skips_empty_pages(self):
        """Don't try to improve pages with no text."""
        pages = [
            _make_page(0, 0.95),
            PageResult(
                page_num=1,
                text="",
                confidence=0.0,
                quality=PageQuality.EMPTY,
                extractor="pymupdf",
            ),
        ]
        improved, name, passes = agentic_improve(pages, Path("test.pdf"), "pymupdf")
        assert passes == 1  # no re-extraction attempted

    def test_improves_low_confidence_page(self):
        """Should replace low-confidence page with better result."""
        pages = [_make_page(0, 0.95), _make_page(1, 0.40, text="garbled")]

        better_page = _make_page(1, 0.88, text="clean text", extractor="docling")

        with (
            patch("pdfmux.agentic._get_fallback_extractors", return_value=["docling"]),
            patch("pdfmux.agentic._extract_pages_with", return_value=[better_page]),
        ):
            improved, name, passes = agentic_improve(pages, Path("test.pdf"), "pymupdf")

        assert improved[1].confidence == 0.88
        assert improved[1].extractor == "docling"
        assert passes == 2

    def test_keeps_original_if_fallback_worse(self):
        """Don't replace if re-extraction is worse."""
        pages = [_make_page(0, 0.60, text="somewhat ok")]

        worse_page = _make_page(0, 0.30, text="worse", extractor="rapidocr")

        with (
            patch("pdfmux.agentic._get_fallback_extractors", return_value=["rapidocr"]),
            patch("pdfmux.agentic._extract_pages_with", return_value=[worse_page]),
        ):
            improved, name, passes = agentic_improve(pages, Path("test.pdf"), "pymupdf")

        assert improved[0].confidence == 0.60  # kept original
        assert improved[0].extractor == "pymupdf"

    def test_respects_budget(self):
        """Should stop when budget is exhausted."""
        pages = [_make_page(0, 0.40), _make_page(1, 0.30)]

        with (
            patch("pdfmux.agentic._get_fallback_extractors", return_value=["llm"]),
            patch("pdfmux.agentic._estimate_cost", return_value=0.01),
        ):
            improved, name, passes = agentic_improve(
                pages, Path("test.pdf"), "pymupdf", budget=0.005
            )

        # Budget too low for 2 pages at $0.01/page — should stop
        assert passes == 1

    def test_respects_max_passes(self):
        """Should not exceed max_passes."""
        pages = [_make_page(0, 0.40)]

        with (
            patch("pdfmux.agentic._get_fallback_extractors", return_value=["a", "b", "c", "d"]),
            patch("pdfmux.agentic._extract_pages_with", return_value=[_make_page(0, 0.45)]),
        ):
            improved, name, passes = agentic_improve(
                pages, Path("test.pdf"), "pymupdf", max_passes=2
            )

        assert passes <= 2

    def test_handles_fallback_failure(self):
        """Should continue to next fallback if one fails."""
        pages = [_make_page(0, 0.40)]

        better = _make_page(0, 0.90, extractor="llm")

        def mock_extract(fp, ext_name, pn):
            if ext_name == "docling":
                raise RuntimeError("Docling crashed")
            return [better]

        with (
            patch("pdfmux.agentic._get_fallback_extractors", return_value=["docling", "llm"]),
            patch("pdfmux.agentic._extract_pages_with", side_effect=mock_extract),
        ):
            improved, name, passes = agentic_improve(pages, Path("test.pdf"), "pymupdf")

        assert improved[0].confidence == 0.90
        assert improved[0].extractor == "llm"

    def test_multiple_pages_mixed_quality(self):
        """Only re-extracts pages that need it."""
        pages = [
            _make_page(0, 0.95),  # good
            _make_page(1, 0.40),  # bad
            _make_page(2, 0.92),  # good
            _make_page(3, 0.30),  # bad
        ]

        def mock_extract(fp, ext_name, page_nums):
            return [_make_page(pn, 0.85, extractor="docling") for pn in page_nums]

        with (
            patch("pdfmux.agentic._get_fallback_extractors", return_value=["docling"]),
            patch("pdfmux.agentic._extract_pages_with", side_effect=mock_extract),
        ):
            improved, name, passes = agentic_improve(pages, Path("test.pdf"), "pymupdf")

        # Pages 0 and 2 should be unchanged
        assert improved[0].confidence == 0.95
        assert improved[2].confidence == 0.92
        # Pages 1 and 3 should be improved
        assert improved[1].confidence == 0.85
        assert improved[3].confidence == 0.85


class TestFallbackExtractors:
    def test_excludes_current(self):
        with patch(
            "pdfmux.extractors.available_extractors",
            return_value=[
                ("fast", MagicMock()),
                ("opendataloader", MagicMock()),
                ("docling", MagicMock()),
            ],
        ):
            fallbacks = _get_fallback_extractors("opendataloader")
            assert "opendataloader" not in fallbacks
            assert "fast" not in fallbacks

    def test_llm_comes_last(self):
        with patch(
            "pdfmux.extractors.available_extractors",
            return_value=[
                ("fast", MagicMock()),
                ("rapidocr", MagicMock()),
                ("llm", MagicMock()),
            ],
        ):
            fallbacks = _get_fallback_extractors("fast")
            if "llm" in fallbacks:
                assert fallbacks[-1] == "llm"


class TestHelpers:
    def test_find_page(self):
        pages = [_make_page(0, 0.9), _make_page(1, 0.8), _make_page(2, 0.7)]
        assert _find_page(pages, 1).page_num == 1
        assert _find_page(pages, 5) is None

    def test_estimate_cost(self):
        assert _estimate_cost("pymupdf") == 0.0
        assert _estimate_cost("llm") == 0.01
        assert _estimate_cost("unknown") == 0.0
