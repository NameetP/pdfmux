"""Tests for the RapidOCR extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfmux.errors import ExtractorNotAvailable
from pdfmux.extractors.rapid_ocr import RapidOCRExtractor, _check_rapidocr


class TestRapidOCRCheck:
    """Tests for RapidOCR availability checking."""

    def test_check_returns_bool(self) -> None:
        """_check_rapidocr should return a boolean."""
        result = _check_rapidocr()
        assert isinstance(result, bool)


class TestRapidOCRExtractor:
    """Tests for the RapidOCR extractor class."""

    def test_extractor_name(self) -> None:
        """Should have a descriptive name."""
        ext = RapidOCRExtractor()
        assert ext.name == "rapidocr"

    def test_extract_digital_pdf(self, digital_pdf: Path) -> None:
        """Should extract PageResults from a digital PDF via OCR."""
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        pages = list(ext.extract(digital_pdf))
        assert isinstance(pages, list)
        assert len(pages) > 0
        # Each page should be a PageResult
        for page in pages:
            assert hasattr(page, "text")
            assert hasattr(page, "page_num")
            assert page.ocr_applied is True

    def test_extract_page(self, digital_pdf: Path) -> None:
        """Should extract text from a single page."""
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        text = ext.extract_page(digital_pdf, 0)
        assert isinstance(text, str)

    def test_extract_empty_pdf(self, empty_pdf: Path) -> None:
        """Should handle empty PDFs gracefully."""
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        pages = list(ext.extract(empty_pdf))
        assert isinstance(pages, list)

    def test_unavailable_raises_on_extract(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise ExtractorNotAvailable when rapidocr is not available."""
        monkeypatch.setattr("pdfmux.extractors.rapid_ocr._check_rapidocr", lambda: False)
        ext = RapidOCRExtractor()
        # Constructor doesn't raise (lazy init) — error comes when extracting
        with pytest.raises(ExtractorNotAvailable, match="RapidOCR is not installed"):
            list(ext.extract(digital_pdf))

    def test_engine_cached(self) -> None:
        """Engine should be created once and cached."""
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        assert hasattr(ext, "_engine")
        assert ext._engine is not None
