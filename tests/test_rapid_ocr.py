"""Tests for the RapidOCR extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

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
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        assert ext.name == "rapidocr (OCR)"

    def test_extract_digital_pdf(self, digital_pdf: Path) -> None:
        """Should extract text from a digital PDF via OCR."""
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        text = ext.extract(digital_pdf)
        assert isinstance(text, str)
        assert "Page 1" in text or "Page 2" in text

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
        text = ext.extract(empty_pdf)
        assert isinstance(text, str)
        assert "No text detected" in text

    def test_import_error_without_rapidocr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ImportError when rapidocr is not available."""
        monkeypatch.setattr(
            "pdfmux.extractors.rapid_ocr._check_rapidocr", lambda: False
        )
        with pytest.raises(ImportError, match="RapidOCR is not installed"):
            RapidOCRExtractor()

    def test_engine_cached(self) -> None:
        """Engine should be created once and cached."""
        if not _check_rapidocr():
            pytest.skip("RapidOCR not installed")
        ext = RapidOCRExtractor()
        assert hasattr(ext, "_engine")
        assert ext._engine is not None
