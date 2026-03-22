"""Tests for extractors."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfmux.errors import ExtractorNotAvailable
from pdfmux.extractors.fast import FastExtractor
from pdfmux.extractors.llm import LLMExtractor
from pdfmux.extractors.ocr import OCRExtractor
from pdfmux.extractors.opendataloader import OpenDataLoaderExtractor
from pdfmux.extractors.tables import TableExtractor


class TestFastExtractor:
    """Tests for the PyMuPDF fast extractor."""

    def test_extract_digital_pdf(self, digital_pdf: Path) -> None:
        """Should extract PageResults from a digital PDF."""
        ext = FastExtractor()
        pages = list(ext.extract(digital_pdf))
        assert pages
        text = "\n".join(p.text for p in pages)
        assert len(text) > 50

    def test_extract_empty_pdf(self, empty_pdf: Path) -> None:
        """Should handle empty PDFs gracefully."""
        ext = FastExtractor()
        pages = list(ext.extract(empty_pdf))
        assert isinstance(pages, list)

    def test_extractor_name(self) -> None:
        """Should have a descriptive name."""
        ext = FastExtractor()
        assert ext.name == "pymupdf4llm"

    def test_extract_text_convenience(self, digital_pdf: Path) -> None:
        """extract_text() should return a full string."""
        ext = FastExtractor()
        text = ext.extract_text(digital_pdf)
        assert isinstance(text, str)
        assert len(text) > 50


class TestTableExtractor:
    """Tests for the table extractor (requires docling optional dep)."""

    def test_requires_docling(self, digital_pdf: Path) -> None:
        """Should raise ExtractorNotAvailable when docling is not installed."""
        ext = TableExtractor()
        if ext.available():
            pytest.skip("Docling is installed")
        with pytest.raises(ExtractorNotAvailable, match="Docling is not installed"):
            list(ext.extract(digital_pdf))

    def test_extractor_name(self) -> None:
        ext = TableExtractor()
        assert ext.name == "docling"


class TestOCRExtractor:
    """Tests for the OCR extractor (requires surya optional dep)."""

    def test_requires_surya(self, digital_pdf: Path) -> None:
        """Should raise ExtractorNotAvailable when surya is not installed."""
        ext = OCRExtractor()
        if ext.available():
            pytest.skip("Surya OCR is installed")
        with pytest.raises(ExtractorNotAvailable, match="Surya OCR is not installed"):
            list(ext.extract(digital_pdf))

    def test_extractor_name(self) -> None:
        ext = OCRExtractor()
        assert ext.name == "surya"


class TestOpenDataLoaderExtractor:
    """Tests for the OpenDataLoader extractor (requires opendataloader-pdf optional dep)."""

    def test_requires_opendataloader(self, digital_pdf: Path) -> None:
        """Should raise ExtractorNotAvailable when opendataloader-pdf is not installed."""
        ext = OpenDataLoaderExtractor()
        if ext.available():
            pytest.skip("OpenDataLoader is installed")
        with pytest.raises(ExtractorNotAvailable, match="OpenDataLoader-PDF is not installed"):
            list(ext.extract(digital_pdf))

    def test_extractor_name(self) -> None:
        ext = OpenDataLoaderExtractor()
        assert ext.name == "opendataloader"

    def test_registry_priority(self) -> None:
        """OpenDataLoader should be registered at priority 15 (between fast and rapidocr)."""
        from pdfmux.extractors import _REGISTRY

        odl_entry = [(p, n) for p, n, _ in _REGISTRY if n == "opendataloader"]
        assert odl_entry, "opendataloader not found in registry"
        priority = odl_entry[0][0]
        assert priority == 15, f"Expected priority 15, got {priority}"


class TestLLMExtractor:
    """Tests for the LLM extractor (requires google-genai optional dep)."""

    def test_requires_genai(self, digital_pdf: Path) -> None:
        """Should raise ExtractorNotAvailable when google-genai is not installed."""
        ext = LLMExtractor()
        if ext.available():
            pytest.skip("Google GenAI is installed")
        with pytest.raises(ExtractorNotAvailable, match="Google GenAI is not installed"):
            list(ext.extract(digital_pdf))

    def test_extractor_name(self) -> None:
        ext = LLMExtractor()
        assert ext.name == "gemini-flash"
