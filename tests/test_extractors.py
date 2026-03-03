"""Tests for extractors."""

from __future__ import annotations

from pathlib import Path

import pytest

from readable.extractors.fast import FastExtractor
from readable.extractors.llm import LLMExtractor
from readable.extractors.ocr import OCRExtractor
from readable.extractors.tables import TableExtractor


class TestFastExtractor:
    """Tests for the PyMuPDF fast extractor."""

    def test_extract_digital_pdf(self, digital_pdf: Path) -> None:
        """Should extract text from a digital PDF."""
        ext = FastExtractor()
        text = ext.extract(digital_pdf)
        assert text
        assert len(text) > 50

    def test_extract_empty_pdf(self, empty_pdf: Path) -> None:
        """Should handle empty PDFs gracefully."""
        ext = FastExtractor()
        text = ext.extract(empty_pdf)
        assert isinstance(text, str)

    def test_extractor_name(self) -> None:
        """Should have a descriptive name."""
        ext = FastExtractor()
        assert ext.name == "pymupdf4llm (fast)"


class TestTableExtractor:
    """Tests for the table extractor (requires docling optional dep)."""

    def test_requires_docling(self, digital_pdf: Path) -> None:
        """Should raise ImportError when docling is not installed."""
        ext = TableExtractor()
        with pytest.raises(ImportError, match="Docling is not installed"):
            ext.extract(digital_pdf)

    def test_extractor_name(self) -> None:
        ext = TableExtractor()
        assert ext.name == "docling (tables)"


class TestOCRExtractor:
    """Tests for the OCR extractor (requires surya optional dep)."""

    def test_requires_surya(self, digital_pdf: Path) -> None:
        """Should raise ImportError when surya is not installed."""
        ext = OCRExtractor()
        with pytest.raises(ImportError, match="Surya OCR is not installed"):
            ext.extract(digital_pdf)

    def test_extractor_name(self) -> None:
        ext = OCRExtractor()
        assert ext.name == "surya (OCR)"


class TestLLMExtractor:
    """Tests for the LLM extractor (requires google-genai optional dep)."""

    def test_requires_genai(self, digital_pdf: Path) -> None:
        """Should raise ImportError when google-genai is not installed."""
        ext = LLMExtractor()
        with pytest.raises(ImportError, match="Google GenAI is not installed"):
            ext.extract(digital_pdf)

    def test_extractor_name(self) -> None:
        ext = LLMExtractor()
        assert ext.name == "gemini-flash (LLM)"
