"""Tests for the extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdfmux.errors import FormatError
from pdfmux.pipeline import process


def test_process_digital_pdf(digital_pdf: Path) -> None:
    """Processing a digital PDF should return Markdown text."""
    result = process(digital_pdf)
    assert result.text
    assert result.format == "markdown"
    assert result.page_count == 2
    assert result.confidence > 0
    assert result.extractor_used == "pymupdf4llm"
    assert result.ocr_pages == []  # Digital PDF needs no OCR


def test_process_with_confidence(digital_pdf: Path) -> None:
    """--confidence flag should add confidence info to output."""
    result = process(digital_pdf, show_confidence=True)
    assert "confidence" in result.text.lower() or "Conversion confidence" in result.text


def test_process_fast_quality(digital_pdf: Path) -> None:
    """Fast quality should always use PyMuPDF."""
    result = process(digital_pdf, quality="fast")
    assert result.extractor_used == "pymupdf4llm"


def test_process_high_quality_fallback(digital_pdf: Path) -> None:
    """High quality without LLM installed should fall back gracefully."""
    result = process(digital_pdf, quality="high")
    # Without google-genai installed, should fall back to fast extractor
    assert result.text
    assert result.confidence > 0


def test_process_unsupported_format(digital_pdf: Path) -> None:
    """Unsupported output formats should raise FormatError."""
    with pytest.raises(FormatError, match="Unknown output format"):
        process(digital_pdf, output_format="xml")


def test_process_json_format(digital_pdf: Path) -> None:
    """JSON format should return valid structured JSON."""
    result = process(digital_pdf, output_format="json")
    data = json.loads(result.text)
    assert data["converter"] == "pdfmux"
    assert data["page_count"] == 2
    assert data["confidence"] > 0
    assert "content" in data
    assert "pages" in data
    assert len(data["pages"]) >= 1


def test_process_csv_no_tables(digital_pdf: Path) -> None:
    """CSV format on a non-table PDF should raise ValueError."""
    with pytest.raises(ValueError, match="No tables found"):
        process(digital_pdf, output_format="csv")


def test_process_multi_page(multi_page_pdf: Path) -> None:
    """Multi-page PDFs should process correctly."""
    result = process(multi_page_pdf)
    assert result.page_count == 5
    assert len(result.text) > 100


def test_process_json_metadata(multi_page_pdf: Path) -> None:
    """JSON output should include correct metadata."""
    result = process(multi_page_pdf, output_format="json")
    data = json.loads(result.text)
    assert data["page_count"] == 5
    assert data["extractor"] == "pymupdf4llm"
    assert isinstance(data["warnings"], list)


def test_process_llm_format(digital_pdf: Path) -> None:
    """LLM format should return valid chunked JSON."""
    result = process(digital_pdf, output_format="llm")
    data = json.loads(result.text)
    assert "document" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) > 0
    # Each chunk should have required fields
    chunk = data["chunks"][0]
    assert "title" in chunk
    assert "text" in chunk
    assert "page_start" in chunk
    assert "page_end" in chunk
    assert "tokens" in chunk
    assert "confidence" in chunk


def test_process_json_has_ocr_pages(digital_pdf: Path) -> None:
    """JSON format should include ocr_pages field."""
    result = process(digital_pdf, output_format="json")
    data = json.loads(result.text)
    assert "ocr_pages" in data
    assert isinstance(data["ocr_pages"], list)
    # Digital PDF should not need OCR
    assert data["ocr_pages"] == []


def test_process_json_has_schema_version(digital_pdf: Path) -> None:
    """JSON format should include schema_version 0.6.0."""
    result = process(digital_pdf, output_format="json")
    data = json.loads(result.text)
    assert data["schema_version"] == "0.6.0"
