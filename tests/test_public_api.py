"""Tests for the public Python API — extract_text, extract_json, load_llm_context."""

from __future__ import annotations

from pathlib import Path

import pdfmux


def test_extract_text_returns_string(digital_pdf: Path) -> None:
    """extract_text() should return a non-empty string."""
    text = pdfmux.extract_text(digital_pdf)
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_text_fast_quality(digital_pdf: Path) -> None:
    """extract_text() with quality=fast should work."""
    text = pdfmux.extract_text(digital_pdf, quality="fast")
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_text_standard_quality(digital_pdf: Path) -> None:
    """extract_text() with quality=standard should work."""
    text = pdfmux.extract_text(digital_pdf, quality="standard")
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_json_returns_dict(digital_pdf: Path) -> None:
    """extract_json() should return a dict, not a JSON string."""
    data = pdfmux.extract_json(digital_pdf)
    assert isinstance(data, dict)


def test_extract_json_schema_version(digital_pdf: Path) -> None:
    """extract_json() should include schema_version 0.5.0."""
    data = pdfmux.extract_json(digital_pdf)
    assert data["schema_version"] == "0.5.0"


def test_extract_json_has_ocr_pages(digital_pdf: Path) -> None:
    """extract_json() should include ocr_pages field."""
    data = pdfmux.extract_json(digital_pdf)
    assert "ocr_pages" in data
    assert isinstance(data["ocr_pages"], list)


def test_extract_json_required_fields(digital_pdf: Path) -> None:
    """extract_json() should include all required schema fields."""
    data = pdfmux.extract_json(digital_pdf)
    required_fields = [
        "schema_version",
        "source",
        "converter",
        "extractor",
        "page_count",
        "confidence",
        "warnings",
        "ocr_pages",
        "content",
        "pages",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_extract_json_page_count(digital_pdf: Path) -> None:
    """extract_json() page_count should match actual PDF pages."""
    data = pdfmux.extract_json(digital_pdf)
    assert data["page_count"] == 2


def test_load_llm_context_returns_list(digital_pdf: Path) -> None:
    """load_llm_context() should return a list of dicts."""
    chunks = pdfmux.load_llm_context(digital_pdf)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], dict)


def test_load_llm_context_chunk_schema(digital_pdf: Path) -> None:
    """Each chunk should have the correct fields."""
    chunks = pdfmux.load_llm_context(digital_pdf)
    required_fields = ["title", "text", "page_start", "page_end", "tokens", "confidence"]
    for chunk in chunks:
        for field in required_fields:
            assert field in chunk, f"Chunk missing field: {field}"


def test_load_llm_context_tokens_positive(digital_pdf: Path) -> None:
    """Token counts should be positive integers."""
    chunks = pdfmux.load_llm_context(digital_pdf)
    for chunk in chunks:
        assert isinstance(chunk["tokens"], int)
        assert chunk["tokens"] > 0


def test_load_llm_context_pages_1_indexed(digital_pdf: Path) -> None:
    """Page numbers should be 1-indexed."""
    chunks = pdfmux.load_llm_context(digital_pdf)
    for chunk in chunks:
        assert chunk["page_start"] >= 1
        assert chunk["page_end"] >= chunk["page_start"]


def test_version_is_0_5_0() -> None:
    """Module version should be 0.5.0."""
    assert pdfmux.__version__ == "0.5.0"


def test_all_exports() -> None:
    """__all__ should expose the three public functions and types."""
    assert "extract_text" in pdfmux.__all__
    assert "extract_json" in pdfmux.__all__
    assert "load_llm_context" in pdfmux.__all__
    # v0.5.0 also exports types and errors
    assert "PageResult" in pdfmux.__all__
    assert "PdfmuxError" in pdfmux.__all__
