"""API stability tests — verify the v1.0 contract.

These tests lock the public API surface. If a 1.x release changes
a function signature or removes a JSON field, these tests will fail.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pdfmux
from pdfmux.pipeline import process


class TestPublicAPISignatures:
    """Lock public API function signatures."""

    def test_extract_text_signature(self) -> None:
        """extract_text(path, *, quality) — locked for 1.x."""
        sig = inspect.signature(pdfmux.extract_text)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "quality" in params

    def test_extract_json_signature(self) -> None:
        """extract_json(path, *, quality) — locked for 1.x."""
        sig = inspect.signature(pdfmux.extract_json)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "quality" in params

    def test_load_llm_context_signature(self) -> None:
        """load_llm_context(path, *, quality) — locked for 1.x."""
        sig = inspect.signature(pdfmux.load_llm_context)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "quality" in params

    def test_process_signature(self) -> None:
        """process(file_path, output_format, quality, show_confidence) — locked."""
        sig = inspect.signature(process)
        params = list(sig.parameters.keys())
        assert "file_path" in params
        assert "output_format" in params
        assert "quality" in params
        assert "show_confidence" in params


class TestJSONSchemaStability:
    """Lock JSON output schema fields."""

    def test_json_schema_version(self, digital_pdf: Path) -> None:
        """Schema version should be 1.0.0."""
        data = pdfmux.extract_json(digital_pdf)
        assert data["schema_version"] == "1.1.0"

    def test_json_required_fields(self, digital_pdf: Path) -> None:
        """All v1.0 schema fields must be present."""
        data = pdfmux.extract_json(digital_pdf)
        required = [
            "schema_version",
            "source",
            "converter",
            "extractor",
            "page_count",
            "confidence",
            "error_code",
            "warnings",
            "ocr_pages",
            "content",
            "pages",
        ]
        for field in required:
            assert field in data, f"Missing required field: {field}"

    def test_llm_chunk_fields(self, digital_pdf: Path) -> None:
        """LLM chunks must have all v1.0 fields."""
        chunks = pdfmux.load_llm_context(digital_pdf)
        assert len(chunks) > 0
        required = [
            "title",
            "text",
            "page_start",
            "page_end",
            "tokens",
            "confidence",
            "extractor",
            "ocr_applied",
        ]
        for chunk in chunks:
            for field in required:
                assert field in chunk, f"Chunk missing field: {field}"


class TestExportStability:
    """Lock __all__ exports."""

    def test_types_exported(self) -> None:
        """Core types must be in __all__."""
        for name in [
            "Quality",
            "OutputFormat",
            "PageQuality",
            "PageResult",
            "DocumentResult",
            "Chunk",
            "PageLayout",
            "WeakRegion",
        ]:
            assert name in pdfmux.__all__, f"Missing export: {name}"

    def test_errors_exported(self) -> None:
        """All errors must be in __all__."""
        for name in [
            "PdfmuxError",
            "FileError",
            "ExtractionError",
            "ExtractorNotAvailable",
            "FormatError",
            "AuditError",
            "OCRTimeoutError",
        ]:
            assert name in pdfmux.__all__, f"Missing export: {name}"

    def test_functions_exported(self) -> None:
        """Public API functions must be in __all__."""
        for name in ["extract_text", "extract_json", "load_llm_context"]:
            assert name in pdfmux.__all__, f"Missing export: {name}"
