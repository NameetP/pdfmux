"""Tests for structured error codes."""

from __future__ import annotations

from pathlib import Path

import pdfmux
from pdfmux.errors import (
    AuditError,
    ExtractionError,
    ExtractorNotAvailable,
    FileError,
    FormatError,
    OCRTimeoutError,
    PdfmuxError,
)


class TestErrorCodes:
    """All exceptions should have a .code attribute."""

    def test_base_error_has_code(self) -> None:
        err = PdfmuxError("test")
        assert hasattr(err, "code")
        assert err.code == "PDFMUX_ERROR"

    def test_file_error_default_code(self) -> None:
        err = FileError("file not found")
        assert err.code == "PDF_NOT_FOUND"

    def test_file_error_custom_code(self) -> None:
        err = FileError("corrupted", code="PDF_CORRUPTED")
        assert err.code == "PDF_CORRUPTED"

    def test_extraction_error_code(self) -> None:
        err = ExtractionError("failed")
        assert err.code == "EXTRACTION_ERROR"

    def test_extraction_error_partial_code(self) -> None:
        err = ExtractionError("partial", code="PARTIAL_EXTRACTION")
        assert err.code == "PARTIAL_EXTRACTION"

    def test_extractor_not_available_code(self) -> None:
        err = ExtractorNotAvailable("missing")
        assert err.code == "NO_EXTRACTOR"

    def test_format_error_code(self) -> None:
        err = FormatError("bad format")
        assert err.code == "FORMAT_ERROR"

    def test_audit_error_code(self) -> None:
        err = AuditError("audit failed")
        assert err.code == "AUDIT_ERROR"

    def test_ocr_timeout_error_code(self) -> None:
        err = OCRTimeoutError("timeout")
        assert err.code == "OCR_TIMEOUT"

    def test_all_errors_catchable_by_base(self) -> None:
        """All errors should be catchable by PdfmuxError."""
        for err_cls in [
            FileError,
            ExtractionError,
            ExtractorNotAvailable,
            FormatError,
            AuditError,
            OCRTimeoutError,
        ]:
            err = err_cls("test")
            assert isinstance(err, PdfmuxError)


class TestErrorCodesInJSON:
    """JSON output should include error_code field."""

    def test_json_has_error_code_field(self, digital_pdf: Path) -> None:
        """JSON output includes error_code (null on success)."""
        data = pdfmux.extract_json(digital_pdf)
        assert "error_code" in data
        assert data["error_code"] is None  # success = null

    def test_json_pages_have_ocr_flag(self, digital_pdf: Path) -> None:
        """Each page in JSON output has an 'ocr' boolean."""
        data = pdfmux.extract_json(digital_pdf)
        for page in data["pages"]:
            assert "ocr" in page
            assert isinstance(page["ocr"], bool)
            assert page["ocr"] is False  # digital PDF, no OCR

    def test_ocr_timeout_exported(self) -> None:
        """OCRTimeoutError should be importable from pdfmux."""
        assert hasattr(pdfmux, "OCRTimeoutError")
        assert pdfmux.OCRTimeoutError is OCRTimeoutError


class TestUserFacingFields:
    """PdfmuxError carries human-friendly metadata for the CLI."""

    def test_default_user_message_falls_back_to_str(self) -> None:
        err = PdfmuxError("something broke")
        assert err.user_message == "something broke"
        assert err.suggestion is None
        assert err.reproduce_cmd is None

    def test_explicit_user_message_overrides(self) -> None:
        err = PdfmuxError(
            "oops",
            user_message="PyMuPDF couldn't parse page 5",
            suggestion="Try --mode premium",
            reproduce_cmd="pdfmux convert foo.pdf --mode premium",
        )
        assert err.user_message == "PyMuPDF couldn't parse page 5"
        assert err.suggestion == "Try --mode premium"
        assert err.reproduce_cmd == "pdfmux convert foo.pdf --mode premium"

    def test_format_for_user_includes_suggestion(self) -> None:
        err = PdfmuxError(
            "oops",
            user_message="Bad page",
            suggestion="Try premium",
            reproduce_cmd="pdfmux convert foo.pdf",
        )
        formatted = err.format_for_user()
        assert "Bad page" in formatted
        assert "Try premium" in formatted
        assert "pdfmux convert foo.pdf" in formatted

    def test_format_for_user_only_message(self) -> None:
        err = PdfmuxError("hello")
        assert err.format_for_user() == "hello"

    def test_file_error_carries_metadata(self) -> None:
        err = FileError(
            "missing",
            code="PDF_NOT_FOUND",
            user_message="PDF not found",
            suggestion="Check the path",
            reproduce_cmd="pdfmux convert x.pdf",
        )
        assert err.code == "PDF_NOT_FOUND"
        assert err.suggestion == "Check the path"

    def test_helpers_produce_actionable_errors(self) -> None:
        from pdfmux.errors import corrupted_page, file_not_found

        e1 = file_not_found("/tmp/x.pdf")
        assert e1.code == "PDF_NOT_FOUND"
        assert e1.suggestion is not None
        assert "x.pdf" in (e1.reproduce_cmd or "")

        e2 = corrupted_page("/tmp/x.pdf", 5)
        assert "page 5" in e2.user_message
        assert "premium" in (e2.suggestion or "")
