"""Error hierarchy — flat, explicit, catch-friendly.

All pdfmux exceptions inherit from PdfmuxError so callers can:

    try:
        pdfmux.extract_text("report.pdf")
    except pdfmux.PdfmuxError as e:
        print(f"pdfmux failed: {e}  (code={e.code})")

Each exception has a `.code` class attribute for programmatic handling.

Every PdfmuxError also carries human-friendly metadata for the CLI:
    .user_message  — what went wrong, in plain language
    .suggestion    — one-line fix the user can try
    .reproduce_cmd — exact command to reproduce the failure (or None)

If user_message is omitted, it falls back to the exception's str() form.

Propagation rules:
    - FileError        → raise immediately, nothing to retry
    - ExtractorNotAvailable → log + try next extractor in registry
    - ExtractionError  → log + try next extractor in registry
    - FormatError      → raise immediately, caller picked bad format
    - AuditError       → log + skip audit, return unaudited result
    - OCRTimeoutError  → log + skip page, mark as unrecovered
"""

from __future__ import annotations


class PdfmuxError(Exception):
    """Base exception for all pdfmux errors.

    Attributes:
        code:           Machine-readable error code.
        user_message:   Human-readable message. Defaults to str(self).
        suggestion:     Optional one-line fix to try.
        reproduce_cmd:  Optional CLI command to reproduce the failure.
    """

    code: str = "PDFMUX_ERROR"

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
        suggestion: str | None = None,
        reproduce_cmd: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or message
        self.suggestion = suggestion
        self.reproduce_cmd = reproduce_cmd

    def format_for_user(self) -> str:
        """Render a multi-line, user-facing error block."""
        lines = [self.user_message]
        if self.suggestion:
            lines.append(f"  Suggestion: {self.suggestion}")
        if self.reproduce_cmd:
            lines.append(f"  Reproduce:  {self.reproduce_cmd}")
        return "\n".join(lines)


class FileError(PdfmuxError):
    """PDF file not found, not a PDF, corrupted, or unreadable.

    Specific codes:
        PDF_NOT_FOUND  — file does not exist
        PDF_CORRUPTED  — file exists but can't be opened as PDF
        PDF_ENCRYPTED  — file is password-protected
        PDF_INVALID    — file is not a PDF at all
    """

    code: str = "PDF_NOT_FOUND"

    def __init__(
        self,
        message: str,
        *,
        code: str = "PDF_NOT_FOUND",
        user_message: str | None = None,
        suggestion: str | None = None,
        reproduce_cmd: str | None = None,
    ) -> None:
        super().__init__(
            message,
            user_message=user_message,
            suggestion=suggestion,
            reproduce_cmd=reproduce_cmd,
        )
        self.code = code


class ExtractionError(PdfmuxError):
    """An extractor failed to produce output.

    Codes:
        EXTRACTION_ERROR    — general extraction failure
        PARTIAL_EXTRACTION  — some pages failed but others succeeded
    """

    code: str = "EXTRACTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXTRACTION_ERROR",
        user_message: str | None = None,
        suggestion: str | None = None,
        reproduce_cmd: str | None = None,
    ) -> None:
        super().__init__(
            message,
            user_message=user_message,
            suggestion=suggestion,
            reproduce_cmd=reproduce_cmd,
        )
        self.code = code


class ExtractorNotAvailable(PdfmuxError):  # noqa: N818
    """An optional extractor dependency is not installed.

    Includes install instructions in the message:
        ExtractorNotAvailable("RapidOCR not installed. pip install pdfmux[ocr]")
    """

    code: str = "NO_EXTRACTOR"


class FormatError(PdfmuxError):
    """Invalid or unsupported output format requested."""

    code: str = "FORMAT_ERROR"


class AuditError(PdfmuxError):
    """Per-page quality audit failed (non-fatal in pipeline)."""

    code: str = "AUDIT_ERROR"


class OCRTimeoutError(PdfmuxError):
    """OCR processing exceeded time limit."""

    code: str = "OCR_TIMEOUT"


# ---------------------------------------------------------------------------
# Helpers — build common error messages with suggestions/repro commands
# ---------------------------------------------------------------------------


def file_not_found(path: str) -> FileError:
    return FileError(
        f"PDF not found: {path}",
        code="PDF_NOT_FOUND",
        user_message=f"PDF not found: {path}",
        suggestion="Check the path and try again. Use an absolute path if unsure.",
        reproduce_cmd=f"pdfmux convert {path}",
    )


def file_not_pdf(path: str) -> FileError:
    return FileError(
        f"Not a PDF file: {path}",
        code="PDF_INVALID",
        user_message=f"Not a PDF file: {path}",
        suggestion="Make sure the file ends in .pdf and isn't a renamed image or text file.",
        reproduce_cmd=f"pdfmux convert {path}",
    )


def corrupted_page(path: str, page_num: int) -> ExtractionError:
    """Page-level extraction failed, likely corrupted PDF stream."""
    return ExtractionError(
        f"PyMuPDF couldn't parse page {page_num} of {path}",
        code="EXTRACTION_ERROR",
        user_message=(
            f"PyMuPDF couldn't parse page {page_num} of {path} — "
            f"the page may be corrupted or use an unsupported format."
        ),
        suggestion="Try `--mode premium` to use the LLM vision fallback.",
        reproduce_cmd=f"pdfmux convert {path} --mode premium",
    )


def extractor_missing(extractor: str, install_hint: str) -> ExtractorNotAvailable:
    """Build an ExtractorNotAvailable with a clean install hint."""
    return ExtractorNotAvailable(
        f"{extractor} not installed. {install_hint}",
        suggestion=install_hint,
    )
