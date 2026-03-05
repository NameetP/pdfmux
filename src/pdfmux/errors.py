"""Error hierarchy — flat, explicit, catch-friendly.

All pdfmux exceptions inherit from PdfmuxError so callers can:

    try:
        pdfmux.extract_text("report.pdf")
    except pdfmux.PdfmuxError as e:
        print(f"pdfmux failed: {e}")

Propagation rules:
    - FileError        → raise immediately, nothing to retry
    - ExtractorNotAvailable → log + try next extractor in registry
    - ExtractionError  → log + try next extractor in registry
    - FormatError      → raise immediately, caller picked bad format
    - AuditError       → log + skip audit, return unaudited result
"""

from __future__ import annotations


class PdfmuxError(Exception):
    """Base exception for all pdfmux errors."""


class FileError(PdfmuxError):
    """PDF file not found, not a PDF, corrupted, or unreadable."""


class ExtractionError(PdfmuxError):
    """An extractor failed to produce output."""


class ExtractorNotAvailable(PdfmuxError):  # noqa: N818
    """An optional extractor dependency is not installed.

    Includes install instructions in the message:
        ExtractorNotAvailable("RapidOCR not installed. pip install pdfmux[ocr]")
    """


class FormatError(PdfmuxError):
    """Invalid or unsupported output format requested."""


class AuditError(PdfmuxError):
    """Per-page quality audit failed (non-fatal in pipeline)."""
