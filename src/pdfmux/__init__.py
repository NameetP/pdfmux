"""pdfmux — PDF extraction that checks its own work.

Public API:
    extract_text(path)       → Markdown string
    extract_json(path)       → dict with locked schema
    load_llm_context(path)   → list of chunk dicts with token estimates

Types:
    Quality, OutputFormat, PageQuality, PageResult, DocumentResult, Chunk

Errors:
    PdfmuxError, FileError, ExtractionError, ExtractorNotAvailable,
    FormatError, AuditError
"""

from __future__ import annotations

# Suppress pymupdf4llm "Consider using pymupdf_layout" noise on import
import io as _io
import sys as _sys
from pathlib import Path

_orig = _sys.stdout
_sys.stdout = _io.StringIO()
try:
    import pymupdf4llm as _pmll  # noqa: F401
except ImportError:
    pass
finally:
    _sys.stdout = _orig
del _orig, _io

__version__ = "1.5.1"
__all__ = [
    # Public API
    "extract_text",
    "extract_json",
    "load_llm_context",
    "chunk",
    # Types
    "Quality",
    "OutputFormat",
    "PageQuality",
    "PageResult",
    "DocumentResult",
    "Chunk",
    "PageLayout",
    "WeakRegion",
    # Errors
    "PdfmuxError",
    "FileError",
    "ExtractionError",
    "ExtractorNotAvailable",
    "FormatError",
    "AuditError",
    "OCRTimeoutError",
]

# Re-export types for convenience: import pdfmux; pdfmux.PageResult(...)
from pdfmux.errors import (  # noqa: E402, F401
    AuditError,
    ExtractionError,
    ExtractorNotAvailable,
    FileError,
    FormatError,
    OCRTimeoutError,
    PdfmuxError,
)
from pdfmux.types import (  # noqa: E402, F401
    Chunk,
    DocumentResult,
    OutputFormat,
    PageLayout,
    PageQuality,
    PageResult,
    Quality,
    WeakRegion,
)


def extract_text(
    path: str | Path,
    *,
    quality: str = "standard",
) -> str:
    """Extract text from a PDF as Markdown.

    Args:
        path: Path to the PDF file.
        quality: "fast", "standard" (default), or "high".

    Returns:
        Markdown text extracted from the PDF.

    Raises:
        FileError: If the file doesn't exist or isn't a PDF.
        PdfmuxError: On extraction failures.

    Example::

        import pdfmux
        text = pdfmux.extract_text("report.pdf")
        print(text[:200])
    """
    from pdfmux.pipeline import process

    result = process(file_path=path, output_format="markdown", quality=quality)
    return result.text


def extract_json(
    path: str | Path,
    *,
    quality: str = "standard",
) -> dict:
    """Extract text from a PDF as a structured dictionary.

    Returns the locked JSON schema with metadata, pages, and confidence.

    Args:
        path: Path to the PDF file.
        quality: "fast", "standard" (default), or "high".

    Returns:
        Dictionary with keys: schema_version, source, converter, extractor,
        page_count, confidence, warnings, ocr_pages, content, pages.

    Raises:
        FileError: If the file doesn't exist or isn't a PDF.
        PdfmuxError: On extraction failures.

    Example::

        import pdfmux
        data = pdfmux.extract_json("report.pdf")
        print(f"{data['page_count']} pages, {data['confidence']:.0%}")
    """
    import json

    from pdfmux.pipeline import process

    result = process(file_path=path, output_format="json", quality=quality)
    return json.loads(result.text)


def load_llm_context(
    path: str | Path,
    *,
    quality: str = "standard",
) -> list[dict]:
    """Extract text from a PDF as LLM-ready chunks.

    Returns section-aware chunks with token estimates, designed
    for RAG pipelines and context windows.

    Args:
        path: Path to the PDF file.
        quality: "fast", "standard" (default), or "high".

    Returns:
        List of chunk dicts, each with: title, text, page_start,
        page_end, tokens, confidence.

    Raises:
        FileError: If the file doesn't exist or isn't a PDF.
        PdfmuxError: On extraction failures.

    Example::

        import pdfmux
        chunks = pdfmux.load_llm_context("report.pdf")
        for c in chunks:
            print(f"{c['title']}: {c['tokens']} tokens")
    """
    import json

    from pdfmux.pipeline import process

    result = process(file_path=path, output_format="llm", quality=quality)
    data = json.loads(result.text)
    return data["chunks"]


def chunk(
    path: str | Path,
    *,
    quality: str = "standard",
    max_tokens: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Extract and chunk a PDF for RAG pipelines.

    Layout-aware chunking that respects document structure.
    Splits at heading boundaries, enforces token limits,
    and adds overlap between adjacent chunks.

    Args:
        path: Path to the PDF file.
        quality: "fast", "standard" (default), or "high".
        max_tokens: Maximum tokens per chunk.
        overlap: Token overlap between adjacent chunks.

    Returns:
        List of chunk dicts, each with: index, title, text,
        page_start, page_end, tokens, confidence.

    Example::

        import pdfmux
        chunks = pdfmux.chunk("report.pdf", max_tokens=500)
        for c in chunks:
            print(f"[{c['tokens']} tok] {c['title']}")
    """
    from pdfmux.chunking import chunk_for_rag
    from pdfmux.pipeline import process

    result = process(file_path=path, output_format="markdown", quality=quality)

    chunks = chunk_for_rag(
        result.text,
        confidence=result.confidence,
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        extractor=result.extractor_used,
    )

    return [
        {
            "index": i,
            "title": c.title,
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "tokens": c.tokens,
            "confidence": c.confidence,
        }
        for i, c in enumerate(chunks)
    ]
