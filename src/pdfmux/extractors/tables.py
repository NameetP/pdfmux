"""Table extractor — Docling for PDFs with complex tables.

Uses IBM's Docling library for 97.9% table accuracy.
Slower than PyMuPDF but dramatically better on structured documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _check_docling() -> bool:
    """Check if docling is installed."""
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


class TableExtractor:
    """Extract tables from PDFs using Docling."""

    @property
    def name(self) -> str:
        return "docling (tables)"

    def extract(self, file_path: str | Path, pages: list[int] | None = None) -> str:
        """Extract text and tables from a PDF using Docling.

        Args:
            file_path: Path to the PDF file.
            pages: Optional list of 0-indexed page numbers to extract.

        Returns:
            Markdown text with accurately extracted tables.
        """
        if not _check_docling():
            raise ImportError("Docling is not installed. Install with: pip install pdfmux[tables]")

        from docling.document_converter import DocumentConverter

        file_path = Path(file_path)
        converter = DocumentConverter()
        result = converter.convert(str(file_path))

        # Export to markdown — Docling handles table formatting
        markdown = result.document.export_to_markdown()

        if pages is not None:
            # Docling processes the whole doc; we filter pages via markers
            # For now return the full markdown (page filtering is complex with Docling)
            logger.info("Page filtering with Docling extracts full document; filtering deferred")

        return markdown
