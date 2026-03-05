"""Table extractor — Docling for PDFs with complex tables.

Uses IBM's Docling library for 97.9% table accuracy.
Slower than PyMuPDF but dramatically better on structured documents.

Install: pip install pdfmux[tables]
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


def _check_docling() -> bool:
    """Check if docling is installed."""
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


@register(name="docling", priority=40)
class TableExtractor:
    """Extract tables from PDFs using Docling.

    Docling processes the full document at once (not per-page),
    so we yield a single PageResult with page_num=0 for the
    full document text, then synthetic pages if page separators
    are found in the output.
    """

    @property
    def name(self) -> str:
        return "docling"

    def available(self) -> bool:
        return _check_docling()

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield PageResults from Docling extraction.

        Docling extracts the full document at once. We split the
        output into pages based on content structure.
        """
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Docling is not installed. Install with: pip install pdfmux[tables]"
            )

        from docling.document_converter import DocumentConverter

        file_path = Path(file_path)
        converter = DocumentConverter()
        result = converter.convert(str(file_path))

        markdown = result.document.export_to_markdown()

        if pages is not None:
            logger.info("Page filtering with Docling: extracting full document")

        # Split on page separators if present, otherwise treat as single page
        page_texts = markdown.split("\n\n---\n\n") if "\n\n---\n\n" in markdown else [markdown]

        for i, text in enumerate(page_texts):
            if pages is not None and i not in pages:
                continue

            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=i,
                text=text,
                confidence=0.95 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
            )
