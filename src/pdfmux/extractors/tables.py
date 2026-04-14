"""Table extractor — Docling for PDFs with complex tables.

Uses IBM's Docling library for 97.9% table accuracy.
Slower than PyMuPDF but dramatically better on structured documents.

Install: pip install pdfmux[tables]
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import fitz

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)

# Module-level singleton with thread safety
_converter_lock = threading.Lock()
_converter_instance = None


def _check_docling() -> bool:
    """Check if docling is installed."""
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def _get_converter():
    """Lazy-load and cache the Docling DocumentConverter.

    Thread-safe singleton. The converter loads transformer models
    on first use (~5-10s), then reuses them for all subsequent calls.
    """
    global _converter_instance
    if _converter_instance is None:
        with _converter_lock:
            if _converter_instance is None:
                from docling.document_converter import DocumentConverter

                _converter_instance = DocumentConverter()
    return _converter_instance


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
        """Yield PageResults from Docling extraction."""
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Docling is not installed. Install with: pip install pdfmux[tables]"
            )

        file_path = Path(file_path)
        converter = _get_converter()
        result = converter.convert(str(file_path))

        markdown = result.document.export_to_markdown()

        if pages is not None:
            logger.info("Page filtering with Docling: extracting full document")

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

    def extract_pages(
        self,
        file_path: str | Path,
        page_nums: list[int],
    ) -> Iterator[PageResult]:
        """Extract specific pages using Docling.

        Creates a temporary PDF subset with only the requested pages,
        then runs Docling on the subset. Avoids processing the entire
        document when only certain pages need table extraction.
        """
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Docling is not installed. Install with: pip install pdfmux[tables]"
            )

        file_path = Path(file_path)

        # Create a temporary PDF with only the requested pages
        doc = fitz.open(str(file_path))
        subset_doc = fitz.open()
        for pn in sorted(page_nums):
            if pn < len(doc):
                subset_doc.insert_pdf(doc, from_page=pn, to_page=pn)
        doc.close()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            subset_doc.save(tmp.name)
            subset_doc.close()
            tmp_path = Path(tmp.name)

        try:
            converter = _get_converter()
            result = converter.convert(str(tmp_path))
            markdown = result.document.export_to_markdown()
            page_texts = markdown.split("\n\n---\n\n") if "\n\n---\n\n" in markdown else [markdown]

            sorted_pages = sorted(page_nums)
            for i, text in enumerate(page_texts):
                if i >= len(sorted_pages):
                    break
                original_page_num = sorted_pages[i]
                has_text = len(text.strip()) > 10

                yield PageResult(
                    page_num=original_page_num,
                    text=text,
                    confidence=0.95 if has_text else 0.0,
                    quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                    extractor=self.name,
                )
        finally:
            tmp_path.unlink(missing_ok=True)
