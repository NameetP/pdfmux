"""Fast extractor — PyMuPDF/pymupdf4llm for digital PDFs.

Primary extractor, handles ~90% of PDFs.
Speed: ~0.01s/page. Cost: $0. Accuracy: 98%+ on digital PDFs.

Streams one PageResult per page via pymupdf4llm page_chunks=True.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import fitz
import pymupdf4llm

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


@register(name="fast", priority=10)
class FastExtractor:
    """Extract text from digital PDFs using pymupdf4llm.

    Yields one PageResult per page. Falls back to raw fitz
    extraction when pymupdf4llm returns empty for a page.
    """

    @property
    def name(self) -> str:
        return "pymupdf4llm"

    def available(self) -> bool:
        return True  # pymupdf + pymupdf4llm are base deps

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page.

        Uses pymupdf4llm with page_chunks=True for per-page data,
        including image counts for downstream audit.
        """
        file_path = Path(file_path)

        chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

        for i, chunk in enumerate(chunks):
            if pages is not None and i not in pages:
                continue

            text = chunk.get("text", "")
            image_count = len(chunk.get("images", []))

            # Fallback: if pymupdf4llm returned near-empty for this page,
            # try raw fitz extraction
            if len(text.strip()) < 50:
                raw = self._extract_raw_page(file_path, i)
                if len(raw.strip()) > len(text.strip()):
                    text = raw

            yield PageResult(
                page_num=i,
                text=text,
                confidence=1.0,  # fast extract starts at full confidence
                quality=PageQuality.GOOD,  # audit will reassess
                extractor=self.name,
                image_count=image_count,
            )

    @staticmethod
    def _extract_raw_page(file_path: Path, page_num: int) -> str:
        """Fallback: extract plain text via fitz for a single page."""
        doc = fitz.open(str(file_path))
        if page_num >= len(doc):
            doc.close()
            return ""
        page = doc[page_num]
        text = page.get_text("text").strip()
        doc.close()
        return text

    def extract_text(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> str:
        """Convenience: return full text as a single string."""
        parts = [p.text for p in self.extract(file_path, pages) if p.text.strip()]
        return "\n\n---\n\n".join(parts)
