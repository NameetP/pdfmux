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

from pdfmux.column_reorder import reorder_text_ab
from pdfmux.extractors import register
from pdfmux.headings import inject_headings
from pdfmux.table_fallback import detect_text_tables
from pdfmux.types import ExtractedTable, PageQuality, PageResult

logger = logging.getLogger(__name__)


def _extract_tables_fast(
    page: fitz.Page,
    page_num: int,
    text: str,
) -> tuple[str, list[ExtractedTable]]:
    """Extract structured tables using PyMuPDF's built-in table finder.

    Uses find_tables() (PyMuPDF 1.23.0+, no ML deps) to detect table
    regions. Returns both enhanced text (markdown tables appended) and
    structured ExtractedTable objects with raw row/column data.

    Returns:
        (enhanced_text, list_of_extracted_tables)
    """
    structured_tables: list[ExtractedTable] = []

    try:
        tables = page.find_tables()
    except (AttributeError, Exception):
        return text, structured_tables

    if not tables.tables:
        # Fallback: try text-based detection for borderless tables
        fallback = detect_text_tables(page, page_num)
        if fallback:
            enhanced = text.rstrip()
            for ft in fallback:
                md_lines = []
                md_lines.append("| " + " | ".join(ft.headers) + " |")
                md_lines.append("| " + " | ".join("---" for _ in ft.headers) + " |")
                for row in ft.rows:
                    md_lines.append("| " + " | ".join(row) + " |")
                enhanced += "\n\n" + "\n".join(md_lines)
            return enhanced, fallback
        return text, structured_tables

    table_markdowns = []
    for table in tables.tables:
        try:
            cells = table.extract()
            if not cells or len(cells) < 2 or len(cells[0]) < 2:
                continue

            # Garbage filter: skip tables with <30% non-empty cells
            # (PyMuPDF 1.27 find_tables() regression produces near-empty tables)
            total_cells = sum(len(row) for row in cells)
            filled_cells = sum(
                1 for row in cells for c in row
                if c and str(c).strip()
            )
            if total_cells > 0 and filled_cells / total_cells < 0.30:
                continue

            headers = tuple(str(c).strip() if c else "" for c in cells[0])
            rows = tuple(
                tuple(str(c).strip() if c else "" for c in row)
                for row in cells[1:]
            )

            # Get bounding box if available
            bbox = None
            if hasattr(table, "bbox"):
                bbox = tuple(table.bbox)

            structured_tables.append(
                ExtractedTable(
                    page_num=page_num,
                    headers=headers,
                    rows=rows,
                    bbox=bbox,
                )
            )

            # Also build markdown for text output
            md_lines = []
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                md_lines.append("| " + " | ".join(row) + " |")
            table_markdowns.append("\n".join(md_lines))

        except Exception:
            continue

    if not table_markdowns:
        return text, structured_tables

    enhanced = text.rstrip()
    for table_md in table_markdowns:
        enhanced += "\n\n" + table_md

    return enhanced, structured_tables


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
        *,
        enhance_tables: bool = False,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page.

        Uses pymupdf4llm with page_chunks=True for per-page data,
        including image counts for downstream audit.

        Args:
            enhance_tables: If True, use PyMuPDF's find_tables() to
                append structured markdown tables to page text.
        """
        file_path = Path(file_path)

        chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

        # Always open doc for heading detection + optional table enhancement
        doc = fitz.open(str(file_path))

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

            # Column-aware reading order (A/B comparison — safe, no-op if uncertain)
            if i < len(doc):
                text = reorder_text_ab(text, doc[i])

            # Heading detection via font-size analysis (after fallback
            # so headings are injected into the fullest text available)
            if i < len(doc):
                text = inject_headings(text, doc[i])

            # Table enhancement (fast mode, no ML deps)
            page_tables: tuple[ExtractedTable, ...] = ()
            if enhance_tables and doc and i < len(doc):
                text, extracted = _extract_tables_fast(doc[i], i, text)
                page_tables = tuple(extracted)

            yield PageResult(
                page_num=i,
                text=text,
                confidence=1.0,  # fast extract starts at full confidence
                quality=PageQuality.GOOD,  # audit will reassess
                extractor=self.name,
                image_count=image_count,
                tables=page_tables,
            )

        doc.close()

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

