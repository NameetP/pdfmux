"""OpenDataLoader extractor — high-accuracy PDF parsing with reading order.

Uses OpenDataLoader-PDF (by Hancom) for structured extraction with
bounding boxes, reading order, and table detection. Best-in-class
accuracy on complex layouts.

Speed: ~0.05s/page (local), ~0.43s/page (hybrid).
Install: pip install pdfmux[opendataloader]

Requires Java 11+ runtime for the core engine.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pdfmux.extractors import register
from pdfmux.types import ExtractedTable, PageQuality, PageResult

logger = logging.getLogger(__name__)


def _check_opendataloader() -> bool:
    """Check if opendataloader-pdf is installed."""
    try:
        import opendataloader_pdf  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_elements_to_markdown(elements: list[dict]) -> str:
    """Convert OpenDataLoader JSON elements to clean Markdown.

    Elements have type (heading, paragraph, table, list, image, formula),
    text content, and optional bounding boxes.
    """
    lines: list[str] = []

    for el in elements:
        el_type = el.get("type", "paragraph")
        text = el.get("text", "").strip()

        if not text:
            continue

        if el_type == "heading":
            level = el.get("level", 1)
            prefix = "#" * min(level, 6)
            lines.append(f"{prefix} {text}")
        elif el_type == "table":
            # Tables come as structured data or markdown
            lines.append(text)
        elif el_type == "list":
            lines.append(text)
        elif el_type == "formula":
            lines.append(f"$${text}$$")
        else:
            lines.append(text)

        lines.append("")  # blank line between elements

    return "\n".join(lines).strip()


def _extract_tables_from_elements(
    elements: list[dict],
    page_num: int,
) -> list[ExtractedTable]:
    """Extract structured table data from OpenDataLoader elements."""
    tables: list[ExtractedTable] = []

    for el in elements:
        if el.get("type") != "table":
            continue

        cells = el.get("cells", [])
        if not cells:
            continue

        # OpenDataLoader provides cells with row/col indices
        # Group by row
        rows_dict: dict[int, list[tuple[int, str]]] = {}
        for cell in cells:
            row_idx = cell.get("row", 0)
            col_idx = cell.get("col", 0)
            cell_text = cell.get("text", "").strip()
            if row_idx not in rows_dict:
                rows_dict[row_idx] = []
            rows_dict[row_idx].append((col_idx, cell_text))

        if not rows_dict:
            continue

        # Sort rows and columns
        sorted_row_keys = sorted(rows_dict.keys())
        all_rows = []
        for rk in sorted_row_keys:
            row_cells = sorted(rows_dict[rk], key=lambda x: x[0])
            all_rows.append(tuple(c[1] for c in row_cells))

        if len(all_rows) < 2:
            continue

        headers = all_rows[0]
        data_rows = tuple(all_rows[1:])

        bbox = None
        if "bbox" in el:
            bbox = tuple(el["bbox"])

        tables.append(
            ExtractedTable(
                page_num=page_num,
                headers=headers,
                rows=data_rows,
                bbox=bbox,
            )
        )

    return tables


@register(name="opendataloader", priority=15)
class OpenDataLoaderExtractor:
    """Extract text from PDFs using OpenDataLoader-PDF.

    Provides high-accuracy extraction with reading order preservation,
    bounding boxes, and structured table detection. Uses the local
    engine by default (fast, no server required). Hybrid mode available
    for complex tables and OCR.
    """

    @property
    def name(self) -> str:
        return "opendataloader"

    def available(self) -> bool:
        return _check_opendataloader()

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
        *,
        mode: str = "local",
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page via OpenDataLoader.

        Args:
            file_path: Path to the PDF.
            pages: Optional 0-indexed page list. None = all pages.
            mode: "local" (fast, no server) or "hybrid" (OCR + tables).
        """
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "OpenDataLoader-PDF is not installed. "
                "Install with: pip install pdfmux[opendataloader]"
            )

        import opendataloader_pdf

        file_path = Path(file_path)

        try:
            result = opendataloader_pdf.convert(
                str(file_path),
                format="json",
            )
        except Exception as e:
            logger.warning("OpenDataLoader failed on %s: %s", file_path.name, e)
            return

        # Parse the result — OpenDataLoader returns structured JSON
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                # If it returned markdown directly, yield as single page
                yield PageResult(
                    page_num=0,
                    text=result,
                    confidence=0.92,
                    quality=PageQuality.GOOD if len(result.strip()) > 10 else PageQuality.EMPTY,
                    extractor=self.name,
                )
                return

        # Handle page-level results
        doc_pages = result.get("pages", [result]) if isinstance(result, dict) else [result]

        for i, page_data in enumerate(doc_pages):
            if pages is not None and i not in pages:
                continue

            # Extract text from elements
            if isinstance(page_data, dict):
                elements = page_data.get("elements", [])
                text = _parse_elements_to_markdown(elements)
                page_tables = _extract_tables_from_elements(elements, i)

                # Fallback: if no elements, check for direct markdown
                if not text:
                    text = page_data.get("markdown", page_data.get("text", ""))
            else:
                text = str(page_data)
                page_tables = []

            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=i,
                text=text,
                confidence=0.92 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                tables=tuple(page_tables) if page_tables else (),
            )
