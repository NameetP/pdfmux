"""Marker extractor — fast neural PDF extraction.

Marker (https://github.com/datalab-to/marker) is a 33K-star neural
PDF→Markdown converter that excels on academic papers and complex
layouts. Local-only, GPU-accelerated when available.

Speed: ~0.3s/page on GPU, ~2s/page on CPU.
Install: pip install pdfmux[marker]
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF — used for page count fallback

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


# Module-level singletons — Marker model dict is heavy (~1GB),
# load it once and reuse across calls.
_models_lock = threading.Lock()
_models_instance = None
_converter_lock = threading.Lock()
_converter_instance = None


def _check_marker() -> bool:
    """Check if marker-pdf is installed."""
    try:
        import marker.converters.pdf  # noqa: F401
        import marker.models  # noqa: F401

        return True
    except ImportError:
        return False


def _get_models():
    """Lazy-load and cache Marker's model dict."""
    global _models_instance
    if _models_instance is None:
        with _models_lock:
            if _models_instance is None:
                from marker.models import create_model_dict

                _models_instance = create_model_dict()
    return _models_instance


def _get_converter():
    """Lazy-load and cache the Marker PdfConverter.

    The converter holds references to all loaded transformer models;
    instantiate it once per process to avoid reloading ~1GB of weights.
    """
    global _converter_instance
    if _converter_instance is None:
        with _converter_lock:
            if _converter_instance is None:
                from marker.converters.pdf import PdfConverter

                _converter_instance = PdfConverter(artifact_dict=_get_models())
    return _converter_instance


def _extract_markdown(rendered) -> str:
    """Pull markdown text out of Marker's rendered output.

    Different Marker versions return slightly different shapes — try
    several known accessors before giving up.
    """
    # Newer versions: `rendered.markdown`
    text = getattr(rendered, "markdown", None)
    if isinstance(text, str):
        return text

    # Tuple form: (markdown, metadata, images)
    if isinstance(rendered, tuple) and rendered:
        first = rendered[0]
        if isinstance(first, str):
            return first

    # `text_from_rendered` helper
    try:
        from marker.output import text_from_rendered

        result = text_from_rendered(rendered)
        if isinstance(result, str):
            return result
        if isinstance(result, tuple) and result and isinstance(result[0], str):
            return result[0]
    except Exception:
        pass

    return str(rendered) if rendered is not None else ""


def _split_pages(markdown: str, expected_pages: int) -> list[str]:
    """Split Marker's whole-document markdown into per-page chunks.

    Marker writes a `\\n\\n---\\n\\n` separator between pages by default.
    If that separator is absent, return everything as a single page.
    """
    if "\n\n---\n\n" in markdown:
        chunks = markdown.split("\n\n---\n\n")
    elif "\n---\n" in markdown and expected_pages > 1:
        chunks = markdown.split("\n---\n")
    else:
        chunks = [markdown]

    return [c.strip() for c in chunks if c.strip()]


@register(name="marker", priority=35)
class MarkerExtractor:
    """Extract text from PDFs using Marker.

    Marker runs whole-document conversion (not per-page); we split
    the resulting markdown on Marker's page separator to produce one
    PageResult per source page.
    """

    @property
    def name(self) -> str:
        return "marker"

    def available(self) -> bool:
        return _check_marker()

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield PageResults from a Marker conversion."""
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Marker is not installed. Install with: pip install pdfmux[marker]"
            )

        file_path = Path(file_path)

        # Determine expected page count for splitting
        try:
            doc = fitz.open(str(file_path))
            expected_pages = len(doc)
            doc.close()
        except Exception:
            expected_pages = 1

        try:
            converter = _get_converter()
            rendered = converter(str(file_path))
        except Exception as e:
            from pdfmux.errors import ExtractionError

            logger.warning("Marker conversion failed on %s: %s", file_path.name, e)
            raise ExtractionError(f"Marker conversion failed: {e}") from e

        markdown = _extract_markdown(rendered)
        page_texts = _split_pages(markdown, expected_pages)

        # If split count doesn't match doc length, fall back to one big page
        if len(page_texts) != expected_pages and expected_pages > 0:
            logger.debug(
                "Marker page split mismatch: got %d chunks for %d pages — emitting one page",
                len(page_texts),
                expected_pages,
            )
            page_texts = ["\n\n".join(page_texts)] if page_texts else [""]

        for i, text in enumerate(page_texts):
            if pages is not None and i not in pages:
                continue

            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=i,
                text=text,
                confidence=0.93 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
            )
