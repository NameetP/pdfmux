"""RapidOCR extractor — lightweight OCR for image-heavy pages.

~200MB install (ONNX runtime + PaddleOCR v4 models).
Default pdfmux[ocr] backend — CPU-only, Apache 2.0.

Install: pip install pdfmux[ocr]
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.extractors import register
from pdfmux.ocr_review import OCRLine
from pdfmux.render import render_pixmap
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)

# Thread-local context populated by `_ocr_page` so the RapidOCR upstream
# logger filter can attach file+page provenance to every translated warning.
_ocr_context = threading.local()


class _PdfmuxRapidOCRFilter(logging.Filter):
    """Translate upstream `RapidOCR` warnings into pdfmux-namespaced messages.

    Background: RapidOCR emits messages like
    ``[RapidOCR] main.py:132: The text detection result is empty`` straight to
    stderr with no file context. From a user's POV that's indistinguishable
    from "OCR gave up on a page that needed re-extraction." This filter:

      1. Reads the current file/page from a thread-local set by ``_ocr_page``.
      2. Re-emits the message via ``pdfmux.extractors.rapid_ocr`` at INFO with
         the document name and 0-indexed page number.
      3. Returns False to drop the original noisy record.
    """

    _NOISY_PHRASES = ("text detection result is empty",)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for phrase in self._NOISY_PHRASES:
            if phrase in msg:
                ctx_file = getattr(_ocr_context, "file_name", "<unknown>")
                ctx_page = getattr(_ocr_context, "page_num", -1)
                logger.info(
                    "OCR found no text on page %s of %s (RapidOCR: %s)",
                    ctx_page,
                    ctx_file,
                    msg,
                )
                return False  # suppress the original noisy record
        # Anything else from RapidOCR — let it through unchanged.
        return True


def _check_rapidocr() -> bool:
    """Check if rapidocr + onnxruntime are installed."""
    try:
        import rapidocr  # noqa: F401

        return True
    except ImportError:
        return False


@register(name="rapidocr", priority=20)
class RapidOCRExtractor:
    """Extract text from PDF pages using RapidOCR.

    RapidOCR uses PaddleOCR v4 models via ONNX runtime.
    CPU-only, no GPU required, ~200MB footprint.
    Engine is created once and cached for the instance lifetime.
    """

    def __init__(self) -> None:
        self._engine = None
        if _check_rapidocr():
            self._init_engine()

    def _init_engine(self) -> None:
        """Lazy-init the RapidOCR engine (loads 3 ONNX models, ~1s)."""
        from rapidocr import RapidOCR

        # Translate noisy upstream RapidOCR warnings into pdfmux-namespaced
        # messages with file+page context (set by _ocr_page just before
        # invoking the engine). The upstream record is suppressed.
        rapid_logger = logging.getLogger("RapidOCR")
        rapid_logger.setLevel(logging.WARNING)
        # Don't double-install if RapidOCRExtractor is constructed twice.
        if not any(isinstance(f, _PdfmuxRapidOCRFilter) for f in rapid_logger.filters):
            rapid_logger.addFilter(_PdfmuxRapidOCRFilter())

        self._engine = RapidOCR()

    @property
    def name(self) -> str:
        return "rapidocr"

    def available(self) -> bool:
        return _check_rapidocr()

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page via OCR.

        Opens the PDF once, renders each page at 200 DPI,
        runs OCR, yields result.
        """
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "RapidOCR is not installed. Install with: pip install pdfmux[ocr]"
            )

        if self._engine is None:
            self._init_engine()

        file_path = Path(file_path)
        try:
            from pdfmux.pdf_cache import get_doc

            doc = get_doc(file_path)
        except ImportError:
            doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))

        from pdfmux.ocr_review import review_page

        for page_num in page_range:
            lines = self._ocr_page_lines(doc, page_num)
            text = "\n".join(line.text for line in lines)
            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=page_num,
                text=text,
                confidence=0.85 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                image_count=len(doc[page_num].get_images(full=True)),
                ocr_applied=True,
                review_flags=review_page(page_num, lines),
                ocr_lines=tuple(lines),
            )

    def extract_page(self, file_path: str | Path, page_num: int) -> str:
        """Extract text from a single page — used by multi-pass merge."""
        if self._engine is None:
            self._init_engine()

        file_path = Path(file_path)
        try:
            from pdfmux.pdf_cache import get_doc

            doc = get_doc(file_path)
        except ImportError:
            doc = fitz.open(str(file_path))
        text = self._ocr_page(doc, page_num)
        return text

    def extract_page_lines(self, file_path: str | Path, page_num: int) -> list[OCRLine]:
        """OCR one page and keep each line's confidence score."""
        if self._engine is None:
            self._init_engine()
        try:
            from pdfmux.pdf_cache import get_doc

            doc = get_doc(Path(file_path))
        except ImportError:
            doc = fitz.open(str(file_path))
        return self._ocr_page_lines(doc, page_num)

    def _ocr_page(self, doc: fitz.Document, page_num: int) -> str:
        """Run OCR on a single page; text only."""
        return "\n".join(line.text for line in self._ocr_page_lines(doc, page_num))

    def _ocr_page_lines(self, doc: fitz.Document, page_num: int) -> list[OCRLine]:
        """Run OCR on a single page at 200 DPI; lines in reading order with score and bbox.

        ``bbox`` is in PDF points relative to the top-left of the visible page, so a
        flag can be drawn on the page image even if its line never reaches the text.
        """
        page = doc[page_num]
        pix = render_pixmap(page, 200)
        img_bytes = pix.tobytes("png")

        # Stash file+page provenance for _PdfmuxRapidOCRFilter so any upstream
        # warnings get re-emitted with context instead of bare module names.
        try:
            _ocr_context.file_name = Path(doc.name).name if doc.name else "<unknown>"
        except Exception:
            _ocr_context.file_name = "<unknown>"
        _ocr_context.page_num = page_num

        result = self._engine(img_bytes)

        if not result.txts:
            return []
        scores = result.scores or ()
        boxes = result.boxes if result.boxes is not None else ()
        # Pixel -> point scale for this render (the pixel cap may have lowered the DPI).
        sx = page.rect.width / pix.width if pix.width else 0.0
        sy = page.rect.height / pix.height if pix.height else 0.0
        lines: list[OCRLine] = []
        for i, t in enumerate(result.txts):
            bbox = None
            if i < len(boxes) and sx and sy:
                try:
                    xs = [float(pt[0]) for pt in boxes[i]]
                    ys = [float(pt[1]) for pt in boxes[i]]
                    bbox = (min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy)
                except (TypeError, ValueError, IndexError):
                    bbox = None
            score = float(scores[i]) if i < len(scores) else None
            lines.append(OCRLine(i, str(t), score, bbox))
        return lines
