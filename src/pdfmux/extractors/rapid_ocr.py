"""RapidOCR extractor — lightweight OCR for image-heavy pages.

~200MB install (ONNX runtime + PaddleOCR v4 models).
Default pdfmux[ocr] backend — CPU-only, Apache 2.0.

Install: pip install pdfmux[ocr]
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


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

        # Suppress noisy RapidOCR INFO logs
        rapid_logger = logging.getLogger("RapidOCR")
        rapid_logger.setLevel(logging.WARNING)
        for handler in rapid_logger.handlers:
            handler.setLevel(logging.WARNING)

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

        for page_num in page_range:
            text = self._ocr_page(doc, page_num)
            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=page_num,
                text=text,
                confidence=0.85 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                image_count=len(doc[page_num].get_images(full=True)),
                ocr_applied=True,
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

    def _ocr_page(self, doc: fitz.Document, page_num: int) -> str:
        """Run OCR on a single page at 200 DPI."""
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        result = self._engine(img_bytes)

        if result.txts:
            return "\n".join(result.txts)
        return ""
