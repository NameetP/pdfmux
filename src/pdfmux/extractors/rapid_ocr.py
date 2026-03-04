"""RapidOCR extractor — lightweight OCR for image-heavy pages.

~200MB install (ONNX runtime + PaddleOCR v4 models).
Replaces Surya (2-5GB, PyTorch, GPL) as the default pdfmux[ocr] backend.

Install: pip install pdfmux[ocr]
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _check_rapidocr() -> bool:
    """Check if rapidocr + onnxruntime are installed."""
    try:
        import rapidocr  # noqa: F401

        return True
    except ImportError:
        return False


class RapidOCRExtractor:
    """Extract text from PDF pages using RapidOCR.

    RapidOCR uses PaddleOCR v4 models via ONNX runtime.
    CPU-only, no GPU required, ~200MB footprint.
    """

    def __init__(self) -> None:
        if not _check_rapidocr():
            raise ImportError(
                "RapidOCR is not installed. "
                "Install with: pip install pdfmux[ocr]"
            )

        from rapidocr import RapidOCR

        # Suppress noisy RapidOCR INFO logs (model paths, engine info).
        # Must be done AFTER import because rapidocr's import resets the
        # logger level and adds its own StreamHandler.
        rapid_logger = logging.getLogger("RapidOCR")
        rapid_logger.setLevel(logging.WARNING)
        for handler in rapid_logger.handlers:
            handler.setLevel(logging.WARNING)

        self._engine = RapidOCR()

    @property
    def name(self) -> str:
        return "rapidocr (OCR)"

    def extract(self, file_path: str | Path, pages: list[int] | None = None) -> str:
        """Extract text from a PDF using OCR.

        Args:
            file_path: Path to the PDF file.
            pages: Optional list of 0-indexed page numbers. If None, all pages.

        Returns:
            Markdown text with page headers.
        """
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))
        all_text: list[str] = []

        for page_num in page_range:
            page_text = self._ocr_page(doc, page_num)

            if page_text.strip():
                all_text.append(f"## Page {page_num + 1}\n\n{page_text}")
            else:
                all_text.append(f"## Page {page_num + 1}\n\n*(No text detected)*")

        doc.close()
        return "\n\n".join(all_text)

    def extract_page(self, file_path: str | Path, page_num: int) -> str:
        """Extract text from a single PDF page using OCR.

        Used by multi-pass merge to surgically re-extract bad pages.

        Args:
            file_path: Path to the PDF file.
            page_num: 0-indexed page number.

        Returns:
            Plain text extracted from the page.
        """
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))
        text = self._ocr_page(doc, page_num)
        doc.close()
        return text

    def _ocr_page(self, doc: fitz.Document, page_num: int) -> str:
        """Run OCR on a single page.

        Pipeline:
        1. Render page to image at 200 DPI
        2. Get PNG bytes
        3. Run RapidOCR engine
        4. Join detected text lines

        Args:
            doc: Open fitz document.
            page_num: 0-indexed page number.

        Returns:
            Extracted text as a string.
        """
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        result = self._engine(img_bytes)

        if result.txts:
            return "\n".join(result.txts)
        return ""
