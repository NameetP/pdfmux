"""Surya OCR extractor — heavy-weight OCR for scanned PDFs.

~5GB install, PyTorch, GPU recommended. Legacy — RapidOCR preferred.

Install: pip install pdfmux[ocr-heavy]
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


def _check_surya() -> bool:
    """Check if surya-ocr is installed."""
    try:
        import surya  # noqa: F401

        return True
    except ImportError:
        return False


@register(name="surya", priority=30)
class OCRExtractor:
    """Extract text from scanned PDFs using Surya OCR."""

    @property
    def name(self) -> str:
        return "surya"

    def available(self) -> bool:
        return _check_surya()

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page via Surya OCR."""
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Surya OCR is not installed. Install with: pip install pdfmux[ocr-heavy]"
            )

        from PIL import Image
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))

        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()

        for page_num in page_range:
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pix.save(tmp.name)
                image = Image.open(tmp.name)

            predictions = rec_predictor([image], [det_predictor([image])[0].bboxes])

            text = ""
            if predictions and predictions[0].text_lines:
                lines = [line.text for line in predictions[0].text_lines]
                text = "\n".join(lines)

            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=page_num,
                text=text,
                confidence=0.80 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                image_count=len(page.get_images(full=True)),
                ocr_applied=True,
            )

        doc.close()
