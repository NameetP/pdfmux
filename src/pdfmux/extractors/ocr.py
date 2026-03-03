"""OCR extractor — for scanned/image-based PDFs.

Uses Surya OCR (preferred) or falls back to PaddleOCR.
Runs locally, no API costs, no GPU required.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _check_surya() -> bool:
    """Check if surya-ocr is installed."""
    try:
        import surya  # noqa: F401

        return True
    except ImportError:
        return False


class OCRExtractor:
    """Extract text from scanned PDFs using OCR."""

    @property
    def name(self) -> str:
        return "surya (OCR)"

    def extract(self, file_path: str | Path, pages: list[int] | None = None) -> str:
        """Extract text from a scanned PDF using OCR.

        Pipeline:
        1. Render PDF pages to images via PyMuPDF
        2. Run OCR on each image via Surya
        3. Concatenate results into Markdown

        Args:
            file_path: Path to the PDF file.
            pages: Optional list of 0-indexed page numbers to extract.

        Returns:
            Markdown text extracted via OCR.
        """
        if not _check_surya():
            raise ImportError("Surya OCR is not installed. Install with: pip install pdfmux[ocr]")

        from PIL import Image
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))
        all_text: list[str] = []

        # Initialize Surya predictors
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()

        for page_num in page_range:
            page = doc[page_num]
            # Render page to image at 300 DPI for good OCR quality
            pix = page.get_pixmap(dpi=300)

            # Save to temp file and load as PIL Image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pix.save(tmp.name)
                image = Image.open(tmp.name)

            # Run OCR
            predictions = rec_predictor([image], [det_predictor([image])[0].bboxes])

            page_text = ""
            if predictions and predictions[0].text_lines:
                lines = [line.text for line in predictions[0].text_lines]
                page_text = "\n".join(lines)

            if page_text.strip():
                all_text.append(f"## Page {page_num + 1}\n\n{page_text}")
            else:
                all_text.append(f"## Page {page_num + 1}\n\n*(No text detected)*")

        doc.close()
        return "\n\n".join(all_text)
