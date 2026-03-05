"""LLM vision extractor — Gemini Flash for the hardest cases.

Premium fallback for handwriting, complex forms, and documents
that defeat rule-based extraction. Uses Gemini 2.5 Flash for best
cost/accuracy ratio (~$0.01-0.05 per document).

Install: pip install pdfmux[llm]
Requires: GEMINI_API_KEY or GOOGLE_API_KEY env variable.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


def _check_genai() -> bool:
    """Check if google-genai is installed."""
    try:
        import google.genai  # noqa: F401

        return True
    except ImportError:
        return False


EXTRACTION_PROMPT = """\
Extract all text from this PDF page image and format as clean Markdown.

Rules:
- Preserve document structure (headings, lists, tables, paragraphs)
- Format tables as Markdown tables with | delimiters
- Preserve bullet points and numbered lists
- Extract ALL visible text including headers, footers, captions
- For handwritten text, do your best to transcribe accurately
- If text is unclear, wrap it in [unclear: best guess]
- Do not add any commentary — only output the extracted content"""


@register(name="llm", priority=50)
class LLMExtractor:
    """Extract text from PDFs using Gemini Flash vision API."""

    @property
    def name(self) -> str:
        return "gemini-flash"

    def available(self) -> bool:
        if not _check_genai():
            return False
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return api_key is not None

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page via LLM vision."""
        if not _check_genai():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Google GenAI is not installed. Install with: pip install pdfmux[llm]"
            )

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            from pdfmux.errors import ExtractionError

            raise ExtractionError("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        from google import genai

        client = genai.Client(api_key=api_key)

        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))

        for page_num in page_range:
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pix.save(tmp_path)
                with open(tmp_path, "rb") as f:
                    image_bytes = f.read()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            image_b64 = base64.b64encode(image_bytes).decode()

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "parts": [
                            {"text": EXTRACTION_PROMPT},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64,
                                }
                            },
                        ]
                    }
                ],
            )

            text = response.text if response.text else ""
            has_text = len(text.strip()) > 10

            yield PageResult(
                page_num=page_num,
                text=text.strip(),
                confidence=0.90 if has_text else 0.0,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                image_count=len(page.get_images(full=True)),
                ocr_applied=True,
            )

        doc.close()
