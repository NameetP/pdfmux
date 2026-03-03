"""LLM vision extractor — Gemini Flash for the hardest cases.

This is the premium fallback for handwriting, complex forms, and documents
that defeat rule-based extraction. Uses Gemini 2.5 Flash for best
cost/accuracy ratio (~$0.01-0.05 per document).
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

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


class LLMExtractor:
    """Extract text from PDFs using LLM vision API (Gemini Flash)."""

    @property
    def name(self) -> str:
        return "gemini-flash (LLM)"

    def extract(self, file_path: str | Path, pages: list[int] | None = None) -> str:
        """Extract text from a PDF using Gemini Flash vision.

        Pipeline:
        1. Render PDF pages to images via PyMuPDF
        2. Send each image to Gemini Flash with extraction prompt
        3. Concatenate results into Markdown

        Args:
            file_path: Path to the PDF file.
            pages: Optional list of 0-indexed page numbers to extract.

        Returns:
            Markdown text extracted via LLM vision.
        """
        if not _check_genai():
            raise ImportError(
                "Google GenAI is not installed. Install with: pip install pdfmux[llm]"
            )

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY env variable."
            )

        from google import genai

        client = genai.Client(api_key=api_key)

        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))
        all_text: list[str] = []

        for page_num in page_range:
            page = doc[page_num]
            # Render at 200 DPI — good balance of quality vs token cost
            pix = page.get_pixmap(dpi=200)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pix.save(tmp.name)
                with open(tmp.name, "rb") as f:
                    image_bytes = f.read()

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

            page_text = response.text if response.text else ""

            if page_text.strip():
                all_text.append(page_text.strip())
            else:
                all_text.append(f"*(Page {page_num + 1}: no text extracted)*")

        doc.close()
        return "\n\n---\n\n".join(all_text)
