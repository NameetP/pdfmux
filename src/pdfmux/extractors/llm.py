"""LLM vision extractor — BYOK multi-provider support.

Premium fallback for handwriting, complex forms, and documents
that defeat rule-based extraction. Supports Gemini, Claude,
GPT-4o, and Ollama — auto-detects from available API keys.

Install providers:
  pip install pdfmux[llm]           # Gemini (default)
  pip install pdfmux[llm-claude]    # Claude
  pip install pdfmux[llm-openai]    # GPT-4o
  pip install pdfmux[llm-ollama]    # Ollama (local)
  pip install pdfmux[llm-all]       # All providers

Env vars:
  PDFMUX_LLM_PROVIDER  — Force a provider (gemini/claude/openai/ollama)
  PDFMUX_LLM_MODEL     — Override default model name
  GEMINI_API_KEY        — Gemini auth
  ANTHROPIC_API_KEY     — Claude auth
  OPENAI_API_KEY        — GPT-4o auth
  OLLAMA_BASE_URL       — Ollama endpoint (default localhost:11434)
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.extractors import register
from pdfmux.render import render_pixmap
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


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
    """Extract text from PDFs using LLM vision APIs.

    Auto-detects the best available provider from installed SDKs
    and configured API keys. Can be overridden with PDFMUX_LLM_PROVIDER.
    """

    def __init__(self) -> None:
        self._provider = None
        self._provider_name: str | None = None

    def _resolve(self):
        """Lazily resolve the provider on first use."""
        if self._provider is not None:
            return
        from pdfmux.providers import resolve_provider

        provider_override = os.environ.get("PDFMUX_LLM_PROVIDER")
        model_override = os.environ.get("PDFMUX_LLM_MODEL")
        self._provider = resolve_provider(provider_override, model_override)
        self._provider_name = self._provider.name

    @property
    def name(self) -> str:
        try:
            self._resolve()
            model = os.environ.get("PDFMUX_LLM_MODEL") or self._provider.default_model
            return f"{self._provider_name}-{model.split('/')[-1].split('-')[0]}"
        except (ValueError, Exception):
            return "llm"

    def available(self) -> bool:
        """True if at least one LLM provider is configured."""
        try:
            from pdfmux.providers import available_providers

            return len(available_providers()) > 0
        except Exception:
            return False

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page via LLM vision."""
        self._resolve()

        model_override = os.environ.get("PDFMUX_LLM_MODEL")

        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        page_range = pages if pages is not None else list(range(len(doc)))

        for page_num in page_range:
            page = doc[page_num]
            pix = render_pixmap(page, 200)

            # Render page to PNG bytes
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

            # Send to LLM provider
            text = self._provider.extract_page(
                image_bytes=image_bytes,
                prompt=EXTRACTION_PROMPT,
                model=model_override,
            )

            has_text = len(text.strip()) > 10

            # Dynamic quality scoring instead of hardcoded 0.90
            if has_text:
                try:
                    from pdfmux.router.scorer import score_llm_output

                    confidence = score_llm_output(text.strip())
                except Exception:
                    confidence = 0.85  # safe fallback
            else:
                confidence = 0.0

            yield PageResult(
                page_num=page_num,
                text=text.strip(),
                confidence=confidence,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                image_count=len(page.get_images(full=True)),
                ocr_applied=True,
            )

        doc.close()
