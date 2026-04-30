"""Ollama provider — local LLM models for vision extraction."""

from __future__ import annotations

import base64
import os

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo
from pdfmux.retry import with_retry


class OllamaProvider(LLMProvider):
    name = "ollama"
    default_model = ""  # must be set via PDFMUX_LLM_MODEL

    def sdk_installed(self) -> bool:
        try:
            import ollama  # noqa: F401

            return True
        except ImportError:
            return False

    def has_credentials(self) -> bool:
        return bool(self._get_model())

    def available(self) -> bool:
        return self.sdk_installed() and self.has_credentials()

    def _get_model(self) -> str:
        return os.environ.get("PDFMUX_LLM_MODEL", "")

    def supported_models(self) -> list[ModelInfo]:
        model = self._get_model()
        if model:
            return [ModelInfo(id=model, capabilities=("ocr",))]
        return []

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        return CostEstimate()  # local = free

    @with_retry(max_attempts=3, backoff_base=2.0)
    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        import ollama

        use_model = model or self._get_model()
        if not use_model:
            raise ValueError(
                "Ollama requires PDFMUX_LLM_MODEL to be set (e.g. 'llava', 'bakllava')"
            )

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        client = ollama.Client(host=base_url)
        image_b64 = base64.b64encode(image_bytes).decode()

        response = client.chat(
            model=use_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
        )
        return response.message.content or ""
