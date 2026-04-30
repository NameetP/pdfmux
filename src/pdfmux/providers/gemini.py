"""Gemini provider — Google's Gemini Flash for vision extraction."""

from __future__ import annotations

import base64
import os

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo
from pdfmux.retry import with_retry


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = "gemini-2.5-flash"

    def sdk_installed(self) -> bool:
        try:
            import google.genai  # noqa: F401

            return True
        except ImportError:
            return False

    def has_credentials(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def available(self) -> bool:
        return self.sdk_installed() and self.has_credentials()

    def supported_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="gemini-2.5-flash",
                capabilities=("ocr", "tables", "structured", "charts"),
                input_cost_per_mtok=0.15,
                output_cost_per_mtok=0.60,
            ),
            ModelInfo(
                id="gemini-2.5-pro",
                capabilities=("ocr", "tables", "structured", "charts", "handwriting"),
                input_cost_per_mtok=1.25,
                output_cost_per_mtok=10.0,
            ),
        ]

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        # ~260 tokens per image for Gemini Flash at 200 DPI
        input_tokens = 260 + prompt_tokens
        output_tokens = 500  # average extraction output
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @with_retry(max_attempts=3, backoff_base=2.0)
    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        image_b64 = base64.b64encode(image_bytes).decode()

        response = client.models.generate_content(
            model=model or self.default_model,
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ]
                }
            ],
        )
        return response.text or ""
