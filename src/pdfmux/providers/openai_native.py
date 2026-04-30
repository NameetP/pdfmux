"""OpenAI native provider — GPT-4o for vision extraction."""

from __future__ import annotations

import base64
import os

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo
from pdfmux.retry import with_retry


class OpenAINativeProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o"

    def sdk_installed(self) -> bool:
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False

    def has_credentials(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def available(self) -> bool:
        return self.sdk_installed() and self.has_credentials()

    def supported_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="gpt-4o",
                capabilities=("ocr", "tables", "structured", "charts"),
                input_cost_per_mtok=2.50,
                output_cost_per_mtok=10.0,
            ),
            ModelInfo(
                id="gpt-4o-mini",
                capabilities=("ocr", "tables", "structured"),
                input_cost_per_mtok=0.15,
                output_cost_per_mtok=0.60,
            ),
        ]

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        # GPT-4o: ~765 tokens per image at 200 DPI (high detail)
        input_tokens = 765 + prompt_tokens
        output_tokens = 500
        cost = (input_tokens * 2.50 + output_tokens * 10.0) / 1_000_000
        return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @with_retry(max_attempts=3, backoff_base=2.0)
    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        import openai

        client = openai.OpenAI()
        image_b64 = base64.b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
