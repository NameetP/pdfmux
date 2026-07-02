"""Claude provider — Anthropic's Claude for vision extraction."""

from __future__ import annotations

import base64
import os

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo
from pdfmux.retry import with_retry


class ClaudeProvider(LLMProvider):
    name = "claude"
    default_model = "claude-sonnet-5"

    def sdk_installed(self) -> bool:
        try:
            import anthropic  # noqa: F401

            return True
        except ImportError:
            return False

    def has_credentials(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def available(self) -> bool:
        return self.sdk_installed() and self.has_credentials()

    def supported_models(self) -> list[ModelInfo]:
        # Pricing per MTok, verified 2026-07-02 against the claude-api model table.
        # Sonnet 5 carries an introductory rate ($2/$10) through 2026-08-31; the
        # standard $3/$15 is used here so cost estimates don't understate spend
        # once the intro window closes.
        return [
            ModelInfo(
                id="claude-sonnet-5",
                capabilities=("ocr", "tables", "structured", "handwriting", "charts"),
                input_cost_per_mtok=3.0,
                output_cost_per_mtok=15.0,
            ),
            ModelInfo(
                id="claude-haiku-4-5",
                capabilities=("ocr", "tables", "structured"),
                input_cost_per_mtok=1.0,
                output_cost_per_mtok=5.0,
            ),
        ]

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        # Claude vision: ~1600 tokens per image at 200 DPI
        input_tokens = 1600 + prompt_tokens
        output_tokens = 500
        cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
        return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @with_retry(max_attempts=3, backoff_base=2.0)
    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        import anthropic

        client = anthropic.Anthropic()
        image_b64 = base64.b64encode(image_bytes).decode()

        response = client.messages.create(
            model=model or self.default_model,
            max_tokens=4096,
            # Sonnet 5 runs adaptive thinking by default; disable it for page
            # extraction — thinking adds latency + cost and eats into max_tokens,
            # and OCR/transcription needs no chain-of-thought. Accepted on
            # Sonnet 5 / 4.7 / 4.8 (only Fable 5 rejects an explicit disable).
            thinking={"type": "disabled"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text if response.content else ""
