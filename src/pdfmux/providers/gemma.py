"""Gemma provider — Google's Gemma vision models via OpenAI-compatible API.

Gemma is Google's open-weight model family. This provider serves the
vision-capable instruct variants of **Gemma 3** (`gemma-3-27b-it`,
`gemma-3-12b-it`), which support Arabic among many other languages, with
native bidirectional script handling — making them the preferred backend
for Arabic-heavy documents in pdfmux.

Gemma 4, not Gemma 3
--------------------
This provider does NOT serve Gemma 4, despite what several docs claimed
until 2026-07-27. Gemma 4 shipped March/April 2026 and Google does expose
two of its sizes on the Gemini API — `gemma-4-31b-it` and
`gemma-4-26b-a4b-it` (https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api).
Note that Gemma 4 has no 27B size at all; 27B is a Gemma 3 size, which is
how the mislabelling was caught.

Adopting Gemma 4 here is a small diff but is NOT a free rename, because
three things are unverified and each can silently degrade extraction:

  1. Whether the OpenAI-compat shim below (`/v1beta/openai/`) serves the
     Gemma 4 IDs. Google documents them on the native `generateContent`
     endpoint; it does not document the OpenAI-compat path for them.
  2. Pricing. `_INPUT_COST_PER_MTOK` / `_OUTPUT_COST_PER_MTOK` below are
     Gemma 3 rates and feed the budget cap, so a wrong value silently
     over- or under-bills every routed page.
  3. Vision behaviour and `max_input_tokens` (Gemma 4's larger sizes
     document a 256K context vs the 128K claimed below for Gemma 3).

Verify all three against a live key before changing `default_model`. Until
then this provider says Gemma 3, because that is what it sends.

Reuses the ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) credential since
Gemma is served from the same Google generative-language endpoint:

    https://generativelanguage.googleapis.com/v1beta/openai/

That endpoint speaks the OpenAI ``/chat/completions`` protocol, so we
build on the ``openai`` Python SDK rather than ``google-genai``.

Cost estimates (rough, verify before billing-sensitive use):
  - input:  $0.075 per 1M tokens
  - output: $0.30 per 1M tokens
These are pulled from Google's public pricing for Gemma served via the
generative-language API. They are intentionally lower than Gemini Flash
because Gemma is the smaller open-weight model.
"""

from __future__ import annotations

import base64
import os

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo

# Google's OpenAI-compatible endpoint serves both Gemini and Gemma.
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Cost per 1M tokens. Verify at:
#   https://ai.google.dev/pricing
_INPUT_COST_PER_MTOK = 0.075
_OUTPUT_COST_PER_MTOK = 0.30


class GemmaProvider(LLMProvider):
    """Google Gemma via the OpenAI-compatible generativelanguage endpoint.

    Strong Arabic + multilingual OCR. Falls back to ``GOOGLE_API_KEY`` if
    ``GEMINI_API_KEY`` is unset since they're interchangeable on this
    endpoint.
    """

    name = "gemma"
    default_model = "gemma-3-27b-it"

    def sdk_installed(self) -> bool:
        try:
            import openai  # noqa: F401

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
                id="gemma-3-27b-it",
                supports_vision=True,
                capabilities=("ocr", "tables", "arabic", "multilingual"),
                input_cost_per_mtok=_INPUT_COST_PER_MTOK,
                output_cost_per_mtok=_OUTPUT_COST_PER_MTOK,
                max_input_tokens=128_000,
            ),
            ModelInfo(
                id="gemma-3-12b-it",
                supports_vision=True,
                capabilities=("ocr", "arabic", "multilingual"),
                input_cost_per_mtok=_INPUT_COST_PER_MTOK,
                output_cost_per_mtok=_OUTPUT_COST_PER_MTOK,
                max_input_tokens=128_000,
            ),
        ]

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        # Gemma vision tokenization is comparable to Gemini Flash:
        # ~260 tokens for a 200-DPI page image.
        input_tokens = 260 + prompt_tokens
        output_tokens = 500  # average extraction output
        cost = (
            input_tokens * _INPUT_COST_PER_MTOK + output_tokens * _OUTPUT_COST_PER_MTOK
        ) / 1_000_000
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        import openai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        client = openai.OpenAI(base_url=_BASE_URL, api_key=api_key)
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

    def extract_page_with_cost(
        self, image_bytes: bytes, prompt: str, model: str | None = None
    ) -> tuple[str, CostEstimate]:
        """Extract page and return text plus actual usage when reported."""
        import openai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        client = openai.OpenAI(base_url=_BASE_URL, api_key=api_key)
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

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            cost = (
                input_tokens * _INPUT_COST_PER_MTOK + output_tokens * _OUTPUT_COST_PER_MTOK
            ) / 1_000_000
            return text, CostEstimate(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        return text, self.estimate_cost(len(image_bytes))
