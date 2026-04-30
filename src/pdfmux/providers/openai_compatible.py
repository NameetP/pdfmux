"""OpenAI-compatible adapter — supports any API with /chat/completions.

Works with: DeepSeek, Kimi (Moonshot), Together AI, Groq, Fireworks,
Mistral, Perplexity, and any other OpenAI-compatible endpoint.

Users configure via ~/.pdfmux/providers.yaml:

    providers:
      kimi:
        type: openai_compatible
        base_url: https://api.moonshot.cn/v1
        api_key_env: KIMI_API_KEY
        models:
          - id: moonshot-v1-128k
            capabilities: [ocr, tables]
            supports_vision: true
            input_cost_per_mtok: 0.8
            output_cost_per_mtok: 0.8
"""

from __future__ import annotations

import base64
import os

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo
from pdfmux.retry import with_retry


class OpenAICompatibleProvider(LLMProvider):
    """Generic adapter for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        models: list[dict] | None = None,
        default_model: str = "",
    ):
        self.name = name
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._models_config = models or []
        self.default_model = default_model or (
            self._models_config[0]["id"] if self._models_config else ""
        )

    def sdk_installed(self) -> bool:
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False

    def has_credentials(self) -> bool:
        return bool(os.environ.get(self._api_key_env))

    def available(self) -> bool:
        return self.sdk_installed() and self.has_credentials() and bool(self.default_model)

    def supported_models(self) -> list[ModelInfo]:
        result = []
        for m in self._models_config:
            result.append(
                ModelInfo(
                    id=m["id"],
                    supports_vision=m.get("supports_vision", True),
                    capabilities=tuple(m.get("capabilities", ())),
                    input_cost_per_mtok=m.get("input_cost_per_mtok"),
                    output_cost_per_mtok=m.get("output_cost_per_mtok"),
                    max_input_tokens=m.get("max_input_tokens", 128_000),
                )
            )
        return result

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        if not self._models_config:
            return CostEstimate()
        m = self._models_config[0]
        input_cost = m.get("input_cost_per_mtok", 0) or 0
        output_cost = m.get("output_cost_per_mtok", 0) or 0
        input_tokens = 800 + prompt_tokens
        output_tokens = 500
        cost = (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000
        return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @with_retry(max_attempts=3, backoff_base=2.0)
    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        import openai

        api_key = os.environ.get(self._api_key_env, "")
        client = openai.OpenAI(base_url=self._base_url, api_key=api_key)
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
