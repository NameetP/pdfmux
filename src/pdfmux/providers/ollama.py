"""Ollama provider — local LLM models for vision extraction."""

from __future__ import annotations

import base64
import ipaddress
import os
from urllib.parse import urlparse

from pdfmux.providers.base import CostEstimate, LLMProvider, ModelInfo
from pdfmux.retry import with_retry

# Cloud metadata service hostnames — never legitimate for an Ollama server.
_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",  # AWS / OpenStack / Azure IMDS
        "metadata.google.internal",  # GCP
        "metadata",  # GCP short form
        "100.100.100.200",  # Alibaba
    }
)


def _validate_ollama_url(url: str) -> str:
    """Validate ``OLLAMA_BASE_URL`` to prevent SSRF (P-N6).

    Allows: http(s), loopback, RFC1918 / link-local / unique-local IPs.
    Rejects: cloud metadata services, public IPs (unless explicitly opted
    in via ``PDFMUX_OLLAMA_ALLOW_PUBLIC=1``).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"OLLAMA_BASE_URL must be http(s), got scheme: {parsed.scheme!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("OLLAMA_BASE_URL is missing a hostname")

    if host in _METADATA_HOSTS:
        raise ValueError(f"OLLAMA_BASE_URL points at a cloud metadata service: {host}")

    if host in ("localhost", "127.0.0.1", "::1"):
        return url

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname (not an IP literal). Require an explicit opt-in for non-IP
        # hosts since we can't validate the destination at config-load time.
        if os.environ.get("PDFMUX_OLLAMA_ALLOW_PUBLIC") == "1":
            return url
        raise ValueError(
            f"OLLAMA_BASE_URL hostname {host!r} is not in a private range; "
            "set PDFMUX_OLLAMA_ALLOW_PUBLIC=1 to allow non-loopback hostnames."
        ) from None

    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return url

    raise ValueError(
        f"OLLAMA_BASE_URL must point at loopback or a private network, got {host}"
    )


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

        base_url = _validate_ollama_url(
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )
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
