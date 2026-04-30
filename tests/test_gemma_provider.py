"""Tests for the Gemma 4 provider — Arabic-capable vision OCR backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdfmux.providers.base import LLMProvider
from pdfmux.providers.gemma import GemmaProvider


class TestGemmaProviderBasics:
    def test_subclass_of_llm_provider(self) -> None:
        assert issubclass(GemmaProvider, LLMProvider)

    def test_name_is_gemma(self) -> None:
        assert GemmaProvider().name == "gemma"

    def test_default_model_set(self) -> None:
        assert GemmaProvider().default_model.startswith("gemma-")


class TestGemmaAvailability:
    def test_available_with_sdk_and_gemini_key(self) -> None:
        p = GemmaProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        ):
            assert p.has_credentials() is True
            assert p.available() is True

    def test_available_with_google_api_key_fallback(self) -> None:
        p = GemmaProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True),
        ):
            assert p.has_credentials() is True
            assert p.available() is True

    def test_unavailable_without_key(self) -> None:
        p = GemmaProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert p.has_credentials() is False
            assert p.available() is False

    def test_unavailable_without_sdk(self) -> None:
        p = GemmaProvider()
        with (
            patch.object(p, "sdk_installed", return_value=False),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        ):
            assert p.available() is False


class TestGemmaSupportedModels:
    def test_lists_at_least_one_model(self) -> None:
        models = GemmaProvider().supported_models()
        assert len(models) >= 1

    def test_advertises_arabic_capability(self) -> None:
        # The whole reason this provider exists.
        models = GemmaProvider().supported_models()
        assert any("arabic" in m.capabilities for m in models)

    def test_advertises_multilingual_capability(self) -> None:
        models = GemmaProvider().supported_models()
        assert any("multilingual" in m.capabilities for m in models)

    def test_advertises_ocr_capability(self) -> None:
        models = GemmaProvider().supported_models()
        assert any("ocr" in m.capabilities for m in models)

    def test_models_support_vision(self) -> None:
        models = GemmaProvider().supported_models()
        assert all(m.supports_vision for m in models)

    def test_default_model_in_supported(self) -> None:
        p = GemmaProvider()
        ids = [m.id for m in p.supported_models()]
        assert p.default_model in ids


class TestGemmaCostEstimate:
    def test_estimate_returns_positive_cost(self) -> None:
        est = GemmaProvider().estimate_cost(image_bytes_count=200_000)
        assert est.cost_usd > 0
        assert est.input_tokens > 0
        assert est.output_tokens > 0

    def test_pricing_lower_than_gemini_pro(self) -> None:
        # Gemma is the smaller open-weight model, so per-token cost
        # should be significantly lower than Gemini Pro ($1.25/$10).
        models = GemmaProvider().supported_models()
        for m in models:
            assert m.input_cost_per_mtok is not None
            assert m.input_cost_per_mtok < 1.25
            assert m.output_cost_per_mtok is not None
            assert m.output_cost_per_mtok < 10.0


class TestGemmaExtractPage:
    def test_extract_page_calls_openai_compatible_api(self) -> None:
        """Mock the openai SDK and verify the request is shaped correctly."""
        p = GemmaProvider()

        fake_choice = MagicMock()
        fake_choice.message.content = "بوليصة الشحن"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = MagicMock(
            prompt_tokens=300, completion_tokens=50
        )

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI.return_value = fake_client

        with (
            patch.dict("sys.modules", {"openai": fake_openai_module}),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        ):
            text = p.extract_page(b"fake-image-bytes", "Extract Arabic text")

        assert text == "بوليصة الشحن"
        # Verify base URL is the OpenAI-compatible Google endpoint.
        ((), kwargs) = fake_openai_module.OpenAI.call_args
        assert kwargs["base_url"].startswith(
            "https://generativelanguage.googleapis.com"
        )
        assert kwargs["api_key"] == "test-key"

        # Verify the message contains both text prompt and image.
        ((), call_kwargs) = fake_client.chat.completions.create.call_args
        message = call_kwargs["messages"][0]
        assert message["role"] == "user"
        contents = message["content"]
        assert any(part.get("type") == "text" for part in contents)
        assert any(part.get("type") == "image_url" for part in contents)
        # Image is base64-encoded into a data URL.
        image_part = next(part for part in contents if part["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_extract_page_with_cost_uses_actual_usage(self) -> None:
        p = GemmaProvider()

        fake_choice = MagicMock()
        fake_choice.message.content = "extracted"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = MagicMock(prompt_tokens=400, completion_tokens=80)

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI.return_value = fake_client

        with (
            patch.dict("sys.modules", {"openai": fake_openai_module}),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        ):
            text, cost = p.extract_page_with_cost(
                b"fake-image-bytes", "Extract"
            )

        assert text == "extracted"
        assert cost.input_tokens == 400
        assert cost.output_tokens == 80
        assert cost.cost_usd > 0

    def test_uses_explicit_model_when_provided(self) -> None:
        p = GemmaProvider()

        fake_choice = MagicMock()
        fake_choice.message.content = "ok"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI.return_value = fake_client

        with (
            patch.dict("sys.modules", {"openai": fake_openai_module}),
            patch.dict("os.environ", {"GEMINI_API_KEY": "k"}, clear=True),
        ):
            p.extract_page(b"img", "prompt", model="gemma-3-12b-it")

        ((), call_kwargs) = fake_client.chat.completions.create.call_args
        assert call_kwargs["model"] == "gemma-3-12b-it"


class TestGemmaInDiscovery:
    """Make sure Gemma shows up in the built-in provider list."""

    def test_listed_in_builtins(self) -> None:
        from pdfmux.providers._discovery import _load_builtins

        # Reset cached builtins so the test reflects the current code.
        import pdfmux.providers._discovery as discovery

        discovery._BUILTIN_CLASSES = []

        builtins = _load_builtins()
        names = [cls().name for cls in builtins]
        assert "gemma" in names

    def test_discovered_by_name(self) -> None:
        from pdfmux.providers import discover_all_providers

        providers = discover_all_providers()
        assert "gemma" in providers
        assert isinstance(providers["gemma"], GemmaProvider)
