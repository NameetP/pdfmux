"""Tests for providers base classes and discovery."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestModelInfo:
    def test_creation(self):
        from pdfmux.providers.base import ModelInfo

        m = ModelInfo(id="test-model", capabilities=("ocr", "tables"))
        assert m.id == "test-model"
        assert m.supports_vision is True
        assert "ocr" in m.capabilities

    def test_defaults(self):
        from pdfmux.providers.base import ModelInfo

        m = ModelInfo(id="test")
        assert m.supports_vision is True
        assert m.capabilities == ()
        assert m.input_cost_per_mtok is None
        assert m.max_input_tokens == 128_000

    def test_frozen(self):
        from pdfmux.providers.base import ModelInfo

        m = ModelInfo(id="test")
        with pytest.raises(AttributeError):
            m.id = "changed"


class TestCostEstimate:
    def test_creation(self):
        from pdfmux.providers.base import CostEstimate

        c = CostEstimate(input_tokens=100, output_tokens=50, cost_usd=0.001)
        assert c.input_tokens == 100
        assert c.cost_usd == 0.001

    def test_zero_defaults(self):
        from pdfmux.providers.base import CostEstimate

        c = CostEstimate()
        assert c.input_tokens == 0
        assert c.cost_usd == 0.0


class TestLLMProviderABC:
    def test_cannot_instantiate_directly(self):
        from pdfmux.providers.base import LLMProvider

        with pytest.raises(TypeError):
            LLMProvider()

    def test_default_estimate_cost_is_zero(self):
        from pdfmux.providers.base import LLMProvider

        class DummyProvider(LLMProvider):
            name = "dummy"
            default_model = "dummy-v1"

            def available(self):
                return True

            def extract_page(self, image_bytes, prompt, model=None):
                return "text"

        p = DummyProvider()
        cost = p.estimate_cost(1000)
        assert cost.cost_usd == 0.0

    def test_supported_models_default(self):
        from pdfmux.providers.base import LLMProvider

        class DummyProvider(LLMProvider):
            name = "dummy"
            default_model = "dummy-v1"

            def available(self):
                return True

            def extract_page(self, image_bytes, prompt, model=None):
                return "text"

        p = DummyProvider()
        models = p.supported_models()
        assert len(models) == 1
        assert models[0].id == "dummy-v1"


class TestOpenAICompatibleProvider:
    def test_creation(self):
        from pdfmux.providers.openai_compatible import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            name="kimi",
            base_url="https://api.moonshot.cn/v1",
            api_key_env="KIMI_API_KEY",
            models=[{"id": "moonshot-v1-128k", "supports_vision": True}],
        )
        assert p.name == "kimi"
        assert p.default_model == "moonshot-v1-128k"

    def test_available_with_sdk_and_key(self):
        from pdfmux.providers.openai_compatible import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            name="test",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            models=[{"id": "test-model"}],
        )
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
        ):
            assert p.available() is True

    def test_unavailable_without_key(self):
        from pdfmux.providers.openai_compatible import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            name="test",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            models=[{"id": "test-model"}],
        )
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert p.available() is False

    def test_supported_models(self):
        from pdfmux.providers.openai_compatible import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            name="test",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            models=[
                {"id": "model-a", "capabilities": ["ocr"], "input_cost_per_mtok": 1.0},
                {"id": "model-b", "capabilities": ["tables", "charts"]},
            ],
        )
        models = p.supported_models()
        assert len(models) == 2
        assert models[0].id == "model-a"
        assert models[0].input_cost_per_mtok == 1.0
        assert "tables" in models[1].capabilities

    def test_cost_estimate(self):
        from pdfmux.providers.openai_compatible import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            name="test",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            models=[{"id": "m", "input_cost_per_mtok": 2.0, "output_cost_per_mtok": 6.0}],
        )
        cost = p.estimate_cost(50000)
        assert cost.cost_usd > 0


class TestDiscovery:
    def test_discover_returns_builtin_providers(self):
        from pdfmux.providers._discovery import discover_all_providers

        providers = discover_all_providers()
        assert "gemini" in providers
        assert "claude" in providers
        assert "openai" in providers
        assert "ollama" in providers

    def test_config_file_loading(self):
        from pdfmux.providers._discovery import _parse_config

        yaml_content = """
providers:
  deepseek:
    type: openai_compatible
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    models:
      - id: deepseek-chat
        supports_vision: true
        capabilities: [ocr]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            providers = _parse_config(Path(f.name))

        assert len(providers) == 1
        assert providers[0].name == "deepseek"
        assert providers[0].default_model == "deepseek-chat"

    def test_config_file_skips_unknown_type(self):
        from pdfmux.providers._discovery import _parse_config

        yaml_content = """
providers:
  custom:
    type: unsupported_type
    base_url: https://example.com
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            providers = _parse_config(Path(f.name))

        assert len(providers) == 0

    def test_resolve_provider_with_config(self):
        from pdfmux.providers._discovery import discover_all_providers

        # Just verify discovery works without errors
        providers = discover_all_providers()
        assert len(providers) >= 4  # at least the 4 built-ins


class TestBackwardCompat:
    """Verify the shim in extractors/llm_providers.py works."""

    def test_import_providers_from_old_path(self):
        from pdfmux.extractors.llm_providers import (
            ClaudeProvider,
            GeminiProvider,
            OllamaProvider,
            OpenAIProvider,
        )

        assert GeminiProvider.name == "gemini"
        assert ClaudeProvider.name == "claude"
        assert OpenAIProvider.name == "openai"
        assert OllamaProvider.name == "ollama"

    def test_import_functions_from_old_path(self):
        from pdfmux.extractors.llm_providers import (
            all_provider_status,
            available_providers,
            resolve_provider,
        )

        assert callable(resolve_provider)
        assert callable(available_providers)
        assert callable(all_provider_status)

    def test_providers_list_from_old_path(self):
        from pdfmux.extractors.llm_providers import PROVIDERS

        assert len(PROVIDERS) == 4

    def test_import_new_types_from_old_path(self):
        from pdfmux.extractors.llm_providers import CostEstimate, LLMProvider, ModelInfo

        assert LLMProvider is not None
        assert ModelInfo is not None
        assert CostEstimate is not None
