"""Tests for BYOK multi-LLM provider support."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    def test_available_with_sdk_and_key(self):
        from pdfmux.extractors.llm_providers import GeminiProvider

        p = GeminiProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            assert p.available() is True

    def test_unavailable_without_key(self):
        from pdfmux.extractors.llm_providers import GeminiProvider

        p = GeminiProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert p.available() is False

    def test_unavailable_without_sdk(self):
        from pdfmux.extractors.llm_providers import GeminiProvider

        p = GeminiProvider()
        with (
            patch.object(p, "sdk_installed", return_value=False),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            assert p.available() is False


class TestClaudeProvider:
    def test_available_with_sdk_and_key(self):
        from pdfmux.extractors.llm_providers import ClaudeProvider

        p = ClaudeProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        ):
            assert p.available() is True

    def test_unavailable_without_key(self):
        from pdfmux.extractors.llm_providers import ClaudeProvider

        p = ClaudeProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert p.available() is False


class TestOpenAIProvider:
    def test_available_with_sdk_and_key(self):
        from pdfmux.extractors.llm_providers import OpenAIProvider

        p = OpenAIProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            assert p.available() is True

    def test_unavailable_without_key(self):
        from pdfmux.extractors.llm_providers import OpenAIProvider

        p = OpenAIProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert p.available() is False


class TestOllamaProvider:
    def test_available_with_sdk_and_model(self):
        from pdfmux.extractors.llm_providers import OllamaProvider

        p = OllamaProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"PDFMUX_LLM_MODEL": "llava"}),
        ):
            assert p.available() is True

    def test_unavailable_without_model(self):
        from pdfmux.extractors.llm_providers import OllamaProvider

        p = OllamaProvider()
        with (
            patch.object(p, "sdk_installed", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert p.available() is False


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class TestResolveProvider:
    def test_explicit_provider_name(self):
        from pdfmux.extractors.llm_providers import GeminiProvider, resolve_provider

        with (
            patch.object(GeminiProvider, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test"}),
        ):
            p = resolve_provider(provider_name="gemini")
            assert p.name == "gemini"

    def test_explicit_env_var_override(self):
        from pdfmux.extractors.llm_providers import ClaudeProvider, resolve_provider

        with (
            patch.object(ClaudeProvider, "sdk_installed", return_value=True),
            patch.dict(
                "os.environ",
                {"PDFMUX_LLM_PROVIDER": "claude", "ANTHROPIC_API_KEY": "sk-ant-test"},
            ),
        ):
            p = resolve_provider()
            assert p.name == "claude"

    def test_auto_detect_gemini_first(self):
        from pdfmux.extractors.llm_providers import (
            ClaudeProvider,
            GeminiProvider,
            resolve_provider,
        )

        with (
            patch.object(GeminiProvider, "sdk_installed", return_value=True),
            patch.object(ClaudeProvider, "sdk_installed", return_value=True),
            patch.dict(
                "os.environ",
                {"GEMINI_API_KEY": "test", "ANTHROPIC_API_KEY": "sk-ant-test"},
            ),
        ):
            p = resolve_provider()
            assert p.name == "gemini"

    def test_auto_detect_claude_when_no_gemini(self):
        from pdfmux.extractors.llm_providers import (
            ClaudeProvider,
            GeminiProvider,
            resolve_provider,
        )

        with (
            patch.object(GeminiProvider, "sdk_installed", return_value=False),
            patch.object(ClaudeProvider, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
        ):
            p = resolve_provider()
            assert p.name == "claude"

    def test_unknown_provider_raises(self):
        from pdfmux.extractors.llm_providers import resolve_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            resolve_provider(provider_name="fake-provider")

    def test_no_provider_available_raises(self):
        from pdfmux.extractors.llm_providers import resolve_provider

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No LLM provider available"):
                resolve_provider()

    def test_provider_sdk_not_installed_raises(self):
        from pdfmux.extractors.llm_providers import ClaudeProvider, resolve_provider

        with patch.object(ClaudeProvider, "sdk_installed", return_value=False):
            with pytest.raises(ValueError, match="SDK not installed"):
                resolve_provider(provider_name="claude")


# ---------------------------------------------------------------------------
# all_provider_status
# ---------------------------------------------------------------------------


class TestAllProviderStatus:
    def test_returns_all_four_providers(self):
        from pdfmux.extractors.llm_providers import all_provider_status

        status = all_provider_status()
        names = [p["name"] for p in status]
        assert "gemini" in names
        assert "claude" in names
        assert "openai" in names
        assert "ollama" in names

    def test_status_fields(self):
        from pdfmux.extractors.llm_providers import all_provider_status

        status = all_provider_status()
        for p in status:
            assert "name" in p
            assert "sdk_installed" in p
            assert "has_credentials" in p
            assert "available" in p
            assert "default_model" in p


# ---------------------------------------------------------------------------
# LLMExtractor integration
# ---------------------------------------------------------------------------


class TestLLMExtractorAvailable:
    def test_available_when_provider_exists(self):
        from pdfmux.extractors.llm import LLMExtractor
        from pdfmux.extractors.llm_providers import GeminiProvider

        ext = LLMExtractor()
        with (
            patch.object(GeminiProvider, "sdk_installed", return_value=True),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test"}),
        ):
            assert ext.available() is True

    def test_unavailable_when_no_provider(self):
        from pdfmux.extractors.llm import LLMExtractor

        ext = LLMExtractor()
        with patch.dict("os.environ", {}, clear=True):
            assert ext.available() is False
