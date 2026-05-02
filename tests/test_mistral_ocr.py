"""Tests for the Mistral OCR extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pdfmux.errors import ExtractionError, ExtractorNotAvailable
from pdfmux.extractors.mistral_ocr import (
    DEFAULT_MISTRAL_MODEL,
    MISTRAL_COST_PER_PAGE,
    MistralOCRExtractor,
    _check_mistral_sdk,
    _has_api_key,
)


class TestMistralOCRChecks:
    """Availability checks."""

    def test_check_returns_bool(self) -> None:
        assert isinstance(_check_mistral_sdk(), bool)

    def test_has_api_key_false_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        assert _has_api_key() is False

    def test_has_api_key_true_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        assert _has_api_key() is True


class TestMistralOCRExtractor:
    """Tests for the MistralOCRExtractor class."""

    def test_extractor_name(self) -> None:
        ext = MistralOCRExtractor()
        assert ext.name == "mistral_ocr"

    def test_unavailable_no_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should report unavailable when SDK missing."""
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        ext = MistralOCRExtractor()
        assert ext.available() is False

    def test_unavailable_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should report unavailable when API key missing."""
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: True)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        ext = MistralOCRExtractor()
        assert ext.available() is False

    def test_available_when_both_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: True)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        ext = MistralOCRExtractor()
        assert ext.available() is True

    def test_extract_raises_when_sdk_missing(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        ext = MistralOCRExtractor()
        with pytest.raises(ExtractorNotAvailable, match="Mistral SDK is not installed"):
            list(ext.extract(digital_pdf))

    def test_extract_raises_when_api_key_missing(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: True)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        ext = MistralOCRExtractor()
        with pytest.raises(ExtractorNotAvailable, match="MISTRAL_API_KEY"):
            list(ext.extract(digital_pdf))

    def test_extract_with_mocked_api(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run a full extraction against a mocked Mistral SDK."""
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: True)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        # Build a fake mistralai module so the lazy import resolves.
        fake_client = MagicMock()
        fake_client.files.upload.return_value = SimpleNamespace(id="file-abc")
        fake_client.files.get_signed_url.return_value = SimpleNamespace(
            url="https://signed.example/abc"
        )
        fake_client.ocr.process.return_value = SimpleNamespace(
            pages=[
                SimpleNamespace(
                    index=0,
                    markdown="# Page one\n\nThis is page one content extracted via OCR.",
                    confidence=0.97,
                ),
                SimpleNamespace(
                    index=1,
                    markdown="# Page two\n\nMore extracted content from the second page.",
                    confidence=0.91,
                ),
            ],
            usage_info=None,
        )

        fake_module = MagicMock()
        fake_module.Mistral = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "mistralai", fake_module)

        ext = MistralOCRExtractor()
        results = list(ext.extract(digital_pdf))

        assert len(results) == 2
        assert results[0].page_num == 0
        assert "Page one" in results[0].text
        assert results[0].confidence == pytest.approx(0.97)
        assert results[0].cost_usd == MISTRAL_COST_PER_PAGE
        assert results[0].ocr_applied is True
        assert results[0].extractor == "mistral_ocr"

        assert results[1].page_num == 1
        assert results[1].confidence == pytest.approx(0.91)

        # Verify the SDK was driven correctly
        fake_client.files.upload.assert_called_once()
        fake_client.files.get_signed_url.assert_called_once_with(file_id="file-abc")
        process_kwargs = fake_client.ocr.process.call_args.kwargs
        assert process_kwargs["model"] == DEFAULT_MISTRAL_MODEL
        assert process_kwargs["document"]["type"] == "document_url"
        assert process_kwargs["document"]["document_url"] == "https://signed.example/abc"

    def test_extract_pages_filter(self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Page filtering should restrict the yielded results."""
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: True)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        fake_client = MagicMock()
        fake_client.files.upload.return_value = SimpleNamespace(id="file-xyz")
        fake_client.files.get_signed_url.return_value = SimpleNamespace(
            url="https://signed.example/xyz"
        )
        fake_client.ocr.process.return_value = SimpleNamespace(
            pages=[
                SimpleNamespace(index=0, markdown="page zero text long enough"),
                SimpleNamespace(index=1, markdown="page one text long enough"),
                SimpleNamespace(index=2, markdown="page two text long enough"),
            ],
            usage_info=None,
        )

        fake_module = MagicMock()
        fake_module.Mistral = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "mistralai", fake_module)

        ext = MistralOCRExtractor()
        results = list(ext.extract(digital_pdf, pages=[1]))
        assert len(results) == 1
        assert results[0].page_num == 1

    def test_extract_handles_api_error(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Network/API errors should surface as ExtractionError."""
        monkeypatch.setattr("pdfmux.extractors.mistral_ocr._check_mistral_sdk", lambda: True)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        fake_client = MagicMock()
        fake_client.files.upload.side_effect = RuntimeError("rate limit hit")

        fake_module = MagicMock()
        fake_module.Mistral = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "mistralai", fake_module)

        ext = MistralOCRExtractor()
        with pytest.raises(ExtractionError, match="Mistral upload failed"):
            list(ext.extract(digital_pdf))

    def test_registry_priority(self) -> None:
        """Should be registered with priority 25."""
        from pdfmux.extractors import _REGISTRY

        entry = [(p, n) for p, n, _ in _REGISTRY if n == "mistral_ocr"]
        assert entry, "mistral_ocr not found in registry"
        assert entry[0][0] == 25
