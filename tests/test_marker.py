"""Tests for the Marker extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pdfmux.errors import ExtractionError, ExtractorNotAvailable
from pdfmux.extractors.marker import (
    MarkerExtractor,
    _check_marker,
    _extract_markdown,
    _split_pages,
)


class TestMarkerCheck:
    """Availability check."""

    def test_check_returns_bool(self) -> None:
        assert isinstance(_check_marker(), bool)


class TestExtractMarkdown:
    """Helpers for parsing Marker's rendered output."""

    def test_extract_from_markdown_attr(self) -> None:
        rendered = SimpleNamespace(markdown="# Hello\n\nbody")
        assert _extract_markdown(rendered) == "# Hello\n\nbody"

    def test_extract_from_tuple(self) -> None:
        rendered = ("# Tuple form", {"meta": True}, [])
        assert _extract_markdown(rendered) == "# Tuple form"

    def test_extract_handles_none(self) -> None:
        assert _extract_markdown(None) == ""


class TestSplitPages:
    """Page-splitting helper."""

    def test_split_on_separator(self) -> None:
        md = "page one\n\n---\n\npage two\n\n---\n\npage three"
        chunks = _split_pages(md, expected_pages=3)
        assert len(chunks) == 3
        assert chunks[0] == "page one"
        assert chunks[2] == "page three"

    def test_no_separator_returns_single_chunk(self) -> None:
        md = "no separator just text"
        chunks = _split_pages(md, expected_pages=1)
        assert chunks == ["no separator just text"]

    def test_strips_blank_chunks(self) -> None:
        md = "\n\n---\n\nactual content\n\n---\n\n"
        chunks = _split_pages(md, expected_pages=1)
        assert chunks == ["actual content"]


class TestMarkerExtractor:
    """Tests for the MarkerExtractor class."""

    def test_extractor_name(self) -> None:
        ext = MarkerExtractor()
        assert ext.name == "marker"

    def test_unavailable_when_marker_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pdfmux.extractors.marker._check_marker", lambda: False)
        ext = MarkerExtractor()
        assert ext.available() is False

    def test_extract_raises_when_unavailable(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pdfmux.extractors.marker._check_marker", lambda: False)
        ext = MarkerExtractor()
        with pytest.raises(ExtractorNotAvailable, match="Marker is not installed"):
            list(ext.extract(digital_pdf))

    def test_extract_with_mocked_converter(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run a full extraction against a mocked Marker converter."""
        monkeypatch.setattr("pdfmux.extractors.marker._check_marker", lambda: True)

        # Reset module-level singletons so our patched _get_converter wins
        monkeypatch.setattr("pdfmux.extractors.marker._converter_instance", None)
        monkeypatch.setattr("pdfmux.extractors.marker._models_instance", None)

        rendered = SimpleNamespace(
            markdown=(
                "# Page one\n\nFirst page body, long enough to be valid."
                "\n\n---\n\n"
                "# Page two\n\nSecond page body, also long enough."
            )
        )
        fake_converter = MagicMock(return_value=rendered)
        monkeypatch.setattr(
            "pdfmux.extractors.marker._get_converter", lambda: fake_converter
        )

        ext = MarkerExtractor()
        results = list(ext.extract(digital_pdf))

        assert len(results) == 2
        assert results[0].page_num == 0
        assert "Page one" in results[0].text
        assert results[0].extractor == "marker"
        assert results[0].confidence > 0.9
        assert results[1].page_num == 1
        assert "Page two" in results[1].text

        fake_converter.assert_called_once_with(str(digital_pdf))

    def test_extract_falls_back_when_split_mismatches(
        self, multi_page_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When Marker doesn't emit page separators, yield one combined page."""
        monkeypatch.setattr("pdfmux.extractors.marker._check_marker", lambda: True)
        monkeypatch.setattr("pdfmux.extractors.marker._converter_instance", None)
        monkeypatch.setattr("pdfmux.extractors.marker._models_instance", None)

        rendered = SimpleNamespace(markdown="One big chunk of text without separators.")
        fake_converter = MagicMock(return_value=rendered)
        monkeypatch.setattr(
            "pdfmux.extractors.marker._get_converter", lambda: fake_converter
        )

        ext = MarkerExtractor()
        results = list(ext.extract(multi_page_pdf))

        # multi_page_pdf has 5 pages, but markdown has no separator → 1 result
        assert len(results) == 1
        assert "One big chunk" in results[0].text

    def test_extract_pages_filter(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pdfmux.extractors.marker._check_marker", lambda: True)
        monkeypatch.setattr("pdfmux.extractors.marker._converter_instance", None)
        monkeypatch.setattr("pdfmux.extractors.marker._models_instance", None)

        rendered = SimpleNamespace(
            markdown="page zero text long enough\n\n---\n\npage one text long enough"
        )
        fake_converter = MagicMock(return_value=rendered)
        monkeypatch.setattr(
            "pdfmux.extractors.marker._get_converter", lambda: fake_converter
        )

        ext = MarkerExtractor()
        results = list(ext.extract(digital_pdf, pages=[1]))
        assert len(results) == 1
        assert results[0].page_num == 1

    def test_extract_handles_conversion_error(
        self, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Conversion errors should surface as ExtractionError."""
        monkeypatch.setattr("pdfmux.extractors.marker._check_marker", lambda: True)
        monkeypatch.setattr("pdfmux.extractors.marker._converter_instance", None)
        monkeypatch.setattr("pdfmux.extractors.marker._models_instance", None)

        def boom():
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr("pdfmux.extractors.marker._get_converter", boom)

        ext = MarkerExtractor()
        with pytest.raises(ExtractionError, match="Marker conversion failed"):
            list(ext.extract(digital_pdf))

    def test_registry_priority(self) -> None:
        """Should be registered with priority 35."""
        from pdfmux.extractors import _REGISTRY

        entry = [(p, n) for p, n, _ in _REGISTRY if n == "marker"]
        assert entry, "marker not found in registry"
        assert entry[0][0] == 35
