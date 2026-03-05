"""Tests for PDF type detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfmux.detect import classify
from pdfmux.errors import FileError


def test_classify_digital_pdf(digital_pdf: Path) -> None:
    """Digital PDFs should be classified as digital."""
    result = classify(digital_pdf)
    assert result.is_digital
    assert not result.is_scanned
    assert result.page_count == 2
    assert result.confidence > 0.5


def test_classify_empty_pdf(empty_pdf: Path) -> None:
    """Empty PDFs should still be classified without errors."""
    result = classify(empty_pdf)
    assert result.page_count == 1
    # Empty pages are treated as digital (no images)
    assert result.is_digital


def test_classify_multi_page(multi_page_pdf: Path) -> None:
    """Multi-page PDFs should report correct page count."""
    result = classify(multi_page_pdf)
    assert result.page_count == 5
    assert result.is_digital


def test_classify_nonexistent_file() -> None:
    """Should raise FileError for missing files."""
    with pytest.raises(FileError):
        classify("/nonexistent/file.pdf")


def test_classify_non_pdf(tmp_path: Path) -> None:
    """Should raise FileError for non-PDF files."""
    txt = tmp_path / "test.txt"
    txt.write_text("not a pdf")
    with pytest.raises(FileError):
        classify(txt)
