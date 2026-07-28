"""Tests for the pdfmux analyze CLI command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from conftest import assert_cli_ok

from pdfmux.cli import app

runner = CliRunner()


def test_analyze_digital_pdf(digital_pdf: Path) -> None:
    """analyze should show per-page breakdown for a digital PDF."""
    result = runner.invoke(app, ["analyze", str(digital_pdf)])
    assert_cli_ok(result)
    assert "Confidence" in result.output
    assert "Extractor" in result.output


def test_analyze_shows_page_count(multi_page_pdf: Path) -> None:
    """analyze should show the correct page count."""
    result = runner.invoke(app, ["analyze", str(multi_page_pdf)])
    assert_cli_ok(result)
    assert "5 pages" in result.output


def test_analyze_shows_page_quality(digital_pdf: Path) -> None:
    """analyze should show quality classification for each page."""
    result = runner.invoke(app, ["analyze", str(digital_pdf)])
    assert_cli_ok(result)
    # Digital PDF should have "good" or "digital" indicators
    assert "good" in result.output.lower() or "digital" in result.output.lower()
