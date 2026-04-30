"""Tests for the diff command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pdfmux.cli import app

runner = CliRunner()


class TestDiffCommand:
    def test_diff_same_file_runs(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["diff", str(digital_pdf), str(digital_pdf)])
        assert result.exit_code == 0, result.output
        assert "Page count" in result.output

    def test_diff_two_files_table(self, digital_pdf: Path, multi_page_pdf: Path) -> None:
        result = runner.invoke(app, ["diff", str(digital_pdf), str(multi_page_pdf)])
        assert result.exit_code == 0
        assert "Page count" in result.output
        assert "Total chars" in result.output
        assert "Confidence" in result.output

    def test_diff_unified_format(self, digital_pdf: Path, multi_page_pdf: Path) -> None:
        result = runner.invoke(
            app, ["diff", str(digital_pdf), str(multi_page_pdf), "--format", "unified"]
        )
        assert result.exit_code == 0

    def test_diff_unified_identical_files(self, digital_pdf: Path) -> None:
        result = runner.invoke(
            app, ["diff", str(digital_pdf), str(digital_pdf), "--format", "unified"]
        )
        assert result.exit_code == 0
        assert "identical" in result.output.lower()

    def test_diff_nonexistent_file_fails(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["diff", str(digital_pdf), "/tmp/nope-abc123.pdf"])
        assert result.exit_code != 0

    def test_diff_shows_delta_for_page_count(
        self, digital_pdf: Path, multi_page_pdf: Path
    ) -> None:
        result = runner.invoke(app, ["diff", str(digital_pdf), str(multi_page_pdf)])
        assert result.exit_code == 0
        # multi_page has 5 pages, digital has 2 → +3 delta somewhere
        assert "+3" in result.output or "+5" in result.output or "+2" in result.output
