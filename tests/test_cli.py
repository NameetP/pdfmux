"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pdfmux.cli import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_command(self) -> None:
        """pdfmux version should print version string."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "pdfmux" in result.output

    def test_version_flag(self) -> None:
        """pdfmux --version should print version and exit."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0


class TestDoctorCommand:
    """Tests for the doctor command."""

    def test_doctor_runs(self) -> None:
        """pdfmux doctor should list extractors without error."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "PyMuPDF" in result.output
        assert "pymupdf4llm" in result.output

    def test_doctor_shows_version(self) -> None:
        """pdfmux doctor should show the pdfmux version."""
        result = runner.invoke(app, ["doctor"])
        assert "pdfmux" in result.output


class TestConvertCommand:
    """Tests for the convert command."""

    def test_convert_to_stdout(self, digital_pdf: Path) -> None:
        """pdfmux convert --stdout should print markdown."""
        result = runner.invoke(app, ["convert", str(digital_pdf), "--stdout"])
        assert result.exit_code == 0
        assert len(result.output) > 0

    def test_convert_to_file(self, digital_pdf: Path, tmp_path: Path) -> None:
        """pdfmux convert -o should write output to file."""
        output = tmp_path / "output.md"
        result = runner.invoke(app, ["convert", str(digital_pdf), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        assert len(output.read_text()) > 0

    def test_convert_json_format(self, digital_pdf: Path, tmp_path: Path) -> None:
        """pdfmux convert -f json should produce valid JSON."""
        import json

        output = tmp_path / "output.json"
        result = runner.invoke(
            app, ["convert", str(digital_pdf), "-f", "json", "-o", str(output)]
        )
        assert result.exit_code == 0
        data = json.loads(output.read_text())
        assert "page_count" in data

    def test_convert_nonexistent_file(self) -> None:
        """pdfmux convert on nonexistent file should fail."""
        result = runner.invoke(app, ["convert", "/tmp/nonexistent_abc123.pdf"])
        assert result.exit_code != 0


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_runs(self, digital_pdf: Path) -> None:
        """pdfmux analyze should show per-page breakdown."""
        result = runner.invoke(app, ["analyze", str(digital_pdf)])
        assert result.exit_code == 0
        assert "Confidence" in result.output
