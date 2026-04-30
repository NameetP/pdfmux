"""Tests for the estimate command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pdfmux.cli import app

runner = CliRunner()


class TestEstimateCommand:
    def test_runs_on_digital_pdf(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["estimate", str(digital_pdf)])
        assert result.exit_code == 0, result.output
        assert digital_pdf.name in result.output

    def test_economy_mode_runs(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["estimate", str(digital_pdf), "--mode", "economy"])
        assert result.exit_code == 0

    def test_balanced_mode_runs(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["estimate", str(digital_pdf), "--mode", "balanced"])
        assert result.exit_code == 0

    def test_premium_mode_runs(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["estimate", str(digital_pdf), "--mode", "premium"])
        assert result.exit_code == 0

    def test_invalid_mode_fails(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["estimate", str(digital_pdf), "--mode", "wat"])
        assert result.exit_code != 0

    def test_nonexistent_file_fails(self) -> None:
        result = runner.invoke(app, ["estimate", "/tmp/no-such-file-abc123.pdf"])
        assert result.exit_code != 0

    def test_output_includes_provider_columns(self, digital_pdf: Path) -> None:
        result = runner.invoke(app, ["estimate", str(digital_pdf), "--mode", "premium"])
        assert result.exit_code == 0
        # At least one of the built-in provider names should appear in the table.
        out = result.output.lower()
        assert any(p in out for p in ("gemini", "claude", "openai", "ollama"))


class TestEstimateMath:
    """Direct unit tests on the cost-estimation primitives we depend on."""

    def test_router_engine_estimate_returns_number(self) -> None:
        from pdfmux.router.engine import RouterEngine
        from pdfmux.router.strategies import Strategy

        engine = RouterEngine()
        cost = engine.estimate_document_cost(["digital", "digital"], Strategy.ECONOMY)
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_provider_estimate_cost_is_nonnegative(self) -> None:
        from pdfmux.providers.gemini import GeminiProvider

        c = GeminiProvider().estimate_cost(image_bytes_count=200_000)
        assert c.cost_usd >= 0.0
        assert c.input_tokens > 0
