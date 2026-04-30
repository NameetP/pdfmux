"""Tests for configuration profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfmux import profiles
from pdfmux.profiles import (
    BUILTIN_PROFILES,
    apply_profile_defaults,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect profiles storage to a tmp dir."""
    monkeypatch.setenv("PDFMUX_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return tmp_path / "pdfmux"


class TestBuiltinPresets:
    def test_invoices_preset_exists(self, isolated_config: Path) -> None:
        s = load_profile("invoices")
        assert s["schema"] == "invoice"
        assert s["format"] == "json"

    def test_receipts_preset_exists(self, isolated_config: Path) -> None:
        s = load_profile("receipts")
        assert s["schema"] == "receipt"

    def test_papers_preset_uses_chunking(self, isolated_config: Path) -> None:
        s = load_profile("papers")
        assert s.get("chunk") is True
        assert s.get("max_tokens", 0) > 0

    def test_contracts_preset_uses_premium(self, isolated_config: Path) -> None:
        s = load_profile("contracts")
        assert s.get("mode") == "premium"

    def test_bulk_rag_preset_uses_economy(self, isolated_config: Path) -> None:
        s = load_profile("bulk-rag")
        assert s.get("mode") == "economy"
        assert s.get("chunk") is True

    def test_all_presets_have_recognized_keys(self, isolated_config: Path) -> None:
        for name, settings in BUILTIN_PROFILES.items():
            for key in settings:
                assert key in profiles.ALLOWED_KEYS, f"{name} has bad key {key}"


class TestSaveLoadDelete:
    def test_save_then_load_roundtrip(self, isolated_config: Path) -> None:
        path = save_profile("custom1", {"quality": "high", "mode": "balanced"})
        assert path.exists()
        loaded = load_profile("custom1")
        assert loaded["quality"] == "high"
        assert loaded["mode"] == "balanced"

    def test_user_profile_overrides_builtin(self, isolated_config: Path) -> None:
        save_profile("invoices", {"quality": "fast", "mode": "economy"})
        loaded = load_profile("invoices")
        assert loaded["quality"] == "fast"
        assert loaded["mode"] == "economy"

    def test_save_rejects_unknown_keys(self, isolated_config: Path) -> None:
        with pytest.raises(ValueError):
            save_profile("bad", {"not_a_real_key": 1})

    def test_save_rejects_empty_name(self, isolated_config: Path) -> None:
        with pytest.raises(ValueError):
            save_profile("", {"quality": "high"})

    def test_delete_user_profile(self, isolated_config: Path) -> None:
        save_profile("temp", {"quality": "high"})
        assert delete_profile("temp") is True
        with pytest.raises(KeyError):
            load_profile("temp")

    def test_delete_missing_profile_returns_false(self, isolated_config: Path) -> None:
        assert delete_profile("does-not-exist") is False

    def test_delete_builtin_raises(self, isolated_config: Path) -> None:
        with pytest.raises(ValueError):
            delete_profile("invoices")

    def test_load_missing_raises_key_error(self, isolated_config: Path) -> None:
        with pytest.raises(KeyError):
            load_profile("nope")


class TestListProfiles:
    def test_list_includes_all_builtins(self, isolated_config: Path) -> None:
        rows = list_profiles()
        names = {n for n, _ in rows}
        for builtin in BUILTIN_PROFILES:
            assert builtin in names

    def test_user_profile_appears_with_user_source(self, isolated_config: Path) -> None:
        save_profile("mine", {"quality": "high"})
        rows = list_profiles()
        sources = dict(rows)
        assert sources["mine"] == "user"

    def test_user_override_marked_as_user(self, isolated_config: Path) -> None:
        save_profile("invoices", {"quality": "fast"})
        rows = list_profiles()
        sources = dict(rows)
        assert sources["invoices"] == "user"


class TestApplyProfileDefaults:
    def test_applies_when_explicit_is_none(self, isolated_config: Path) -> None:
        merged = apply_profile_defaults(
            "invoices",
            {"quality": None, "format": None, "schema": None},
        )
        assert merged["schema"] == "invoice"
        assert merged["format"] == "json"

    def test_explicit_wins(self, isolated_config: Path) -> None:
        merged = apply_profile_defaults(
            "invoices",
            {"format": "markdown", "schema": None, "quality": None},
        )
        assert merged["format"] == "markdown"  # explicit wins

    def test_no_profile_passes_through(self, isolated_config: Path) -> None:
        merged = apply_profile_defaults(None, {"quality": "fast"})
        assert merged == {"quality": "fast"}

    def test_unknown_profile_raises(self, isolated_config: Path) -> None:
        with pytest.raises(KeyError):
            apply_profile_defaults("does-not-exist", {})


class TestProfilesCli:
    """Smoke-test the CLI subcommands route through profiles correctly."""

    def test_list_runs(self, isolated_config: Path) -> None:
        from typer.testing import CliRunner

        from pdfmux.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["profiles", "list"])
        assert result.exit_code == 0
        assert "invoices" in result.output

    def test_show_builtin(self, isolated_config: Path) -> None:
        from typer.testing import CliRunner

        from pdfmux.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["profiles", "show", "invoices"])
        assert result.exit_code == 0
        assert "invoice" in result.output

    def test_save_then_show(self, isolated_config: Path) -> None:
        from typer.testing import CliRunner

        from pdfmux.cli import app

        runner = CliRunner()
        save = runner.invoke(
            app,
            ["profiles", "save", "myprof", "--quality", "high", "--mode", "premium"],
        )
        assert save.exit_code == 0, save.output
        show = runner.invoke(app, ["profiles", "show", "myprof"])
        assert show.exit_code == 0
        assert "premium" in show.output

    def test_delete_user_profile_via_cli(self, isolated_config: Path) -> None:
        from typer.testing import CliRunner

        from pdfmux.cli import app

        runner = CliRunner()
        runner.invoke(app, ["profiles", "save", "tmp", "--quality", "high"])
        result = runner.invoke(app, ["profiles", "delete", "tmp"])
        assert result.exit_code == 0
