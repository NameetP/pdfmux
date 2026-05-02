"""Tests for watch mode (mocked watchdog)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner


@pytest.fixture
def mock_watchdog(monkeypatch: pytest.MonkeyPatch):
    """Install a fake `watchdog` package so the CLI command can import it.

    The fake Observer is inert — it just captures schedule() calls so the
    test can fire a synthetic created-event manually.
    """
    fake_pkg = types.ModuleType("watchdog")
    fake_events = types.ModuleType("watchdog.events")
    fake_observers = types.ModuleType("watchdog.observers")

    class FileSystemEventHandler:
        def on_created(self, event: object) -> None:  # noqa: D401
            return None

    captured: dict[str, object] = {"handler": None, "path": None}

    class Observer:
        def schedule(self, handler: object, path: str, recursive: bool = False) -> None:
            captured["handler"] = handler
            captured["path"] = path

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def join(self) -> None:
            return None

    fake_events.FileSystemEventHandler = FileSystemEventHandler  # type: ignore[attr-defined]
    fake_observers.Observer = Observer  # type: ignore[attr-defined]
    fake_pkg.events = fake_events  # type: ignore[attr-defined]
    fake_pkg.observers = fake_observers  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "watchdog", fake_pkg)
    monkeypatch.setitem(sys.modules, "watchdog.events", fake_events)
    monkeypatch.setitem(sys.modules, "watchdog.observers", fake_observers)
    return captured


class TestWatchImportError:
    def test_missing_watchdog_shows_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Force an ImportError by removing watchdog from sys.modules and
        # blocking re-import via meta path.
        for key in list(sys.modules):
            if key == "watchdog" or key.startswith("watchdog."):
                monkeypatch.delitem(sys.modules, key, raising=False)

        class _Blocker:
            def find_spec(self, name: str, path=None, target=None):
                if name == "watchdog" or name.startswith("watchdog."):
                    raise ImportError("blocked for test")
                return None

        monkeypatch.setattr(sys, "meta_path", [_Blocker()] + sys.meta_path)

        from pdfmux.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["watch", str(tmp_path)])
        assert result.exit_code != 0
        assert "watchdog" in result.output.lower()


class TestWatchHappyPath:
    def test_watch_processes_existing_files(
        self,
        mock_watchdog: dict,
        digital_pdf: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Prep watch directory: copy the digital PDF in.
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        (watch_dir / "doc.pdf").write_bytes(digital_pdf.read_bytes())

        # Stub time.sleep so the main loop returns immediately on first
        # iteration via KeyboardInterrupt.
        sleeps = {"n": 0}
        original_sleep = __import__("time").sleep

        def fake_sleep(seconds: float) -> None:
            # First sleep inside main loop — interrupt it.
            sleeps["n"] += 1
            if sleeps["n"] >= 1:
                raise KeyboardInterrupt
            original_sleep(0)

        import pdfmux.cli as cli_mod

        monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)

        from pdfmux.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["watch", str(watch_dir), "--process-existing"],
        )
        assert result.exit_code == 0, result.output
        # Output should contain a processed filename.
        assert "doc.pdf" in result.output
        # Output file should have been written next to / into the watch dir.
        out = watch_dir / "doc.md"
        assert out.exists()

    def test_watch_handles_synthetic_event(
        self,
        mock_watchdog: dict,
        digital_pdf: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        new_pdf = watch_dir / "incoming.pdf"
        new_pdf.write_bytes(digital_pdf.read_bytes())

        # The handler calls time.sleep(0.5) inside on_created. We need to
        # let those short sleeps no-op while making the main 1s sleep raise
        # KeyboardInterrupt — but only after firing one synthetic event.
        state = {"fired": False, "main_loop_calls": 0}

        def fake_sleep(seconds: float) -> None:
            # Short sleeps inside the event handler — no-op.
            if seconds < 1.0:
                return
            # Main loop sleep(1).
            handler = mock_watchdog["handler"]
            if handler is not None and not state["fired"]:
                state["fired"] = True
                event = MagicMock()
                event.is_directory = False
                event.src_path = str(new_pdf)
                handler.on_created(event)
                return
            raise KeyboardInterrupt

        import pdfmux.cli as cli_mod

        monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)

        from pdfmux.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["watch", str(watch_dir)])
        assert result.exit_code == 0, result.output
        assert (watch_dir / "incoming.md").exists()
