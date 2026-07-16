"""Tests for the hard-terminating extraction timeout (``pdfmux._timeout``).

Covers both backends (daemon-thread and forked-process), the SIGTERM→SIGKILL
escalation that makes the process backend a *hard* kill, and the pipeline
mapping of :class:`HardTimeoutError` to :class:`OCRTimeoutError`.

The process-backend tests are skipped where ``fork`` is unavailable (Windows).
Their target functions live at module scope so they survive a fork cleanly and
carry no native-library state.
"""

from __future__ import annotations

import signal
import time

import pytest

from pdfmux._timeout import HardTimeoutError, _fork_available, run_with_timeout

fork_only = pytest.mark.skipif(not _fork_available(), reason="requires os.fork")


# --- module-level targets (fork-safe, no native state) ---------------------


def _add(a: int, b: int) -> int:
    return a + b


def _sleep_then(secs: float, val: str) -> str:
    time.sleep(secs)
    return val


def _raise_value(msg: str) -> None:
    raise ValueError(msg)


def _ignore_sigterm_then_sleep(secs: float) -> str:
    # Refuse SIGTERM so only SIGKILL can stop us — proves the escalation path.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(secs)
    return "should-never-return"


# --- thread backend --------------------------------------------------------


def test_thread_returns_result(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "thread")
    assert run_with_timeout(_add, (2, 3), 5) == 5


def test_thread_propagates_exception(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "thread")
    with pytest.raises(ValueError, match="boom"):
        run_with_timeout(_raise_value, ("boom",), 5)


def test_thread_timeout_frees_caller_fast(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "thread")
    start = time.monotonic()
    with pytest.raises(HardTimeoutError):
        run_with_timeout(_sleep_then, (30, "x"), 0.5)
    # The whole point: the caller is freed at the deadline, NOT after 30s.
    assert time.monotonic() - start < 5.0


# --- process backend -------------------------------------------------------


@fork_only
def test_process_returns_result(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "process")
    assert run_with_timeout(_add, (7, 8), 5) == 15


@fork_only
def test_process_propagates_exception(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "process")
    with pytest.raises(ValueError, match="kaboom"):
        run_with_timeout(_raise_value, ("kaboom",), 5)


@fork_only
def test_process_timeout_frees_caller_fast(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "process")
    start = time.monotonic()
    with pytest.raises(HardTimeoutError):
        run_with_timeout(_sleep_then, (30, "x"), 0.5)
    assert time.monotonic() - start < 6.0


@fork_only
def test_process_hard_kills_uncooperative_child(monkeypatch):
    """A child that ignores SIGTERM must still be SIGKILL-ed — and the caller
    must return in the grace window, never after the child's 30s sleep."""
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "process")
    start = time.monotonic()
    with pytest.raises(HardTimeoutError):
        run_with_timeout(_ignore_sigterm_then_sleep, (30,), 0.5)
    elapsed = time.monotonic() - start
    # 0.5s deadline + SIGTERM grace + SIGKILL grace, with margin — and well under
    # the 30s the child would have slept if we'd waited for it.
    assert elapsed < 10.0


# --- off mode --------------------------------------------------------------


def test_off_mode_runs_inline(monkeypatch):
    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "off")
    assert run_with_timeout(_add, (1, 1), 0.001) == 2  # no deadline applied


# --- pipeline integration --------------------------------------------------


def test_pipeline_maps_hardtimeout_to_ocrtimeout(monkeypatch, digital_pdf):
    """A wedged extraction surfaces as the domain OCRTimeoutError, not a hang."""
    from pdfmux import pipeline
    from pdfmux.errors import OCRTimeoutError

    monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "thread")  # fast + cross-platform
    monkeypatch.setattr(pipeline, "EXTRACTION_TIMEOUT_S", 0.5, raising=True)

    def _wedge(*_args, **_kwargs):
        time.sleep(30)

    monkeypatch.setattr(pipeline, "_route_and_extract", _wedge, raising=True)

    start = time.monotonic()
    with pytest.raises(OCRTimeoutError) as exc:
        pipeline.process(str(digital_pdf))
    assert "timed out" in str(exc.value).lower()
    assert time.monotonic() - start < 5.0  # freed at the deadline, not after 30s
