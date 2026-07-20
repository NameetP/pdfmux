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

from pdfmux._timeout import HardTimeoutError, _start_method, run_with_timeout

fork_only = pytest.mark.skipif(
    _start_method() is None, reason="requires a forkserver-capable multiprocessing"
)


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


# --- concurrency: the case that shipped broken -----------------------------


def test_process_backend_never_uses_bare_fork() -> None:
    """The process backend must never choose a bare ``fork``.

    This is the deterministic guard on the 2026-07-20 bug, and it is an
    assertion about the DECISION rather than about winning a race — so it
    cannot flake and cannot pass by luck.

    Why the decision matters: ``pipeline.process_batch`` extracts each document
    in a ``ThreadPoolExecutor``, so every ``run_with_timeout`` call happens on a
    worker thread while sibling workers are mid-extraction. ``fork`` copies the
    address space with sibling-held locks still marked held — by threads that do
    not exist in the child. glibc's malloc survives this via ``pthread_atfork``;
    MuPDF has no such handler, so the child blocked on its first PyMuPDF call
    and the parent waited out the entire deadline, then reported a timeout for a
    document that extracts in ~50ms. Measured on Linux/py3.12: 5 of 25
    `pdfmux audit` runs flagged a good document (~20%). CPython 3.12 warns about
    this exact hazard ("This process is multi-threaded, use of fork() may lead
    to deadlocks in the child").

    ``forkserver`` forks from a single-threaded helper instead, which is why it
    is the only acceptable answer here.
    """
    assert _start_method() in (None, "forkserver"), (
        f"process backend selected {_start_method()!r}. A bare fork() from "
        "pipeline.process_batch's worker threads inherits MuPDF's lock held and "
        "deadlocks the child — see this test's docstring."
    )


@fork_only
def test_concurrent_extraction_does_not_spuriously_time_out(tmp_path) -> None:
    """The end-to-end shape of the bug: real extraction, real thread pool.

    Complements the decision-level guard above by exercising what actually
    deadlocked — PyMuPDF, on worker threads, under isolation. Two trivial
    digital PDFs through ``process_batch``; both MUST extract. Against the fork
    backend this failed ~20% of the time per document.

    The timeout is deliberately generous. These pages extract in milliseconds,
    so the assertion can only fail on a child that never came back — never on a
    merely slow machine.
    """
    import fitz

    from pdfmux.pipeline import process_batch

    paths = []
    for name, body in (("a.pdf", "Alpha chemistry safety."), ("b.pdf", "Beta regulation.")):
        p = tmp_path / name
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), body, fontsize=11)
        doc.save(str(p))
        doc.close()
        paths.append(p)

    for _ in range(5):  # ~10 isolated extractions — catches a ~20% regression ~90%+
        for path, result in process_batch(paths, output_format="markdown", quality="fast"):
            assert not isinstance(result, Exception), (
                f"{path.name} failed under concurrent isolation: {result!r}. A "
                "spurious timeout here means the isolation backend is forking "
                "unsafely from a worker thread."
            )


@fork_only
def test_process_backend_works_without_a_real_main_module() -> None:
    """Isolation must not depend on the host having an importable ``__main__``.

    Non-fork children re-import the parent's main module by default. pdfmux is a
    library: its caller's ``__main__`` may be ``<stdin>`` (REPL, ``python -c``,
    notebook) — where the re-import raises FileNotFoundError in every child — or
    may do real work at import time, which would then run again inside our
    worker, per document. Measured with the fixup left in: constant stderr
    tracebacks and 182 ms/doc against 25 ms/doc.

    Simulates the REPL/`-c` shape by pointing ``__main__.__file__`` at a path
    that does not exist. If the child still needs it, this raises.
    """
    import sys

    main = sys.modules["__main__"]
    sentinel = object()
    saved_file = getattr(main, "__file__", sentinel)
    saved_spec = getattr(main, "__spec__", sentinel)
    main.__file__ = "/nonexistent/<stdin>"
    main.__spec__ = None
    try:
        assert run_with_timeout(_add, (21, 21), 20) == 42
    finally:
        if saved_file is sentinel:
            del main.__file__
        else:
            main.__file__ = saved_file
        if saved_spec is not sentinel:
            main.__spec__ = saved_spec


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
