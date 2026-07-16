"""Hard-terminating timeout for the extraction step.

Extraction calls native code (PyMuPDF, ONNX/RapidOCR, Docling) that can wedge on
a malformed page. A ``ThreadPoolExecutor`` timeout does **not** save you there:
a ``Future`` cannot cancel a thread that is already running, and the executor's
``__exit__`` blocks on ``shutdown(wait=True)`` until the native call returns —
which may be never. So the old "timeout" raised on schedule but then hung the
caller on context-exit anyway. This module gives a timeout that actually returns
control to the caller.

Two backends, chosen by platform:

* **Linux — process isolation.** Run the work in a *forked* child and, on
  timeout, ``SIGTERM`` then ``SIGKILL`` it. The wedged native extractor is truly
  killed and its memory reclaimed. This is the path the cloud worker (Linux
  Docker) runs, where an un-killable hang is most costly (a stuck paid job).

* **macOS / Windows — daemon thread.** Forking *after* fitz / onnxruntime have
  loaded native frameworks is unsafe on Darwin, and Windows has no ``fork`` at
  all, so we fall back to a daemon thread. The caller is still freed the instant
  the deadline passes and process exit is never blocked (daemon threads are not
  joined at interpreter shutdown), but a genuinely-wedged native call leaks until
  the process ends. That is a CPython limitation — a running thread cannot be
  force-killed — not a defect in this code.

Override the choice with ``PDFMUX_TIMEOUT_ISOLATION``:

* ``auto`` (default) — process on Linux, thread elsewhere.
* ``process`` — force the forked-child backend on any platform that has ``fork``
  (used by the test-suite; only safe for fork-safe payloads).
* ``thread`` — force the daemon-thread backend everywhere.
* ``off`` — no isolation; run inline (mainly for debugging).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

# SIGTERM grace before escalating to SIGKILL, and the reap window after a clean
# finish. Small, fixed — a cooperative child exits well within these.
_TERM_GRACE_S = 3.0
_REAP_S = 5.0
# Poll granularity while waiting on the child: bounds how long a silent crash
# (segfault with no result on the queue) can masquerade as "still running".
_POLL_S = 0.5


class HardTimeoutError(Exception):
    """The work exceeded its deadline (or the isolated worker died without a
    result). Callers map this to their own domain timeout error."""


def _isolation_mode() -> str:
    mode = os.environ.get("PDFMUX_TIMEOUT_ISOLATION", "auto").strip().lower()
    return mode if mode in ("auto", "process", "thread", "off") else "auto"


def _fork_available() -> bool:
    try:
        import multiprocessing as mp

        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


def _auto_prefers_process() -> bool:
    # fork is the native, fork-safe idiom on Linux. On Darwin, forking after
    # native frameworks are loaded can hard-crash, so auto stays on threads.
    return sys.platform.startswith("linux") and _fork_available()


def run_with_timeout(func: Callable[..., Any], args: tuple, timeout_s: float) -> Any:
    """Run ``func(*args)`` under a hard deadline.

    Returns the function's result. Re-raises any exception the function raised.
    Raises :class:`HardTimeoutError` if the deadline passes before the work finishes
    (the isolated worker is terminated first, on the process backend).
    """
    mode = _isolation_mode()
    if mode == "off":
        return func(*args)
    if mode == "process" or (mode == "auto" and _auto_prefers_process()):
        if _fork_available():
            return _run_in_process(func, args, timeout_s)
        # forced process but no fork (e.g. Windows) — degrade, don't fail
    return _run_in_thread(func, args, timeout_s)


# ---------------------------------------------------------------------------
# Process backend (fork + SIGTERM/SIGKILL) — the real hard timeout
# ---------------------------------------------------------------------------


def _process_target(func: Callable[..., Any], args: tuple, q: Any) -> None:
    """Child entrypoint: run the work, ferry (status, payload) to the parent."""
    try:
        payload: tuple[str, Any] = ("ok", func(*args))
    except BaseException as exc:  # noqa: BLE001 — ferry *any* failure to the parent
        payload = ("err", _picklable_exc(exc))
    try:
        q.put(payload)
    except Exception:
        # Result couldn't be serialized/sent — don't let the parent wait out the
        # full timeout for nothing.
        try:
            q.put(("err", RuntimeError("extraction produced an unreturnable result")))
        except Exception:
            pass


def _picklable_exc(exc: BaseException) -> BaseException:
    import pickle

    try:
        pickle.dumps(exc)
        return exc
    except Exception:
        return RuntimeError(f"{type(exc).__name__}: {exc}")


def _run_in_process(func: Callable[..., Any], args: tuple, timeout_s: float) -> Any:
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    q: Any = ctx.Queue()
    proc = ctx.Process(target=_process_target, args=(func, args, q), daemon=True)
    proc.start()
    try:
        # Drain the result BEFORE joining: mp.Queue can deadlock if you join a
        # child that is still flushing a large payload through the feeder thread.
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(proc)
                raise HardTimeoutError()
            try:
                status, payload = q.get(timeout=min(remaining, _POLL_S))
                break
            except queue.Empty:
                if not proc.is_alive():
                    # Child exited without a result: it crashed or was killed.
                    # One last non-blocking read in case of a put/exit race.
                    try:
                        status, payload = q.get_nowait()
                        break
                    except queue.Empty:
                        raise HardTimeoutError()
                # still running — keep waiting
        proc.join(timeout=_REAP_S)
        if proc.is_alive():
            _terminate(proc)
        if status == "err":
            raise payload
        return payload
    finally:
        if proc.is_alive():
            _terminate(proc)
        try:
            q.close()
            q.cancel_join_thread()
        except Exception:
            pass


def _terminate(proc: Any) -> None:
    """SIGTERM, a brief grace, then SIGKILL — so a child that ignores SIGTERM
    (or is stuck in native code) is still guaranteed to die."""
    if not proc.is_alive():
        return
    proc.terminate()  # SIGTERM
    proc.join(timeout=_TERM_GRACE_S)
    if proc.is_alive():
        proc.kill()  # SIGKILL
        proc.join(timeout=_TERM_GRACE_S)


# ---------------------------------------------------------------------------
# Thread backend (daemon thread) — non-blocking fallback where fork is unsafe
# ---------------------------------------------------------------------------


def _run_in_thread(func: Callable[..., Any], args: tuple, timeout_s: float) -> Any:
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["ok"] = func(*args)
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        # A CPython thread cannot be force-killed. It's a daemon, so it won't
        # block process exit and dies with the interpreter. Free the caller now.
        raise HardTimeoutError()
    if "err" in box:
        raise box["err"]
    return box.get("ok")
