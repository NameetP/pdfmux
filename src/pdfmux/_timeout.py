"""Hard-terminating timeout for the extraction step.

Extraction calls native code (PyMuPDF, ONNX/RapidOCR, Docling) that can wedge on
a malformed page. A ``ThreadPoolExecutor`` timeout does **not** save you there:
a ``Future`` cannot cancel a thread that is already running, and the executor's
``__exit__`` blocks on ``shutdown(wait=True)`` until the native call returns —
which may be never. So the old "timeout" raised on schedule but then hung the
caller on context-exit anyway. This module gives a timeout that actually returns
control to the caller.

Two backends, chosen by platform:

* **Linux — process isolation via forkserver.** Run the work in a child and, on
  timeout, ``SIGTERM`` then ``SIGKILL`` it. The wedged native extractor is truly
  killed and its memory reclaimed. This is the path the cloud worker (Linux
  Docker) runs, where an un-killable hang is most costly (a stuck paid job).

  The child comes from a ``forkserver``, never a bare ``fork``, and that is not
  a style preference. :func:`pdfmux.pipeline.process_batch` extracts documents
  concurrently in a ``ThreadPoolExecutor``, so isolation is always entered FROM
  a worker thread with live siblings. ``fork`` there copies the address space
  with sibling-held locks marked held by threads absent from the child; MuPDF
  has no ``pthread_atfork`` handler, so the child blocks on its first PyMuPDF
  call forever and the parent reports a timeout for a document that extracts in
  milliseconds. That shipped in 1.8.x and cost ~20% of documents in concurrent
  runs. See :func:`_start_method`.

* **macOS / Windows — daemon thread.** Forking *after* fitz / onnxruntime have
  loaded native frameworks is unsafe on Darwin, and Windows has no ``fork`` at
  all, so we fall back to a daemon thread. The caller is still freed the instant
  the deadline passes and process exit is never blocked (daemon threads are not
  joined at interpreter shutdown), but a genuinely-wedged native call leaks until
  the process ends. That is a CPython limitation — a running thread cannot be
  force-killed — not a defect in this code.

Override the choice with ``PDFMUX_TIMEOUT_ISOLATION``:

* ``auto`` (default) — process on Linux, thread elsewhere.
* ``process`` — force the isolated-child backend anywhere a forkserver exists.
* ``thread`` — force the daemon-thread backend everywhere.
* ``off`` — no isolation; run inline (mainly for debugging).
"""

from __future__ import annotations

import contextlib
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


def _start_method() -> str | None:
    """The safest available start method for the process backend, or None.

    ``forkserver`` — not ``fork`` — is the correct choice here, and the reason is
    not theoretical. :func:`pdfmux.pipeline.process_batch` extracts documents
    concurrently in a ``ThreadPoolExecutor``, so every ``run_with_timeout`` call
    happens ON a worker thread while sibling workers are mid-extraction. A bare
    ``fork`` copies the address space with whatever locks those siblings hold
    still marked held — by threads that do not exist in the child. glibc's malloc
    survives this via ``pthread_atfork``; MuPDF has no such handler, so the child
    blocks on its first PyMuPDF call, forever. The parent then waits out the
    entire deadline and reports a timeout for a document that extracts in ~50ms.

    Measured on Linux/py3.12 before this change: 5 of 25 `pdfmux audit` runs over
    two trivial digital PDFs flagged a document with a bogus OCRTimeoutError —
    ~20%, and the CLI blamed the user's extractor for the disagreement. With
    ``forkserver`` (children forked from a single-threaded helper, not from the
    live worker thread) the same loop is 0 of 100.

    ``fork`` is deliberately NOT used as a fallback: it is the bug.
    """
    try:
        import multiprocessing as mp

        available = mp.get_all_start_methods()
    except Exception:
        return None
    if "forkserver" in available:
        return "forkserver"
    return None


def _auto_prefers_process() -> bool:
    # Process isolation is the only backend that can actually kill a wedged
    # native extractor, so it is preferred wherever forkserver exists (Linux —
    # including the cloud worker's Docker image, where an un-killable hang is
    # most costly). Darwin has forkserver too, but forking after CoreFoundation
    # frameworks are loaded remains unsafe there, so auto stays on threads.
    return sys.platform.startswith("linux") and _start_method() is not None


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
        if _start_method() is not None:
            return _run_in_process(func, args, timeout_s)
        # forced process but no forkserver (e.g. Windows) — degrade, don't fail
    return _run_in_thread(func, args, timeout_s)


# ---------------------------------------------------------------------------
# Process backend (fork + SIGTERM/SIGKILL) — the real hard timeout
# ---------------------------------------------------------------------------


_preload_done = False
_preload_lock = threading.Lock()
# Serializes child creation. Our callers ARE concurrent worker threads, and the
# main-module suppression below mutates process-global state for the duration.
_spawn_lock = threading.RLock()


@contextlib.contextmanager
def _no_main_module_fixup() -> Any:
    """Stop the forkserver child from re-importing the host's ``__main__``.

    ``multiprocessing.spawn.get_preparation_data`` tells every non-fork child to
    re-import the parent's main module so that ``__main__``-defined pickles
    resolve. pdfmux never needs that — the only thing we send across is
    ``pdfmux.pipeline._route_and_extract``, importable by name — and for a
    *library* the fixup is actively harmful in two ways:

    * It breaks outright when there is no real main file. Under ``python -c``,
      a REPL, a notebook, or an embedded interpreter the path is a placeholder
      like ``<stdin>`` and every child dies with FileNotFoundError. Measured:
      constant stderr tracebacks and 182 ms/doc instead of 25 ms/doc, because
      each dead child had to be replaced.
    * Even when it works, it executes somebody else's module inside our worker.
      A host app whose ``__main__`` does anything at import time gets it done
      again, per child, as a side effect of extracting a PDF.

    Clearing ``__file__``/``__spec__`` for the moment of ``Process.start()``
    makes ``get_preparation_data`` emit neither ``init_main_from_path`` nor
    ``init_main_from_name``, so the child imports only what it is told to
    preload. Held under ``_spawn_lock`` because this is global state.
    """
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    sentinel = object()
    saved_file = getattr(main, "__file__", sentinel)
    saved_spec = getattr(main, "__spec__", sentinel)
    try:
        if saved_file is not sentinel:
            del main.__file__
        main.__spec__ = None
        yield
    finally:
        if saved_file is not sentinel:
            main.__file__ = saved_file
        if saved_spec is not sentinel:
            main.__spec__ = saved_spec
        else:  # pragma: no cover — __main__ always has __spec__ in practice
            del main.__spec__


def _preload_forkserver(ctx: Any, method: str) -> None:
    """Warm the forkserver once per process. Best-effort by design."""
    global _preload_done
    if method != "forkserver" or _preload_done:
        return
    with _preload_lock:
        if _preload_done:
            return
        try:
            ctx.set_forkserver_preload(["pdfmux.pipeline"])
        except Exception:
            # A preload failure only costs startup time, never correctness —
            # the child imports what it needs on its own.
            pass
        _preload_done = True


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

    method = _start_method()
    if method is None:  # pragma: no cover — guarded by the caller
        return _run_in_thread(func, args, timeout_s)
    ctx = mp.get_context(method)
    # The forkserver child re-imports rather than inheriting, so hand it the
    # heavy modules up front — one import in the server, inherited by every
    # child after it. Without this each call pays a cold `import fitz`.
    _preload_forkserver(ctx, method)
    with _spawn_lock, _no_main_module_fixup():
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
