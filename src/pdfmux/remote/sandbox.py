"""Run the parse step in isolation.

Two modes, chosen by ``PDFMUX_SANDBOX``:

* ``docker`` (production): ``docker run --rm --network none --read-only`` with memory, CPU and
  pid limits, the PDF bind-mounted read-only. A malicious PDF that exploits the parser gets a
  container with no network, no writable filesystem and a hard kill at the timeout.
* ``process`` (default; dev and tests): a child Python process with RLIMIT_AS / RLIMIT_CPU /
  RLIMIT_NOFILE / RLIMIT_FSIZE and a wall-clock kill. Same contract, weaker isolation.

Either way: SIGALRM inside the parser can't interrupt C code (PyMuPDF), so the timeout is
enforced from the OUTSIDE by killing the process or container.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 90
MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GiB
DOCKER_IMAGE = os.environ.get("PDFMUX_SANDBOX_IMAGE", "pdfmux-sandbox:latest")


class SandboxError(RuntimeError):
    """The parse failed, timed out or produced no usable output. Message is safe to show."""


@dataclass
class SandboxResult:
    data: dict[str, Any]


def _limits() -> None:  # runs in the child before exec (process mode only)
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_TIMEOUT_S, DEFAULT_TIMEOUT_S + 5))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES * 2, MEMORY_BYTES * 2))
    except (ValueError, OSError):
        pass  # macOS rejects RLIMIT_AS; the wall-clock kill still applies
    os.setsid()


def _command(pdf: Path, max_pages: int) -> tuple[list[str], dict[str, Any]]:
    mode = os.environ.get("PDFMUX_SANDBOX", "process")
    if mode == "docker":
        name = f"pdfmux-parse-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=64m,noexec",
            "--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--pids-limit", "64",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "--user", "65534:65534",
            "-v", f"{pdf.parent}:/in:ro",
            DOCKER_IMAGE,
            "python", "-m", "pdfmux.remote.worker", f"/in/{pdf.name}", str(max_pages),
        ]  # fmt: skip
        return cmd, {"container": name}
    return [sys.executable, "-m", "pdfmux.remote.worker", str(pdf), str(max_pages)], {}


def run_parse(pdf: Path, max_pages: int = 30, timeout_s: int = DEFAULT_TIMEOUT_S) -> SandboxResult:
    cmd, ctx = _command(pdf, max_pages)
    process_mode = "container" not in ctx
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd,
            capture_output=True,
            timeout=timeout_s,
            preexec_fn=_limits if process_mode else None,  # noqa: PLW1509
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
            if process_mode
            else None,
        )
    except subprocess.TimeoutExpired:
        if not process_mode:
            subprocess.run(["docker", "kill", ctx["container"]], capture_output=True, timeout=15)  # noqa: S603,S607
        raise SandboxError("This PDF took too long to read.") from None
    try:
        data = json.loads(proc.stdout.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SandboxError("This PDF couldn't be read.") from None
    if "error" in data:
        if data.get("error") == "unreadable" and "password" in str(data.get("detail", "")).lower():
            raise SandboxError("This PDF is password-protected.")
        raise SandboxError("This PDF couldn't be read.")
    return SandboxResult(data=data)


def describe() -> str:
    """For logs and the conformance checklist: which isolation is actually in force."""
    cmd, _ = _command(Path("/tmp/x.pdf"), 1)
    return shlex.join(cmd[:8]) + " …"
