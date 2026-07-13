"""Adapter interface for pdfmux-bench.

Every engine — local or hosted, open-source or closed API — is wrapped in one
`Adapter` subclass that implements a single method:

    extract(pdf_path: Path) -> str      # returns Markdown

The harness never special-cases an engine. If your tool can turn a PDF into
Markdown, it can be benchmarked here. Closed APIs are first-class citizens: the
whole point of pdfmux-bench is to include the commercial engines that academic
benchmarks omit.

Contract
--------
- `name`        : short id used in the leaderboard (e.g. "reducto").
- `label`       : human-readable name (e.g. "Reducto").
- `homepage`    : URL, for attribution in the leaderboard.
- `license`     : "closed-api", "MIT", "Apache-2.0", ... (informational).
- `env_var`     : the environment variable holding the API key, or None if the
                  engine runs locally with no key.
- `available()` : True if this engine can run right now (key present / package
                  importable). If False, the harness SKIPS the engine with a
                  clear message and NEVER fabricates a result row.
- `extract()`   : do the work. Raise on failure — the harness records the error
                  as data (a failed row), it does not crash the run.

Radical-honesty rule: an adapter must return the engine's *real* output. It must
never fall back to another engine, never synthesize plausible Markdown, and
never swallow an error into empty output. A missing key -> SKIP. A real failure
-> recorded error. A real success -> real Markdown.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class Adapter(ABC):
    name: str = "adapter"
    label: str = "Adapter"
    homepage: str = ""
    license: str = "unknown"
    env_var: str | None = None  # API-key env var, or None for local engines

    #: Human-readable one-liner shown when the engine is skipped.
    def unavailable_reason(self) -> str:
        if self.env_var:
            return (
                f"{self.label} requires the {self.env_var} environment variable "
                f"(no key found). Set it and re-run to include {self.label}."
            )
        return f"{self.label} is not installed / not importable in this environment."

    def available(self) -> bool:
        """Default: available iff the API key env var is set. Local engines that
        depend on an importable package should override this."""
        if self.env_var is None:
            return True
        return bool(os.environ.get(self.env_var))

    @abstractmethod
    def extract(self, pdf_path: Path) -> str:
        """Convert a PDF to Markdown. Raise on failure."""
        raise NotImplementedError

    # Convenience for adapters that call HTTP APIs.
    def _key(self) -> str:
        if not self.env_var:
            raise RuntimeError(f"{self.label} has no env_var configured")
        key = os.environ.get(self.env_var)
        if not key:
            raise RuntimeError(self.unavailable_reason())
        return key
