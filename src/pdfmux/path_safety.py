"""Path-safety helpers — shared between mcp_server and schema loaders.

Lifted out of ``mcp_server`` so that ``schema.py`` can reuse the same
allowed-directories check without creating a circular import (mcp_server
imports from pipeline → pipeline → schema).

The single source of truth for ``PDFMUX_ALLOWED_DIRS`` lives here. Modules
that previously imported ``ALLOWED_DIRS`` / ``_is_path_allowed`` /
``_check_path`` from ``mcp_server`` continue to work — those names are
re-exported there for backward compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowed-directories sandbox
# ---------------------------------------------------------------------------

_ALLOWED_DIRS_ENV = os.environ.get("PDFMUX_ALLOWED_DIRS", "")
ALLOWED_DIRS: list[Path] = (
    [Path(d).resolve() for d in _ALLOWED_DIRS_ENV.split(":") if d.strip()]
    if _ALLOWED_DIRS_ENV
    else [Path.cwd().resolve()]
)


def is_path_allowed(file_path: Path) -> bool:
    """Check if the given path is inside one of the allowed directories."""
    resolved = file_path.resolve()
    return any(resolved == d or d in resolved.parents for d in ALLOWED_DIRS)


def check_path(file_path: str, label: str = "file_path") -> Path:
    """Validate and return a resolved Path, raising ValueError on access denial."""
    if not file_path:
        raise ValueError(f"{label} is required")
    p = Path(file_path)
    if not is_path_allowed(p):
        raise ValueError(
            f"Access denied: {file_path} is outside allowed directories. "
            "Set PDFMUX_ALLOWED_DIRS to configure access."
        )
    return p
