"""Extractor protocol + registry.

Every extractor implements the same protocol: yield PageResult objects
one page at a time. The registry tracks what's available and orders
them by priority for fallback chains.

Usage:
    from pdfmux.extractors import get_extractor, available_extractors

    ext = get_extractor("fast")
    for page in ext.extract("report.pdf"):
        print(page.page_num, page.char_count)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from pdfmux.types import PageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Extractor(Protocol):
    """Protocol every extractor must satisfy.

    Extractors yield PageResult objects — one per page, streamed.
    The pipeline consumes the iterator so memory stays bounded.
    """

    @property
    def name(self) -> str:
        """Human-readable name (e.g. 'pymupdf4llm')."""
        ...

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page, in page order.

        Args:
            file_path: Path to the PDF.
            pages: Optional 0-indexed page list. None → all pages.

        Yields:
            PageResult for each page.
        """
        ...

    def available(self) -> bool:
        """Return True if this extractor's dependencies are installed."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# (priority, name, factory_function)
# Lower priority number = tried first.
_REGISTRY: list[tuple[int, str, type]] = []


def register(name: str, priority: int):
    """Decorator to register an extractor class.

    Args:
        name: Short name like "fast", "rapidocr", "docling".
        priority: Order in fallback chain. Lower = tried first.
                  fast=10, rapidocr=20, surya=30, docling=40, llm=50.
    """

    def decorator(cls: type) -> type:
        _REGISTRY.append((priority, name, cls))
        _REGISTRY.sort(key=lambda t: t[0])  # keep sorted
        return cls

    return decorator


def get_extractor(name: str) -> Extractor:
    """Get a specific extractor by name.

    Raises:
        KeyError: If no extractor with that name is registered.
        pdfmux.errors.ExtractorNotAvailable: If deps aren't installed.
    """
    for _, reg_name, cls in _REGISTRY:
        if reg_name == name:
            instance = cls()
            if not instance.available():
                from pdfmux.errors import ExtractorNotAvailable

                raise ExtractorNotAvailable(
                    f"Extractor '{name}' is registered but its dependencies are not installed."
                )
            return instance
    raise KeyError(f"No extractor registered with name '{name}'")


def available_extractors() -> list[tuple[str, Extractor]]:
    """Return all installed extractors in priority order.

    Skips extractors whose dependencies are missing.

    Returns:
        List of (name, instance) tuples, sorted by priority.
    """
    result = []
    for _, name, cls in _REGISTRY:
        try:
            instance = cls()
            if instance.available():
                result.append((name, instance))
        except Exception:
            continue
    return result


def extractor_names() -> list[str]:
    """Return names of all registered extractors (installed or not)."""
    return [name for _, name, _ in _REGISTRY]
