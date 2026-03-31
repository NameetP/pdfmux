"""PDF document cache — open once, reuse everywhere.

Eliminates redundant fitz.open() calls across the pipeline.
A single extraction can trigger 8-14 fitz.open() on the same file.
This module provides a thread-safe cache that opens each PDF once.

Usage:
    from pdfmux.pdf_cache import get_doc, close_all

    doc = get_doc(file_path)  # opens on first call, cached after
    page = doc[0]
    # ... use the page ...
    close_all()  # cleanup at end of pipeline
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict[str, fitz.Document] = {}


def get_doc(file_path: str | Path) -> fitz.Document:
    """Get a fitz.Document handle, opening the file only on first access.

    Thread-safe. The same document handle is returned for repeated calls
    with the same file path.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Open fitz.Document handle. Do NOT close it — use close_all().
    """
    key = str(Path(file_path).resolve())

    with _lock:
        if key in _cache:
            doc = _cache[key]
            if not doc.is_closed:
                return doc
            # Document was closed externally — reopen
            del _cache[key]

        doc = fitz.open(str(file_path))
        _cache[key] = doc
        logger.debug("Opened PDF: %s (%d pages)", Path(file_path).name, len(doc))
        return doc


def close_doc(file_path: str | Path) -> None:
    """Close and remove a specific document from the cache."""
    key = str(Path(file_path).resolve())

    with _lock:
        doc = _cache.pop(key, None)
        if doc and not doc.is_closed:
            doc.close()


def close_all() -> None:
    """Close all cached documents. Call at the end of pipeline processing."""
    with _lock:
        for key, doc in list(_cache.items()):
            if not doc.is_closed:
                doc.close()
        _cache.clear()


def cache_stats() -> dict[str, int]:
    """Return cache statistics for debugging."""
    with _lock:
        return {
            "cached_docs": len(_cache),
            "open_docs": sum(1 for d in _cache.values() if not d.is_closed),
        }
