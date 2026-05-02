"""Smart result cache — keyed by (file_hash, quality, format, schema).

Avoids re-processing the same PDF when nothing has changed. The cache key
includes the SHA-256 of the file contents plus the extraction parameters,
so any change to the file or to the extraction request produces a fresh
result.

Cache layout::

    ~/.cache/pdfmux/results/
        <hash>__<quality>__<format>__<schema_hash>.json   ← entries
        index.json                                        ← LRU + metadata

Configuration via environment variables:

    PDFMUX_CACHE_DIR     — override the cache directory path
    PDFMUX_CACHE_TTL     — TTL in seconds (default: 30 days)
    PDFMUX_CACHE_MAX_MB  — max total cache size in MB (default: 1024)
    PDFMUX_NO_CACHE      — when set, ResultCache.enabled is False

The cache survives process restarts. File hashes are themselves cached on
the ``pdfmux.pdf_cache`` document handle to avoid re-hashing the same PDF
multiple times within one process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- Defaults ---------------------------------------------------------------

_DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
_DEFAULT_MAX_MB = 1024  # 1 GB

_HASH_BLOCK_SIZE = 1024 * 1024  # 1 MB read buffer

# In-process map of resolved-path -> sha256 (mirrors pdf_cache lifetime)
_HASH_LOCK = threading.Lock()
_HASH_BY_PATH: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Return the default cache directory.

    Honours ``PDFMUX_CACHE_DIR`` if set, otherwise uses
    ``~/.cache/pdfmux/results``.
    """
    env = os.environ.get("PDFMUX_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "pdfmux" / "results"


def _ttl_seconds() -> int:
    """Effective TTL (seconds), read from ``PDFMUX_CACHE_TTL``."""
    raw = os.environ.get("PDFMUX_CACHE_TTL")
    if not raw:
        return _DEFAULT_TTL_SECONDS
    try:
        val = int(raw)
        return max(0, val)
    except ValueError:
        return _DEFAULT_TTL_SECONDS


def _max_bytes() -> int:
    """Effective max cache size in bytes, read from ``PDFMUX_CACHE_MAX_MB``."""
    raw = os.environ.get("PDFMUX_CACHE_MAX_MB")
    if not raw:
        return _DEFAULT_MAX_MB * 1024 * 1024
    try:
        val = int(raw)
        return max(0, val) * 1024 * 1024
    except ValueError:
        return _DEFAULT_MAX_MB * 1024 * 1024


def file_hash(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents.

    Memoised per-process by the resolved path. The cached digest is also
    invalidated automatically if the path's mtime/size change between
    calls (cheap stat check).
    """
    p = Path(file_path).resolve()
    key = str(p)
    try:
        st = p.stat()
        sig = (st.st_size, int(st.st_mtime_ns))
    except OSError:
        sig = None

    with _HASH_LOCK:
        cached = _HASH_BY_PATH.get(key)
        sig_key = key + "::sig"
        cached_sig = _HASH_BY_PATH.get(sig_key)
        if cached and cached_sig == repr(sig):
            return cached

    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(_HASH_BLOCK_SIZE)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()

    with _HASH_LOCK:
        _HASH_BY_PATH[key] = digest
        _HASH_BY_PATH[key + "::sig"] = repr(sig)

    return digest


def _hash_str(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalise_schema(schema: str | None) -> str:
    """Return a stable cache token for the schema parameter.

    For file-path schemas we hash the file contents so that updating the
    schema busts the cache. For preset names we use the literal string.
    Falsy values map to ``"none"``.
    """
    if not schema:
        return "none"
    p = Path(schema).expanduser()
    try:
        if p.is_file():
            return "file:" + _hash_str(p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass
    return "preset:" + schema


def _build_key(
    *,
    pdf_hash: str,
    quality: str,
    output_format: str,
    schema: str | None,
) -> tuple[str, str]:
    """Return (filename, composite_key_str) for an entry."""
    schema_token = _normalise_schema(schema)
    composite = f"{pdf_hash}|{quality}|{output_format}|{schema_token}"
    short_schema = _hash_str(schema_token)[:8]
    fname = f"{pdf_hash}__{quality}__{output_format}__{short_schema}.json"
    return fname, composite


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Serialise a ConversionResult (or compatible object) to a JSON dict.

    Handles nested dataclasses (e.g. PDFClassification) by recursing through
    ``asdict`` semantics.
    """
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"Cannot serialise result of type {type(result)!r}")


def _dict_to_result(data: dict[str, Any]) -> Any:
    """Inverse of :func:`_result_to_dict` — rebuild a ``ConversionResult``."""
    # Imports are local to avoid circular imports at module load.
    from pdfmux.detect import PDFClassification
    from pdfmux.pipeline import ConversionResult

    classification_data = data.get("classification") or {}

    # PDFClassification may have evolved; tolerate unknown keys.
    try:
        valid_keys = set(PDFClassification.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    except AttributeError:
        valid_keys = set(classification_data.keys())
    classification = PDFClassification(
        **{k: v for k, v in classification_data.items() if k in valid_keys}
    )

    return ConversionResult(
        text=data["text"],
        format=data["format"],
        confidence=float(data["confidence"]),
        extractor_used=data["extractor_used"],
        page_count=int(data["page_count"]),
        warnings=list(data.get("warnings") or []),
        classification=classification,
        ocr_pages=list(data.get("ocr_pages") or []),
    )


# ---------------------------------------------------------------------------
# Cache entry / index
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    filename: str
    composite_key: str
    pdf_hash: str
    quality: str
    output_format: str
    schema_token: str
    size_bytes: int
    created_at: float
    last_used_at: float
    hits: int = 0


# ---------------------------------------------------------------------------
# ResultCache
# ---------------------------------------------------------------------------


class ResultCache:
    """LRU result cache keyed by ``(file_hash, quality, format, schema)``.

    The cache is process- and thread-safe (a single ``RLock`` serialises
    index updates and disk writes). Reads only take the lock briefly.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        ttl_seconds: int | None = None,
        max_bytes: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else _ttl_seconds()
        self.max_bytes = max_bytes if max_bytes is not None else _max_bytes()
        if enabled is None:
            enabled = not bool(os.environ.get("PDFMUX_NO_CACHE"))
        self.enabled = bool(enabled)
        self._index_path = self.cache_dir / "index.json"
        self._lock = threading.RLock()
        self._index: dict[str, _Entry] = {}

        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._load_index()
            except OSError as e:
                logger.warning("ResultCache disabled — %s", e)
                self.enabled = False

    # ----- index persistence -------------------------------------------------

    def _load_index(self) -> None:
        if not self._index_path.exists():
            self._index = {}
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            entries = raw.get("entries") or {}
            self._index = {
                key: _Entry(**vals) for key, vals in entries.items() if isinstance(vals, dict)
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Cache index unreadable, resetting: %s", e)
            self._index = {}

    def _save_index(self) -> None:
        if not self.enabled:
            return
        payload = {
            "version": 1,
            "entries": {k: asdict(v) for k, v in self._index.items()},
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            tmp.replace(self._index_path)
        except OSError as e:
            logger.warning("Failed to write cache index: %s", e)

    # ----- public API --------------------------------------------------------

    def get(
        self,
        file_path: str | Path,
        quality: str,
        output_format: str,
        schema: str | None = None,
    ) -> Any | None:
        """Return a cached result or ``None`` on miss."""
        if not self.enabled:
            return None
        try:
            pdf_hash = file_hash(file_path)
        except OSError:
            return None
        fname, composite = _build_key(
            pdf_hash=pdf_hash,
            quality=quality,
            output_format=output_format,
            schema=schema,
        )

        with self._lock:
            entry = self._index.get(composite)
            if entry is None:
                return None

            # TTL check
            now = time.time()
            if self.ttl_seconds > 0 and (now - entry.created_at) > self.ttl_seconds:
                self._evict(composite)
                self._save_index()
                return None

            entry_path = self.cache_dir / entry.filename
            if not entry_path.exists():
                self._index.pop(composite, None)
                self._save_index()
                return None

            try:
                data = json.loads(entry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._evict(composite)
                self._save_index()
                return None

            entry.last_used_at = now
            entry.hits += 1
            self._save_index()

        try:
            return _dict_to_result(data)
        except (KeyError, TypeError) as e:
            logger.warning("Cache entry corrupt, evicting: %s", e)
            with self._lock:
                self._evict(composite)
                self._save_index()
            return None

    def put(
        self,
        file_path: str | Path,
        quality: str,
        output_format: str,
        schema: str | None,
        result: Any,
    ) -> None:
        """Store a result. Silent no-op if the cache is disabled or write fails."""
        if not self.enabled:
            return
        try:
            pdf_hash = file_hash(file_path)
        except OSError:
            return
        fname, composite = _build_key(
            pdf_hash=pdf_hash,
            quality=quality,
            output_format=output_format,
            schema=schema,
        )

        try:
            payload = _result_to_dict(result)
        except TypeError as e:
            logger.debug("ResultCache: cannot serialise result: %s", e)
            return

        entry_path = self.cache_dir / fname
        try:
            entry_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            size_bytes = entry_path.stat().st_size
        except OSError as e:
            logger.warning("ResultCache write failed: %s", e)
            return

        now = time.time()
        with self._lock:
            self._index[composite] = _Entry(
                filename=fname,
                composite_key=composite,
                pdf_hash=pdf_hash,
                quality=quality,
                output_format=output_format,
                schema_token=_normalise_schema(schema),
                size_bytes=size_bytes,
                created_at=now,
                last_used_at=now,
                hits=0,
            )
            self._enforce_size_limit()
            self._save_index()

    def clear(self) -> int:
        """Remove every cache entry. Returns the count removed."""
        if not self.cache_dir.exists():
            return 0
        removed = 0
        with self._lock:
            for entry in list(self._index.values()):
                p = self.cache_dir / entry.filename
                try:
                    if p.exists():
                        p.unlink()
                        removed += 1
                except OSError:
                    pass
            self._index.clear()
            # Also sweep any orphan .json files (other than the index).
            try:
                for p in self.cache_dir.glob("*.json"):
                    if p.name == "index.json":
                        continue
                    try:
                        p.unlink()
                        removed += 1
                    except OSError:
                        pass
            except OSError:
                pass
            self._save_index()
        return removed

    def stats(self) -> dict[str, Any]:
        """Return counts, total bytes, hit summary."""
        with self._lock:
            entries = list(self._index.values())
        total_bytes = sum(e.size_bytes for e in entries)
        total_hits = sum(e.hits for e in entries)
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "entries": len(entries),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 3),
            "max_mb": round(self.max_bytes / (1024 * 1024), 3),
            "ttl_seconds": self.ttl_seconds,
            "hits": total_hits,
        }

    # ----- internal eviction ------------------------------------------------

    def _evict(self, composite_key: str) -> None:
        entry = self._index.pop(composite_key, None)
        if entry is None:
            return
        p = self.cache_dir / entry.filename
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    def _enforce_size_limit(self) -> None:
        if self.max_bytes <= 0:
            return
        total = sum(e.size_bytes for e in self._index.values())
        if total <= self.max_bytes:
            return
        # LRU eviction: oldest last_used_at first.
        # Always keep the most recently used entry, even if it exceeds the
        # cap on its own — otherwise a freshly-written value can be evicted
        # immediately, defeating the purpose of the cache.
        ordered = sorted(self._index.values(), key=lambda e: e.last_used_at)
        if not ordered:
            return
        most_recent = ordered[-1]
        for entry in ordered[:-1]:
            if total <= self.max_bytes:
                break
            self._evict(entry.composite_key)
            total -= entry.size_bytes
        # If still over budget, the only remaining entry is `most_recent`.
        # Leave it in place — the cap is best-effort, not a hard wall.
        _ = most_recent


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_default_cache: ResultCache | None = None
_default_lock = threading.Lock()


def get_default_cache() -> ResultCache:
    """Return the lazily-initialised process-wide ResultCache."""
    global _default_cache
    with _default_lock:
        if _default_cache is None:
            _default_cache = ResultCache()
        return _default_cache


def reset_default_cache() -> None:
    """Drop the cached singleton (used by tests)."""
    global _default_cache
    with _default_lock:
        _default_cache = None
