"""Review manifest: OCR review flags as tracked work, per document.

A conversion's ``review_flags`` say what to check. For an archive, the checking
is work product too: which flags a reviewer cleared, when, and with what note.
The manifest keeps that state in one JSON file next to the documents, keyed by
the PDF's SHA-256, so:

* re-running a folder (after an upgrade, or with new files) reports only flags
  that are new or still open; flags a reviewer resolved stay resolved;
* a flag that no longer appears is marked ``gone``, not deleted, and reopens if
  it comes back;
* every flag in a folder exports as one CSV or JSON file, worst documents first.

Flag ids come from :class:`pdfmux.ocr_review.ReviewFlag` and are stable across
runs for the same reading on the same page.

Schema (version 1)::

    {"version": 1,
     "documents": {"<sha256>": {"path": "...", "last_run": "...", "pdfmux_version": "...",
                                "flags": {"<id>": {<flag fields>, "state": "open|resolved|gone",
                                                    "first_seen": "...", "last_seen": "...",
                                                    "note": "...", "resolved_at": "..."}}}}}
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = ".pdfmux-review.json"
SCHEMA_VERSION = 1
STATES = ("open", "resolved", "gone")
_STATE_KEYS = {"state", "first_seen", "last_seen", "note", "resolved_at"}

EXPORT_COLUMNS = (
    "document",
    "sha256",
    "page",
    "kind",
    "id",
    "state",
    "score",
    "text",
    "reason",
    "line_index",
    "bbox",
    "anchor",
    "first_seen",
    "last_seen",
    "note",
)


class ManifestError(ValueError):
    """The manifest file is unreadable or has an unknown schema."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_manifest_path(targets: Iterable[str | Path]) -> Path:
    """``<folder>/.pdfmux-review.json`` for one folder; next to one file; else the cwd."""
    targets = [Path(t) for t in targets]
    if len(targets) == 1:
        t = targets[0]
        if t.suffix == ".json":
            return t
        return (t if t.is_dir() else t.parent) / MANIFEST_NAME
    return Path.cwd() / MANIFEST_NAME


def new_manifest() -> dict:
    return {"version": SCHEMA_VERSION, "documents": {}}


def load(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return new_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"Cannot read review manifest {path}: {e}") from e
    if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
        raise ManifestError(f"{path} is not a pdfmux review manifest (version {SCHEMA_VERSION})")
    data.setdefault("documents", {})
    return data


def save(path: str | Path, manifest: dict) -> None:
    """Atomic write: a crash mid-save never leaves a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pdfmux-review-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class RunSummary:
    """What one document's run changed in the manifest."""

    document: str
    sha256: str
    new: list[dict] = field(default_factory=list)
    still_open: list[dict] = field(default_factory=list)
    reopened: list[dict] = field(default_factory=list)
    resolved_skipped: int = 0
    gone: list[str] = field(default_factory=list)

    @property
    def to_review(self) -> list[dict]:
        """Flags a reviewer needs to look at after this run."""
        return self.new + self.reopened + self.still_open


def record_run(
    manifest: dict,
    *,
    sha256: str,
    document: str,
    flags: Iterable[dict],
    pdfmux_version: str = "",
    now: str | None = None,
) -> RunSummary:
    """Merge one conversion's ``review_flags`` into the manifest."""
    now = now or _now()
    doc = manifest["documents"].setdefault(sha256, {"flags": {}})
    doc.update({"path": document, "last_run": now, "pdfmux_version": pdfmux_version})
    known: dict = doc.setdefault("flags", {})
    summary = RunSummary(document=document, sha256=sha256)

    current = {}
    for f in flags:
        fid = f.get("id")
        if fid:
            current[fid] = f

    for fid, f in current.items():
        fresh = {k: v for k, v in f.items() if k not in _STATE_KEYS}
        entry = known.get(fid)
        if entry is None:
            entry = {**fresh, "state": "open", "first_seen": now, "last_seen": now}
            known[fid] = entry
            summary.new.append(entry)
            continue
        entry.update(fresh)
        entry["last_seen"] = now
        if entry.get("state") == "gone":
            entry["state"] = "open"
            summary.reopened.append(entry)
        elif entry.get("state") == "resolved":
            summary.resolved_skipped += 1
        else:
            entry["state"] = "open"
            summary.still_open.append(entry)

    for fid, entry in known.items():
        if fid not in current and entry.get("state") == "open":
            entry["state"] = "gone"
            summary.gone.append(fid)
    return summary


def set_state(
    manifest: dict,
    ids: Iterable[str],
    state: str,
    *,
    note: str | None = None,
    now: str | None = None,
) -> tuple[list[str], list[str]]:
    """Set ``state`` on flags by id (a unique id prefix of 6+ characters also works).

    Returns ``(changed, unknown)``.
    """
    if state not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}")
    now = now or _now()
    index: dict[str, list[dict]] = {}
    for doc in manifest["documents"].values():
        for fid, entry in doc.get("flags", {}).items():
            index.setdefault(fid, []).append(entry)

    changed, unknown = [], []
    for raw in ids:
        wanted = raw.strip()
        hits = index.get(wanted)
        if hits is None and len(wanted) >= 6:
            prefixed = [fid for fid in index if fid.startswith(wanted)]
            if len(prefixed) == 1:
                wanted, hits = prefixed[0], index[prefixed[0]]
        if not hits:
            unknown.append(raw)
            continue
        for entry in hits:
            entry["state"] = state
            if note is not None:
                entry["note"] = note
            if state == "resolved":
                entry["resolved_at"] = now
            else:
                entry.pop("resolved_at", None)
        changed.append(wanted)
    return changed, unknown


def rows(manifest: dict, states: Iterable[str] = ("open",)) -> list[dict]:
    """Flat rows for export: worst documents (most selected flags) first."""
    wanted = set(states)
    per_doc = []
    for sha, doc in manifest["documents"].items():
        doc_rows = [
            {
                "document": doc.get("path", ""),
                "sha256": sha,
                **{k: entry.get(k) for k in EXPORT_COLUMNS if k not in ("document", "sha256")},
                "id": fid,
            }
            for fid, entry in doc.get("flags", {}).items()
            if entry.get("state") in wanted
        ]
        doc_rows.sort(key=lambda r: (r.get("page") or 0, str(r.get("kind")), r["id"]))
        per_doc.append((len(doc_rows), doc.get("path", ""), doc_rows))
    per_doc.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, _, doc_rows in per_doc for r in doc_rows]


def to_csv(export_rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(EXPORT_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for r in export_rows:
        out = dict(r)
        for k in ("bbox", "anchor"):
            if out.get(k) is not None:
                out[k] = json.dumps(out[k])
        writer.writerow(out)
    return buf.getvalue()


def to_json(export_rows: list[dict]) -> str:
    return json.dumps(export_rows, indent=2, ensure_ascii=False) + "\n"
