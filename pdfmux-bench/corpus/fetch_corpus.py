#!/usr/bin/env python3
"""Download the pdfmux-bench corpus from manifest.json.

Reads corpus/manifest.json and downloads each document whose `url` is set and
`needs_pin` is false into `corpus/<category>/<id>.pdf`. Verifies `sha256` when
present. Records marked `needs_pin: true` (real source named, exact URL not yet
finalized) are reported and skipped — never fabricated.

Stdlib only (urllib) so the corpus fetch has no dependencies.

Usage:
    python corpus/fetch_corpus.py                 # fetch all pinned docs
    python corpus/fetch_corpus.py --category rtl  # one category
    python corpus/fetch_corpus.py --list          # show manifest status, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "manifest.json"
USER_AGENT = "pdfmux-bench-corpus-fetcher/1.0 (+https://github.com/NameetP/pdfmux)"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def fetch_one(doc: dict) -> tuple[str, str]:
    """Return (status, detail). status in {ok, skip, error, exists}."""
    doc_id = doc["id"]
    category = doc["category"]
    url = doc.get("url")
    if doc.get("needs_pin") or not url:
        return "skip", f"needs_pin — pin a URL from: {doc.get('source', '?')}"

    dest_dir = HERE / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc_id}.pdf"
    if dest.exists():
        return "exists", str(dest)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted manifest URLs)
            data = resp.read()
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {e}"

    expected = doc.get("sha256")
    if expected:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            return "error", f"sha256 mismatch (expected {expected[:12]}…, got {actual[:12]}…)"

    dest.write_bytes(data)
    return "ok", f"{len(data):,} bytes -> {dest}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", help="only this category")
    p.add_argument("--list", action="store_true", help="print status, download nothing")
    args = p.parse_args(argv)

    manifest = load_manifest()
    docs = manifest["documents"]
    if args.category:
        docs = [d for d in docs if d["category"] == args.category]

    if args.list:
        print(f"{'id':38s} {'category':16s} {'gt':8s} {'pinned':7s} url")
        for d in docs:
            pinned = "no" if (d.get("needs_pin") or not d.get("url")) else "yes"
            print(f"{d['id']:38s} {d['category']:16s} {d.get('gt_status','?'):8s} {pinned:7s} {d.get('url') or '(needs pin)'}")
        return 0

    counts = {"ok": 0, "exists": 0, "skip": 0, "error": 0}
    for d in docs:
        status, detail = fetch_one(d)
        counts[status] += 1
        icon = {"ok": "✓", "exists": "•", "skip": "–", "error": "✗"}[status]
        print(f"{icon} {d['id']:38s} {status:7s} {detail}")

    print(
        f"\nDone: {counts['ok']} fetched, {counts['exists']} already present, "
        f"{counts['skip']} need pinning, {counts['error']} errors."
    )
    if counts["skip"]:
        print("Records that need pinning name a real source — finalize the exact URL + sha256 and set needs_pin=false. See corpus/README.md.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
