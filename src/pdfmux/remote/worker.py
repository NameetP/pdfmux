"""Sandboxed parse step: ``python -m pdfmux.remote.worker <file.pdf> [max_pages]``.

Reads one PDF, prints the extract_tables result as JSON on stdout, exits. It has no network
access in production (run inside ``docker run --network none``, see sandbox.py) and never sees a
URL: the front end downloads, the worker only parses. A crash or hang here kills only this
process.
"""

from __future__ import annotations

import json
import sys

from pdfmux.remote.tables import extract_tables, result_to_dict


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(json.dumps({"error": "usage"}))
        return 2
    max_pages = int(argv[2]) if len(argv) > 2 else 30
    try:
        result = result_to_dict(extract_tables(argv[1], max_pages=max_pages))
    except ValueError as e:  # e.g. password-protected
        print(json.dumps({"error": "unreadable", "detail": str(e)[:200]}))
        return 3
    except Exception:  # noqa: BLE001 — any parser failure is a clean "couldn't read", never a trace
        print(json.dumps({"error": "unreadable"}))
        return 3
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
