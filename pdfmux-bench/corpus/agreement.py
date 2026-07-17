#!/usr/bin/env python3
"""Recompute GT-0 inter-annotator agreement from the two committed annotation passes.

Reads annotations/pass1/<id>.gt.md and annotations/pass2/<id>.gt.md and prints
per-dimension agreement, using the benchmark's own scoring.py metrics.

Usage:
    python corpus/agreement.py            # from pdfmux-bench/
    python agreement.py                   # from pdfmux-bench/corpus/
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # pdfmux-bench/ for scoring.py
import scoring  # noqa: E402

PASS1 = HERE / "annotations" / "pass1"
PASS2 = HERE / "annotations" / "pass2"

DOCS = {
    "irs-fw9": "forms",
    "arxiv-1512.03385-resnet": "academic",
    "gao-24-106214": "complex-tables",
    "rfc2616-http": "digital-native",
    "ar-morocco-petitions-44-14": "rtl",
    "telegram-garfield-1881": "scanned",
    "letter-peabody-1863": "handwriting",
    "statute-1-1789": "degraded",
}


def cell_f1(a: str, b: str) -> float:
    def cells(t: str) -> Counter:
        c: Counter = Counter()
        for tbl in scoring._tables(t):
            for row in tbl:
                for cell in row:
                    n = scoring._normalize_cell(cell)
                    if n:
                        c[n] += 1
        return c

    ca, cb = cells(a), cells(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    tp = sum((ca & cb).values())
    prec = tp / sum(cb.values())
    rec = tp / sum(ca.values())
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def main() -> int:
    rows = []
    for doc, cat in DOCS.items():
        t1 = (PASS1 / f"{doc}.gt.md").read_text(encoding="utf-8")
        t2 = (PASS2 / f"{doc}.gt.md").read_text(encoding="utf-8")
        rows.append(
            {
                "doc": doc,
                "cat": cat,
                "text_sim": round(scoring.text_similarity(t2, t1), 4),
                "reading_order": round(scoring.reading_order(t2, t1), 4),
                "heading_f1": round(scoring.heading_f1(t2, t1), 4),
                "table_cell_f1": round(cell_f1(t1, t2), 4),
                "has_table": bool(scoring._tables(t1)),
            }
        )

    def mean(k: str) -> float:
        return round(sum(r[k] for r in rows) / len(rows), 4)

    tabled = [r for r in rows if r["has_table"]]
    summary = {
        "documents": rows,
        "means": {
            "text_sim": mean("text_sim"),
            "reading_order": mean("reading_order"),
            "heading_f1": mean("heading_f1"),
            "table_cell_f1_over_tabled_docs": (
                round(sum(r["table_cell_f1"] for r in tabled) / len(tabled), 4) if tabled else None
            ),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
