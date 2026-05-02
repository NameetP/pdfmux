"""Run pdfmux against every labeled fixture and record the predicted confidence.

Reads:    eval/labels.csv
Writes:   eval/outputs/raw_scores.csv  (fixture, label, category, confidence,
                                        char_count, extractor, error)

The output is the input to eval/calibrate.py. This script is intentionally
narrow — extract per-document confidence and that's it. Don't compute ROC
here, don't recommend thresholds, don't filter rows. Keep eval steps
composable.

Run with:
    python eval/run_eval.py [--quality fast|standard|high]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
LABELS_PATH = EVAL_DIR / "labels.csv"
OUTPUTS_DIR = EVAL_DIR / "outputs"
SCORES_PATH = OUTPUTS_DIR / "raw_scores.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality",
        choices=["fast", "standard", "high"],
        default="standard",
        help="Quality preset passed to pdfmux.extract_json. Default: standard.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the smart result cache for this run (useful for timing).",
    )
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        sys.stderr.write(
            f"labels.csv not found at {LABELS_PATH}. Run eval/build_fixtures.py first.\n"
        )
        return 2

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Suppress upstream library noise so the run output is readable.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    # Deferred import — pdfmux pulls in pymupdf4llm/etc. on import.
    import pdfmux

    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixtures_dir = EVAL_DIR / "fixtures"
    output_rows: list[dict[str, str]] = []
    start = time.time()

    for i, row in enumerate(rows, start=1):
        fixture = row["fixture"]
        label = row["label"]
        category = row["category"]
        path = fixtures_dir / fixture

        confidence: float | None = None
        char_count: int = 0
        extractor: str = ""
        error: str = ""

        try:
            data = pdfmux.extract_json(
                str(path),
                quality=args.quality,
            )
            confidence = float(data.get("confidence", 0.0))
            char_count = len((data.get("content") or "").strip())
            extractor = str(data.get("extractor", ""))[:60]
        except Exception as e:
            # Per-fixture failure is data, not a crash. Record the error.
            error = type(e).__name__ + ": " + str(e)[:200]

        sys.stdout.write(
            f"  [{i:3d}/{len(rows)}] {fixture:30s} label={label:4s} "
            f"conf={confidence if confidence is not None else 'ERR'!s:>5} "
            f"chars={char_count}\n"
        )

        output_rows.append(
            {
                "fixture": fixture,
                "label": label,
                "category": category,
                "confidence": "" if confidence is None else f"{confidence:.4f}",
                "char_count": str(char_count),
                "extractor": extractor,
                "error": error,
            }
        )

    duration = time.time() - start

    fields = ["fixture", "label", "category", "confidence", "char_count", "extractor", "error"]
    with SCORES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nWrote {len(output_rows)} rows to {SCORES_PATH}")
    print(f"Duration: {duration:.1f}s ({duration / max(len(rows), 1):.2f}s/fixture)")
    print(f"Errors:   {sum(1 for r in output_rows if r['error'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
