"""Compute ROC + recommend confidence thresholds at fixed precision targets.

Reads:    eval/outputs/raw_scores.csv
Writes:   eval/outputs/calibration_report.md
          eval/outputs/calibration_summary.json

Stdlib-only — no sklearn dependency. The math is trivial: we sweep
threshold T over the sorted unique confidence values, compute precision,
recall, and F1 at each, then pick the T that achieves a target precision.

Threshold semantics:
    "predict good if confidence >= T"
        positive class = "good"  (extraction is usable)
        negative class = "bad"   (extraction is not usable)

We pick two thresholds:
    strict_gate    — T at precision >= 0.95 (this is what 1.7 will use as
                     --min-confidence default for `pdfmux convert <dir>
                     --strict`).
    warning_gate   — T at precision >= 0.80 (stderr WARNING fires below
                     this; manifest flags as 'medium').

If a precision target cannot be achieved on the eval set, we report the
best precision achieved and recommend collecting more data instead of
locking a number that doesn't generalize.

Run with:
    python eval/calibrate.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
SCORES_PATH = EVAL_DIR / "outputs" / "raw_scores.csv"
REPORT_PATH = EVAL_DIR / "outputs" / "calibration_report.md"
SUMMARY_PATH = EVAL_DIR / "outputs" / "calibration_summary.json"


def _load_scores() -> list[dict]:
    with SCORES_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _confusion_at_threshold(rows: list[dict], threshold: float) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for 'predict good if confidence >= threshold'.

    Rows where the extractor errored entirely are treated as predicted=bad
    (lowest possible confidence). This mirrors how pdfmux behaves: a raised
    exception means we don't even have a confidence number.
    """
    tp = fp = tn = fn = 0
    for r in rows:
        is_good = r["label"] == "good"
        try:
            conf = float(r["confidence"]) if r.get("confidence") else -1.0
        except ValueError:
            conf = -1.0
        predicted_good = conf >= threshold and conf >= 0
        if predicted_good and is_good:
            tp += 1
        elif predicted_good and not is_good:
            fp += 1
        elif not predicted_good and not is_good:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _candidate_thresholds(rows: list[dict]) -> list[float]:
    """Unique confidence values plus the canonical bands."""
    seen: set[float] = set()
    for r in rows:
        try:
            v = float(r["confidence"]) if r.get("confidence") else None
        except ValueError:
            v = None
        if v is not None and v >= 0:
            seen.add(round(v, 4))
    seen.update({0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95})
    return sorted(seen)


def _pick_threshold_at_precision(
    rows: list[dict],
    target_precision: float,
) -> tuple[float | None, dict]:
    """Find the LOWEST threshold T such that precision(T) >= target_precision.

    Lowest, not highest, because we want to maximize recall subject to the
    precision constraint. Returns (threshold, metrics_at_threshold) or
    (None, best_metrics_seen) if no threshold meets the target.
    """
    best: tuple[float, dict] | None = None
    candidates = _candidate_thresholds(rows)
    # Sort ascending so the first qualifying T is the lowest.
    for t in candidates:
        m = _metrics(*_confusion_at_threshold(rows, t))
        if m["precision"] >= target_precision and (m["tp"] + m["fp"]) > 0:
            return t, m
        # Track best F1 as a fallback to report
        if best is None or m["f1"] > best[1]["f1"]:
            best = (t, m)
    return None, (best[1] if best else _metrics(0, 0, 0, 0))


def _per_category_breakdown(rows: list[dict]) -> dict[str, dict]:
    """How does each fixture category score? Surfaces systematic failures."""
    out: dict[str, dict] = {}
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, cat_rows in by_cat.items():
        confs: list[float] = []
        labels: list[str] = []
        for r in cat_rows:
            try:
                v = float(r["confidence"]) if r.get("confidence") else None
            except ValueError:
                v = None
            if v is not None and v >= 0:
                confs.append(v)
            labels.append(r["label"])
        mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
        out[cat] = {
            "n": len(cat_rows),
            "n_good": sum(1 for label in labels if label == "good"),
            "n_bad": sum(1 for label in labels if label == "bad"),
            "mean_confidence": mean_conf,
            "n_extraction_errors": sum(1 for r in cat_rows if r.get("error")),
        }
    return out


def main() -> int:
    if not SCORES_PATH.exists():
        print(f"raw_scores.csv not found at {SCORES_PATH}. Run eval/run_eval.py first.")
        return 2

    rows = _load_scores()

    n_total = len(rows)
    n_good = sum(1 for r in rows if r["label"] == "good")
    n_bad = sum(1 for r in rows if r["label"] == "bad")
    n_errors = sum(1 for r in rows if r.get("error"))

    # Threshold sweep at the canonical bands — the table that goes in the report.
    bands = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = []
    for t in bands:
        m = _metrics(*_confusion_at_threshold(rows, t))
        sweep.append({"threshold": t, **m})

    strict_threshold, strict_metrics = _pick_threshold_at_precision(rows, 0.95)
    warning_threshold, warning_metrics = _pick_threshold_at_precision(rows, 0.80)

    by_category = _per_category_breakdown(rows)

    summary = {
        "n_total": n_total,
        "n_good": n_good,
        "n_bad": n_bad,
        "n_extraction_errors": n_errors,
        "sweep": sweep,
        "recommended": {
            "strict_gate": {
                "target_precision": 0.95,
                "threshold": strict_threshold,
                "metrics": strict_metrics,
            },
            "warning_gate": {
                "target_precision": 0.80,
                "threshold": warning_threshold,
                "metrics": warning_metrics,
            },
        },
        "by_category": by_category,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- Markdown report ---------------------------------------------------
    md: list[str] = []
    md.append("# pdfmux confidence calibration report\n")
    md.append(
        f"Generated from `{SCORES_PATH.relative_to(EVAL_DIR.parent)}` "
        f"({n_total} fixtures: {n_good} good / {n_bad} bad, "
        f"{n_errors} extraction errors).\n"
    )

    md.append("## Threshold sweep\n")
    md.append("Predict `good` if `confidence >= threshold`. "
              "Higher precision = fewer false-good. Higher recall = fewer false-bad.\n")
    md.append("| threshold | tp | fp | tn | fn | precision | recall |   f1 | accuracy |")
    md.append("|----------:|---:|---:|---:|---:|----------:|-------:|-----:|---------:|")
    for s in sweep:
        md.append(
            f"|     {s['threshold']:.2f} | {s['tp']:>2} | {s['fp']:>2} | "
            f"{s['tn']:>2} | {s['fn']:>2} |     {s['precision']:.3f} | "
            f"{s['recall']:.3f} | {s['f1']:.3f} |    {s['accuracy']:.3f} |"
        )
    md.append("")

    md.append("## Recommended thresholds\n")
    md.append("**`--strict --min-confidence` default for 1.7** — picked at "
              "P >= 0.95 (we'd rather flag 5% false-bads than ship 5% false-goods):\n")
    if strict_threshold is not None:
        m = strict_metrics
        md.append(f"- threshold: **{strict_threshold:.2f}**")
        md.append(
            f"- precision={m['precision']:.3f}, recall={m['recall']:.3f}, "
            f"f1={m['f1']:.3f}, accuracy={m['accuracy']:.3f}"
        )
        md.append(
            f"- confusion: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}"
        )
    else:
        md.append("- **no threshold achieves P >= 0.95 on this eval set.** "
                  "Best precision observed:")
        m = strict_metrics
        md.append(
            f"- precision={m['precision']:.3f}, recall={m['recall']:.3f}"
        )
        md.append("- Action: expand the eval set or re-examine the audit features.")
    md.append("")

    md.append("**stderr-WARNING gate (already shipped at 0.50 in 1.6.1)** — "
              "picked at P >= 0.80:\n")
    if warning_threshold is not None:
        m = warning_metrics
        md.append(f"- threshold: **{warning_threshold:.2f}**")
        md.append(
            f"- precision={m['precision']:.3f}, recall={m['recall']:.3f}, "
            f"f1={m['f1']:.3f}"
        )
        md.append(
            f"- confusion: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}"
        )
    else:
        md.append("- no threshold achieves P >= 0.80.")
    md.append("")

    md.append("## By category\n")
    md.append("Mean confidence per fixture category. Categories where mean confidence ")
    md.append("contradicts the label are systematic gaps to investigate.\n")
    md.append("| category | n | n_good | n_bad | mean_conf | extract_errors |")
    md.append("|----------|---:|------:|------:|----------:|---------------:|")
    for cat, stats in sorted(by_category.items()):
        md.append(
            f"| {cat} | {stats['n']} | {stats['n_good']} | {stats['n_bad']} | "
            f"{stats['mean_confidence']:.3f} | {stats['n_extraction_errors']} |"
        )
    md.append("")

    md.append("## How to act on this\n")
    md.append("1. If recommended thresholds look reasonable, lock them in `audit.py` "
              "as `STRICT_THRESHOLD_DEFAULT` and `WARNING_THRESHOLD_DEFAULT`.")
    md.append("2. If a category's mean confidence contradicts its label distribution "
              "(e.g. `bad` category with mean ≥ 0.8), the audit features miss that "
              "failure mode — add a check or fix the existing one.")
    md.append("3. Expand the eval set with real customer PDFs once you have them. "
              "Programmatic fixtures cover the failure modes we know; real PDFs cover "
              "the failure modes we don't.")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote calibration report → {REPORT_PATH}")
    print(f"Wrote summary JSON       → {SUMMARY_PATH}")
    print()
    if strict_threshold is not None:
        print(
            f"Recommended --min-confidence (1.7 default): {strict_threshold:.2f} "
            f"(precision {strict_metrics['precision']:.2f}, "
            f"recall {strict_metrics['recall']:.2f})"
        )
    else:
        print(
            "No threshold achieves P>=0.95 on the current eval set. "
            "Inspect the report and either expand the dataset or revisit the audit features."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
