#!/usr/bin/env python3
"""Render the pdfmux-bench leaderboard from a results JSON.

Reads a `results/*.json` produced by run_bench.py and emits:
  - LEADERBOARD.md   — human-readable ranked tables (overall + per category)
  - leaderboard.json — machine-readable aggregate for downstream trackers

Radical-honesty rules baked in:
  - Engines that were SKIPPED (no key / not installed) are listed explicitly as
    "not run", never omitted silently and never assigned a number.
  - Error and empty outputs count against an engine's coverage and are shown.
  - If NO results exist for a metric, the cell renders `PLACEHOLDER` — we never
    invent a score.

Usage:
    python leaderboard.py --results results/latest.json
    python leaderboard.py --results results/latest.json --out LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent

METRIC_ORDER = [
    ("overall", "Overall"),
    ("text_similarity", "Text"),
    ("reading_order", "Read-Order"),
    ("heading_f1", "Heading F1"),
    ("table_teds", "Table TEDS"),
    ("structure_score", "Structure"),
    ("hallucination_rate", "Halluc↓"),
]

ENGINE_LABELS = {
    "pdfmux": "pdfmux",
    "reducto": "Reducto",
    "llamaparse": "LlamaParse",
    "mistral_ocr": "Mistral OCR",
    "docling": "Docling",
    "marker": "Marker",
    "mineru": "MinerU",
}
ALL_ENGINES = list(ENGINE_LABELS)


def aggregate(results: list[dict]) -> dict:
    """engine -> {metric -> mean, coverage stats}."""
    agg: dict[str, dict] = {}
    for eng in {r["engine"] for r in results}:
        rows = [r for r in results if r["engine"] == eng]
        ok = [r for r in rows if r["status"] == "ok"]
        metric_means: dict[str, float | None] = {}
        for key, _ in METRIC_ORDER:
            vals = [r["metrics"][key] for r in ok if key in r.get("metrics", {})]
            metric_means[key] = round(statistics.mean(vals), 4) if vals else None
        latencies = [r["latency_ms"] for r in ok if r.get("latency_ms")]
        agg[eng] = {
            "metrics": metric_means,
            "n_total": len(rows),
            "n_ok": len(ok),
            "n_error": sum(1 for r in rows if r["status"] == "error"),
            "n_empty": sum(1 for r in rows if r["status"] == "empty"),
            "avg_latency_ms": round(statistics.mean(latencies)) if latencies else None,
        }
    return agg


def _cell(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "PLACEHOLDER"


def render_markdown(payload: dict) -> str:
    results = payload["results"]
    agg = aggregate(results)
    ran = set(payload.get("engines_run", []))

    # Rank by overall desc; engines with no overall sink to the bottom.
    def sort_key(eng: str):
        ov = agg[eng]["metrics"].get("overall")
        return (ov is None, -(ov or 0))

    ranked = sorted(agg, key=sort_key)

    lines: list[str] = []
    lines.append("# pdfmux-bench leaderboard\n")
    lines.append(
        f"Bench `v{payload.get('bench_version', '?')}` · generated "
        f"`{payload.get('generated_at', '?')}` · "
        f"{payload.get('n_documents', '?')} documents.\n"
    )
    lines.append(
        "> Neutral, reproducible document-extraction benchmark. Regenerate with "
        "`python run_bench.py && python leaderboard.py --results results/latest.json`. "
        "Higher is better except Halluc (lower is better). `PLACEHOLDER` = not yet "
        "measured; skipped engines are listed below and never assigned a number.\n"
    )

    # Overall table
    header = (
        "| Rank | Engine | "
        + " | ".join(lbl for _, lbl in METRIC_ORDER)
        + " | Docs (ok/err/empty) | Avg ms |"
    )
    sep = "|---:|---|" + "".join("--:|" for _ in METRIC_ORDER) + "---|--:|"
    lines.append("## Overall\n")
    lines.append(header)
    lines.append(sep)
    for i, eng in enumerate(ranked, 1):
        a = agg[eng]
        cells = [_cell(a["metrics"].get(k)) for k, _ in METRIC_ORDER]
        cov = f"{a['n_ok']}/{a['n_error']}/{a['n_empty']}"
        ms = str(a["avg_latency_ms"]) if a["avg_latency_ms"] is not None else "—"
        lines.append(
            f"| {i} | **{ENGINE_LABELS.get(eng, eng)}** | "
            + " | ".join(cells)
            + f" | {cov} | {ms} |"
        )
    lines.append("")

    # Per-category tables (overall metric only, for compactness)
    categories = sorted({r["category"] for r in results})
    if len(categories) > 1:
        lines.append("## By category (overall score)\n")
        cat_header = "| Engine | " + " | ".join(categories) + " |"
        cat_sep = "|---|" + "".join("--:|" for _ in categories)
        lines.append(cat_header)
        lines.append(cat_sep)
        for eng in ranked:
            row = [ENGINE_LABELS.get(eng, eng)]
            for cat in categories:
                vals = [
                    r["metrics"]["overall"]
                    for r in results
                    if r["engine"] == eng and r["category"] == cat
                    and r["status"] == "ok" and "overall" in r.get("metrics", {})
                ]
                row.append(f"{statistics.mean(vals):.3f}" if vals else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Not-run engines — the honesty section.
    not_run = [e for e in ALL_ENGINES if e not in ran]
    if not_run:
        lines.append("## Not run this cycle\n")
        lines.append(
            "These registered engines were skipped (missing API key or package). "
            "They are **not** assigned a score — supply the key/package and re-run "
            "to include them:\n"
        )
        for e in not_run:
            lines.append(f"- **{ENGINE_LABELS.get(e, e)}** (`{e}`)")
        lines.append("")

    lines.append("---\n")
    lines.append(
        "_Metric weights (fixed, identical for every engine): "
        + ", ".join(f"`{k}`={v}" for k, v in payload.get("metric_weights", {}).items())
        + ". Full methodology in README.md. Losses are published on principle._\n"
    )
    return "\n".join(lines)


def render_json(payload: dict) -> dict:
    agg = aggregate(payload["results"])
    return {
        "bench_version": payload.get("bench_version"),
        "generated_at": payload.get("generated_at"),
        "n_documents": payload.get("n_documents"),
        "metric_weights": payload.get("metric_weights"),
        "engines_run": payload.get("engines_run"),
        "leaderboard": agg,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", default=str(HERE / "results" / "latest.json"))
    p.add_argument("--out", default=str(HERE / "LEADERBOARD.md"))
    p.add_argument("--json-out", default=str(HERE / "results" / "leaderboard.json"))
    args = p.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.is_file():
        p.error(f"results file not found: {results_path}. Run run_bench.py first.")

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    md = render_markdown(payload)
    Path(args.out).write_text(md, encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(render_json(payload), indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
