#!/usr/bin/env python3
"""pdfmux-bench harness — run every engine over the corpus and score it.

Discovers documents in `corpus/` (each `<doc>.pdf` paired with a `<doc>.gt.md`
ground-truth Markdown file), runs each registered engine adapter, scores the
output against ground truth with scoring.py, and writes a timestamped results
JSON into results/. Feed that JSON to leaderboard.py to render the scoreboard.

Design rules (mirrors pdfmux's own eval harness):
  - One engine's failure on one doc is DATA (a recorded error row), never a crash.
  - An engine without its API key is SKIPPED with a clear message, never faked.
  - Every engine is scored by the identical metric code — no home-field edge.

Usage:
    python run_bench.py                          # all engines, full corpus
    python run_bench.py --engines pdfmux,docling # subset of engines
    python run_bench.py --oss-only               # only local no-key engines (CI)
    python run_bench.py --category complex-tables # one corpus category
    python run_bench.py --corpus corpus --out results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import scoring
from adapters import OSS_LOCAL, get_adapters

HERE = Path(__file__).parent


@dataclass
class DocResult:
    engine: str
    doc_id: str
    category: str
    status: str  # "ok" | "error" | "empty"
    latency_ms: int = 0
    metrics: dict = field(default_factory=dict)
    error: str = ""
    char_count: int = 0


def discover_corpus(corpus_dir: Path, category: str | None) -> list[tuple[Path, Path, str]]:
    """Return (pdf, ground_truth_md, category) triples that have ground truth."""
    triples: list[tuple[Path, Path, str]] = []
    if not corpus_dir.is_dir():
        sys.stderr.write(f"corpus dir not found: {corpus_dir}\n")
        return triples
    for pdf in sorted(corpus_dir.rglob("*.pdf")):
        gt = pdf.with_suffix(".gt.md")
        if not gt.is_file():
            continue
        cat = pdf.parent.name if pdf.parent != corpus_dir else "uncategorized"
        if category and cat != category:
            continue
        triples.append((pdf, gt, cat))
    return triples


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines", help="comma-separated engine names (default: all registered)")
    p.add_argument("--oss-only", action="store_true", help="only local no-key engines (CI-safe)")
    p.add_argument("--category", help="restrict to one corpus category (subdir name)")
    p.add_argument("--corpus", default=str(HERE / "corpus"), help="corpus directory")
    p.add_argument("--out", default=str(HERE / "results"), help="results output directory")
    p.add_argument("--quality", default="standard", help="pdfmux quality preset (fast|standard|high)")
    args = p.parse_args(argv)

    corpus_dir = Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_names = args.engines.split(",") if args.engines else None
    adapters = get_adapters(engine_names)
    if args.oss_only:
        adapters = [a for a in adapters if a.name in OSS_LOCAL]
    # Apply quality preset to the pdfmux adapter if present.
    for a in adapters:
        if a.name == "pdfmux":
            a.quality = args.quality

    docs = discover_corpus(corpus_dir, args.category)
    if not docs:
        sys.stderr.write(
            "No scorable documents found (need <doc>.pdf + <doc>.gt.md pairs in corpus/).\n"
            "See corpus/README.md — download the corpus, then re-run.\n"
        )
        return 2

    print(f"pdfmux-bench: {len(docs)} document(s), {len(adapters)} engine(s)\n")

    results: list[DocResult] = []
    for adapter in adapters:
        if not adapter.available():
            print(f"SKIP  {adapter.label:14s} — {adapter.unavailable_reason()}")
            continue
        print(f"RUN   {adapter.label} ({adapter.license})")
        for pdf, gt_path, cat in docs:
            doc_id = pdf.stem
            gt_md = gt_path.read_text(encoding="utf-8")
            start = time.perf_counter()
            try:
                extracted = adapter.extract(pdf)
                latency = int((time.perf_counter() - start) * 1000)
                if not (extracted or "").strip():
                    results.append(DocResult(adapter.name, doc_id, cat, "empty", latency))
                    print(f"      {doc_id:28s} EMPTY  ({latency} ms)")
                    continue
                metrics = scoring.score_document(extracted, gt_md)
                results.append(
                    DocResult(
                        adapter.name, doc_id, cat, "ok", latency,
                        metrics=metrics, char_count=len(extracted),
                    )
                )
                print(f"      {doc_id:28s} overall={metrics['overall']:.3f}  ({latency} ms)")
            except Exception as e:  # per-doc failure is data, not a crash
                latency = int((time.perf_counter() - start) * 1000)
                err = f"{type(e).__name__}: {str(e)[:200]}"
                results.append(DocResult(adapter.name, doc_id, cat, "error", latency, error=err))
                print(f"      {doc_id:28s} ERROR  {err}")
        print()

    ran_engines = sorted({r.engine for r in results})
    payload = {
        "schema_version": "1.0",
        "bench_version": _bench_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "n_documents": len(docs),
        "engines_run": ran_engines,
        "metric_weights": scoring.DEFAULT_WEIGHTS,
        "results": [asdict(r) for r in results],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"bench-{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {latest}  (leaderboard reads this)")
    print(f"\nNext: python leaderboard.py --results {latest}")
    return 0


def _bench_version() -> str:
    vf = HERE / "VERSION"
    if vf.is_file():
        return vf.read_text(encoding="utf-8").strip()
    return "0.1.0-dev"


if __name__ == "__main__":
    raise SystemExit(run())
