# pdfmux-bench

**The neutral, reproducible document-extraction benchmark — including the closed APIs that hide from academic benchmarks.**

Every serious PDF/document-extraction benchmark today has one of two problems:

1. **Vendor-controlled and stale.** Reducto's RD-TableBench is the most-cited table benchmark, and it has not been refreshed in ~20 months. A vendor's own benchmark, frozen in time, is marketing — not measurement.
2. **Academic but incomplete.** OmniDocBench, olmOCR-Bench, and the rest are rigorous and open — and they **omit every closed commercial API**. Reducto, LlamaParse, and Mistral OCR are the tools teams actually pay for, and no maintained public benchmark scores them head-to-head against the open-source engines on a fresh corpus.

Nobody maintains a fresh, reproducible benchmark that **includes the closed APIs**. That's the gap. pdfmux-bench fills it.

```
Engines scored side by side:
  pdfmux (local)   Reducto (API)   LlamaParse (API)   Mistral OCR (API)
  Docling (OSS)    Marker (OSS)    MinerU (OSS)
```

If your extraction tool turns a PDF into Markdown, it belongs on this board. Closed or open, hosted or local — same corpus, same metric code, same yardstick.

---

## Why trust this one

**It's reproducible by anyone.** The harness, the metrics, the corpus manifest, and the CI that regenerates the leaderboard are all in this repo. Bring your own API keys and you get the *exact* numbers we publish — or file a PR if you don't.

**It refreshes.** The corpus and results are versioned and re-run quarterly (see `VERSION` and the dated files in `results/`). A benchmark that doesn't move is a snapshot, not a benchmark.

**The metric code is engine-agnostic and public.** Every engine — pdfmux included — is scored by the identical functions in [`scoring.py`](scoring.py). pdfmux gets no home-field advantage. The weights are fixed and printed on every leaderboard.

**We publish losses.** pdfmux is maintained by the same people who run this benchmark, and pdfmux has publicly stated it is **#2 on reading order** on `opendataloader-bench`, not #1. That is the house rule here: **if an engine loses a category, the board says so.** A benchmark that only shows its sponsor winning is worthless, and everyone knows it.

> **The radical-honesty rule.** No fabricated numbers, ever. A score that was not actually computed renders as `PLACEHOLDER`/`TBD`. An engine without its API key is listed as *not run* — never silently dropped, never assigned a plausible number. An engine that errors or returns empty output has that recorded against its coverage. The board would rather show a hole than a lie.

---

## What it measures

Each document has a ground-truth Markdown file (`<doc>.gt.md`). Every engine's output is scored against it on six metrics ([`scoring.py`](scoring.py)):

| Metric | What it captures | Direction |
|---|---|---|
| **text_similarity** | Normalized character edit distance (token-Jaccard fallback for long docs). Did the words survive? | higher = better |
| **reading_order** | Kendall-tau agreement of block order vs ground truth. The metric pdfmux publicly reports #2 on. | higher = better |
| **heading_f1** | F1 over detected heading text. Did the document structure survive? | higher = better |
| **table_teds** | TEDS-style grid similarity over Markdown tables (cell-content F1 + row/col shape agreement). | higher = better |
| **structure_score** | Agreement on counts of headings / lists / table rows / code blocks / paragraphs. | higher = better |
| **hallucination_rate** | Fraction of output content-words absent from the source. | **lower** = better |

`overall` is a fixed-weight roll-up (printed on every board), but the leaderboard shows every sub-metric so no engine can hide behind one blended number.

> **On `table_teds`:** it is a grid-structure-aware similarity, not a byte-identical port of the original PubTabNet TEDS (which runs APTED tree-edit distance over parsed HTML). It correlates strongly with TEDS on well-formed tables and is documented candidly in `scoring.py`. Swapping in a full APTED TEDS is a tracked good-first-issue — see [CONTRIBUTING.md](CONTRIBUTING.md). We'd rather ship an honest approximation labeled as one than overclaim.

---

## Quick start

```bash
git clone https://github.com/NameetP/pdfmux-bench    # (repo name TBD at launch)
cd pdfmux-bench
pip install -r requirements.txt

# 1. Get the corpus (public-domain / openly-licensed docs, downloaded from the manifest).
python corpus/fetch_corpus.py            # see corpus/README.md

# 2. Run the engines you have. pdfmux runs with no key; APIs need env vars.
pip install pdfmux                       # the one engine that runs out of the box
python run_bench.py --engines pdfmux

# 3. Render the leaderboard.
python leaderboard.py --results results/latest.json
open LEADERBOARD.md
```

### Run everything you can

```bash
export REDUCTO_API_KEY=...        # Reducto
export LLAMA_CLOUD_API_KEY=...    # LlamaParse
export MISTRAL_API_KEY=...        # Mistral OCR
pip install docling marker-pdf mineru   # OSS local engines (heavy; optional)

python run_bench.py               # runs every engine whose key/package is present
python leaderboard.py --results results/latest.json
```

Engines you can't run are **skipped with a clear message and listed as "not run"** on the board — they never get a fabricated row.

### Useful flags

```bash
python run_bench.py --oss-only                  # only local, no-key engines (what CI runs)
python run_bench.py --engines pdfmux,docling    # a subset
python run_bench.py --category complex-tables   # one corpus category
python run_bench.py --quality high              # pdfmux quality preset (fast|standard|high)
```

---

## How it works

```
corpus/<category>/<doc>.pdf   +   <doc>.gt.md   (ground-truth Markdown)
                    │
                    ▼
run_bench.py   ── for each engine adapter ──▶  adapter.extract(pdf) -> markdown
                    │                                     │
                    │                              scoring.score_document(markdown, gt)
                    ▼                                     │
         results/bench-<timestamp>.json  ◀───────────────┘
                    │
                    ▼
leaderboard.py ─▶  LEADERBOARD.md  +  results/leaderboard.json
```

- **`adapters/`** — one file per engine, each a subclass of `Adapter` with a single `extract(pdf) -> markdown` method. Adding an engine is adding one file. See [`adapters/base.py`](adapters/base.py).
- **`run_bench.py`** — discovers the corpus, runs each available engine, scores every output, writes a timestamped results JSON. A per-document failure is recorded as an error row; it never crashes the run.
- **`scoring.py`** — the metrics. Zero dependencies, zero pdfmux imports, so the same code scores every engine identically and runs anywhere.
- **`leaderboard.py`** — aggregates a results JSON into ranked Markdown + JSON, including the "not run this cycle" honesty section.

---

## The engines

| Engine | Type | License | Runs in CI? | Adapter |
|---|---|---|---|---|
| **pdfmux** | local | MIT | ✅ yes | [`adapters/pdfmux_local.py`](adapters/pdfmux_local.py) |
| **Docling** | local (OSS) | MIT | ✅ yes | [`adapters/docling.py`](adapters/docling.py) |
| **Marker** | local (OSS) | GPL-3.0 | opt-in | [`adapters/marker.py`](adapters/marker.py) |
| **MinerU** | local (OSS) | AGPL-3.0 | opt-in | [`adapters/mineru.py`](adapters/mineru.py) |
| **Reducto** | closed API | commercial | ❌ needs key | [`adapters/reducto.py`](adapters/reducto.py) |
| **LlamaParse** | closed API | commercial | ❌ needs key | [`adapters/llamaparse.py`](adapters/llamaparse.py) |
| **Mistral OCR** | closed API | commercial | ❌ needs key | [`adapters/mistral_ocr.py`](adapters/mistral_ocr.py) |

CI runs the OSS/local engines on the public corpus every push and on a quarterly schedule, and regenerates the leaderboard. Closed-API engines are run by maintainers (and by anyone with keys) and their numbers are folded into the published board. See [`.github/workflows/bench.yml`](.github/workflows/bench.yml).

---

## Corpus

The corpus spans the failure modes that separate real engines from demo-ware: digital-native, scanned, complex tables, multi-column academic, forms/checkboxes, handwriting, Arabic/RTL, and degraded scans. **Only redistributable (public-domain or openly-licensed) documents are used**, pointed to by a manifest so licensing stays clean and the corpus can grow by PR. Full design, schema, and the starter manifest of real public-domain sources: [`corpus/README.md`](corpus/README.md).

---

## Governance

- **Neutrality.** pdfmux maintains this repo and is transparent about it. The defense against bias is not a promise — it's that anyone can rerun the whole thing and check. If pdfmux ever games a metric, the reproducibility is the receipt that catches it.
- **Adding your engine.** If you think your tool wins, **send a PR with an adapter.** We will run it and publish the result — win or lose. See [CONTRIBUTING.md](CONTRIBUTING.md).
- **Disputes.** Open an issue. Methodology changes are versioned; historical results stay in `results/` so the board's history is auditable.

## License

Code: **MIT** (see [LICENSE](LICENSE)) — matches pdfmux's OSS posture. Corpus documents retain their own upstream licenses, recorded per-document in the manifest.

---

*pdfmux-bench is built and maintained by the [pdfmux](https://github.com/NameetP/pdfmux) team. pdfmux is one engine on this board, scored by the same code as everyone else.*
