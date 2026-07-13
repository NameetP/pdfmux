# pdfmux-bench leaderboard

> **No official run has been published yet.** This file is a committed template so
> the format is visible and reviewable. Every score cell is `PLACEHOLDER` because
> no full engine run has been executed against a finalized, human-reviewed corpus.
> Per the radical-honesty rule, **we do not put a number here until it is real.**
>
> To generate the real board:
>
> ```bash
> python corpus/fetch_corpus.py            # fetch the corpus
> #   (build + review corpus/<category>/<id>.gt.md ground truth first)
> export REDUCTO_API_KEY=... LLAMA_CLOUD_API_KEY=... MISTRAL_API_KEY=...
> pip install pdfmux docling marker-pdf mineru
> python run_bench.py                      # runs every engine whose key/package is present
> python leaderboard.py --results results/latest.json   # overwrites this file with real numbers
> ```

Bench `v0.1.0` · **not yet run** · corpus assembly in progress (see `corpus/manifest.json`).

## Overall (template — awaiting first run)

| Rank | Engine | Overall | Text | Read-Order | Heading F1 | Table TEDS | Structure | Halluc↓ | Docs (ok/err/empty) | Avg ms |
|---:|---|--:|--:|--:|--:|--:|--:|--:|---|--:|
| — | **pdfmux** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |
| — | **Reducto** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |
| — | **LlamaParse** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |
| — | **Mistral OCR** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |
| — | **Docling** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |
| — | **Marker** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |
| — | **MinerU** | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | TBD | TBD |

_Ranks are intentionally blank. The board is not ranked until real numbers exist. Higher is better except Halluc (lower is better)._

---

_Metric weights (fixed, identical for every engine): `text_similarity`=0.30, `reading_order`=0.20, `heading_f1`=0.15, `table_teds`=0.20, `structure_score`=0.10, `hallucination`=0.05. Full methodology in [README.md](README.md). Losses are published on principle._
