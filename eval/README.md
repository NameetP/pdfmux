# pdfmux confidence calibration eval set

This directory holds the harness for calibrating pdfmux's per-document confidence score against ground-truth "is this extraction usable?" labels. The output is the empirically-defended threshold that 1.7 will use as the default `--min-confidence` for strict-mode batch runs.

It is intentionally separate from `src/pdfmux/eval/` (which is the `pdfmux benchmark` library that measures extraction *quality* against ground-truth markdown). This harness measures *confidence calibration*.

## What's here

```
eval/
├── README.md                — this file
├── build_fixtures.py        — generate 50 labeled PDFs into fixtures/
├── run_eval.py              — run pdfmux on each fixture, write raw_scores.csv
├── calibrate.py             — ROC + threshold recommendations at fixed precision targets
├── labels.csv               — ground-truth labels (committed)
├── fixtures/                — generated PDFs (committed; deterministic from a seed)
└── outputs/                 — raw_scores.csv, calibration_report.md, .json (gitignored)
```

## The pipeline

```
build_fixtures.py  →  fixtures/*.pdf  +  labels.csv  (good/bad ground truth)
                          │
                          ▼
run_eval.py        →  outputs/raw_scores.csv  (predicted confidence per fixture)
                          │
                          ▼
calibrate.py       →  outputs/calibration_report.md  (ROC + recommended thresholds)
                      outputs/calibration_summary.json
```

Three independent steps, three artifacts. You can re-label without re-extracting, re-extract without re-labeling, and re-calibrate as often as you like.

## How to run

```bash
# 1. Generate the fixture set + labels (idempotent; deterministic seed).
python eval/build_fixtures.py

# 2. Extract every fixture and record predicted confidence.
#    Use --quality standard if you want OCR on image-only fixtures.
#    The result cache is honored by default — pass --no-cache for clean re-runs.
python eval/run_eval.py --quality fast

# 3. Compute ROC + threshold recommendations.
python eval/calibrate.py

# 4. Read the report.
open eval/outputs/calibration_report.md
```

## Label semantics

Binary classification:

- **`good`** — a reasonable extractor (PyMuPDF + RapidOCR) should produce coherent, usable text from this document.
- **`bad`** — no extractor can produce useful text. The correct extractor output is "I extracted nothing usable, confidence near zero."

Labels are set by *construction* in `build_fixtures.py`, not by running pdfmux. That's deliberate: ground truth must be independent of the system under test.

## Threshold recommendations

`calibrate.py` picks two thresholds:

| Gate | Target | Used as |
|---|---|---|
| **strict** | precision >= 0.95 | The 1.7 default for `--min-confidence` on `pdfmux convert <dir> --strict`. We'd rather flag 5% false-bads than ship 5% false-goods. |
| **warning** | precision >= 0.80 | The stderr-WARNING line below this confidence (already shipped at 0.50 in 1.6.1). |

If a target precision can't be achieved on the eval set, `calibrate.py` reports the best precision observed and recommends expanding the dataset rather than locking a number that doesn't generalize.

## Calibration run (2026-07-16, `quality=standard` + `pdfmux[ocr]`)

Genuine uncached run over the 50 fixtures (28 good / 22 bad):

| Threshold | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|
| 0.00 | 0.683 | 1.000 | 0.812 | 0.740 |
| 0.10–0.70 | 0.849 | 1.000 | 0.918 | 0.900 |
| **0.75** | **1.000** | **1.000** | **1.000** | **1.000** |
| 0.90 | 1.000 | 0.821 | 0.902 | 0.900 |

**Recommended `--min-confidence` for the 1.7+ strict gate: 0.75** — precision 1.00, recall 1.00 (tp=28 fp=0 tn=22 fn=0).

**Read this honestly.** This is a small **internal regression guard** — ~7 distinct document shapes across 50 synthetic fixtures — not a competitive benchmark. Its job is to catch a regression in the confidence gate, not to prove field accuracy. The separation is currently clean, but the margin is thin: the `html_as_pdf` counterfeit sits at **0.730** against the 0.75 gate (0.02 of headroom). **Don't cite these numbers as a headline result** — the real, external proof point is the ARK 433-document customer batch (naive v1: 16 silently dropped / 11 with no log line → v4: 433/433, 0 silent). Expand the set (below) before leaning on it.

> *History: an earlier run reported **recall 0.714** here — an artifact of a broken Arabic fixture whose generator silently emitted `U+00B7` middle-dots instead of Arabic (PyMuPDF's default font substitutes rather than raising, so the `try/except` fallback was dead code). Fixed 2026-07-16: `_make_arabic` in `build_fixtures.py` now renders with a real Arabic font and **self-verifies** the glyphs survive extraction, raising rather than shipping garbage. Recall corrected 0.714 → 1.000; the Arabic category's mean confidence went 0.65 → 0.90.*

## A real bug surfaced by the first run

The first calibration run produced this:

```
| threshold | precision |
|----------:|----------:|
|     0.50  |    0.683  |
|     0.95  |    0.683  |
```

— precision flat across every threshold. Cause: `compute_document_confidence` was doing a content-weighted average of the per-page confidence values that the extractor wrote at yield time (always 1.0 for `fast`, with the comment "audit will reassess"). The reassessment never happened. Documents with zero or one extracted character returned confidence 1.0.

The fix is in `src/pdfmux/audit.py:compute_document_confidence`: re-score each page with `score_page(p.text, p.image_count)` before averaging, and stop flooring the per-page weight at 1 (which let blank pages register full weight).

Without this eval set the bug was invisible. With the eval set it was unmissable.

## Expanding the eval set

The 50 programmatic fixtures cover the failure modes we already knew about. To strengthen the calibration:

1. Add real customer PDFs as labeled fixtures (with explicit consent and PII redaction). Drop the file in `fixtures/` and add a row to `labels.csv` with the right label.
2. Add fixtures that hit the audit's individual penalties (mojibake, excessive whitespace, single-letter words, etc.) so the threshold sweep has more signal.
3. For Arabic / RTL: `_make_arabic` now embeds a real Arabic font and self-verifies the rendered glyphs (raising if none survive), so the fixture is a fair test rather than silent middle-dots. Add more RTL shapes — mixed LTR/RTL, ligature-heavy scripts — to broaden coverage.

Treat `labels.csv` as the source of truth and re-run all three scripts after any change.
