---
date: 2026-09-02
product: pdfmux
type: eval-report
pipeline_item: 2026-09-02-pdf-inspector-eval
dispatch_id: 7d3b8f6b27cb1630
status: partial
executed_by: cto-agent dispatch (Claude Code Remote, ephemeral container)
pdfmux_pin: main @ c9cb5f5ad855ebd7e6bf4fb129de965a50431605 (v1.8.7)
pdf_inspector_pin: 1.17.0 (PyPI wheel, manylinux2014 x86_64, cp38-abi3)
---

# Benchmark Report: pdf-inspector vs pdfmux `detect.py` — scanned-vs-text classification

**Decision line (see §8 for the full rule evaluation):**

> **`SWAP-PROPOSE`** — all four mechanical criteria hold on this corpus. Per the PRD this
> files a **new S1 pipeline item** and authorizes nothing else. The strategic-dependency
> note (§9) travels with it as mandatory input, and §8 records two material caveats the
> CEO stage should weigh: the macro-accuracy win is concentrated in two 5-doc synthetic
> cells, and on the **cost-weighted axis pdfmux is strictly cleaner (0 silent failures, 0
> wasted-OCR docs, 0 hard errors vs pdf-inspector's 3 wasted-OCR docs + 5 hard errors on
> repairable files)** — a swap would still need `detect.py` behind it as a fallback.

**Completeness: PARTIAL.** Every number below comes from runs executed on 2026-09-02 in
this container. What makes it partial: 47 labeled docs instead of the PRD's 60; no
opendataloader reproducibility probe; all-synthetic corpus. Exact caveats in §3.1.

---

## 1. What was run

Head-to-head, classification-only, same corpus, same machine, 3 timed runs per tool:

- **pdfmux baseline:** `src/pdfmux/detect.py::classify()` at `main` =
  `c9cb5f5ad855ebd7e6bf4fb129de965a50431605` (version 1.8.7, last commit 2026-08-13),
  imported unmodified from the read-only clone. 7-class output mapped to the shared
  4-class axis using the production precedence from `pipeline._classify_to_page_type`
  (arabic > scanned > tables > graphical > mixed > digital), then
  arabic/tables/digital → TextBased, scanned → Scanned, graphical → ImageBased,
  mixed → Mixed — exactly the PRD §4 mapping.
- **pdf-inspector:** official PyPI bindings `pdf-inspector==1.17.0`,
  `pdf_inspector.classify_pdf(path)` — its documented lightweight classification-only
  entry point ("Faster than detect_pdf as it skips building the full PdfProcessResult").
  Returns `pdf_type` ∈ {text_based, scanned, image_based, mixed}, `confidence`,
  `page_count`, `pages_needing_ocr` (0-indexed).

**Zero pdfmux source files were modified** (PRD anti-metric). All eval code and outputs
live in `eval/pdf-inspector-eval/` in the code repo (uncommitted in the container; this
report embeds the full runner + manifest in §11 for the Mac-side commit).

## 2. Step 0 — fact re-verification at the pin

| §1 "as fetched" claim | Verified at pin? | What was actually found |
|---|---|---|
| Official Python bindings on PyPI | **CONFIRMED** | Package name `pdf-inspector`, latest at eval time **1.17.0**; installs through the proxy in seconds; native manylinux2014 x86_64 wheel, cp38-abi3, 5.3 MB |
| MIT license | **CONFIRMED** | Wheel metadata `License: MIT`; repo shows MIT |
| `process_pdf()` / `process_pdf_with_ocr()` API | **CONFIRMED, and richer** | Both present, plus `classify_pdf` / `detect_pdf` / `extract_text` / `extract_pages_markdown` / structure + region extraction, all with `_bytes` variants. `classify_pdf` is the correct classification-only comparator and was used |
| 4-class output with 0.0–1.0 confidence + per-page OCR routing | **CONFIRMED** | `pdf_type` ∈ text_based/scanned/image_based/mixed; `confidence`; `pages_needing_ocr` 0-indexed |
| Rust core | Consistent, not directly inspected | Native abi3 extension wheel is consistent with a Rust/PyO3 core |
| CLI tools `detect-pdf`, `pdf2md` | **NOT in the wheel** | The PyPI wheel installs **no CLI entry points** (venv/bin gains nothing). CLIs presumably ship via cargo/npm. Irrelevant to swap path (a); would matter for path (b) subprocess wrapper — path (b) cost estimate raised accordingly |
| 18.1k★, +541/day, 492 commits, 88 issues | Not re-verified | As-fetched 2026-09-02 only (fetch reported 18.2k★); GitHub API not queried from the container |
| Self-reported 0.875 @ 0.470s on opendataloader (200 PDFs) | **NOT reproduced** | The 10-doc opendataloader probe was not materialized (§3.1). Treat as an unverified vendor claim |
| Version note | — | The repo README's benchmark section cites 0.2.6 (2026-07-31); PyPI is at 1.17.0 five weeks later. The "fast-moving dependency" characterization is confirmed by version churn alone |

## 3. Corpus

**70 documents total: 47 classifiable (ground-truth labeled on the shared 4-class axis) +
23 robustness probes** (invalid/degenerate inputs where a 4-class label is not defensible;
scored separately in §7, excluded from accuracy).

Every label is set **by construction** (the fixture generator determines the class),
never by running either system under test — same principle as the repo's existing
`eval/build_fixtures.py`. Sources: the repo's `eval/fixtures/` set (51 docs, labels.csv,
already labeled by construction) + 19 docs generated by `build_corpus_extra.py` (§11) to
fill the Mixed / ImageBased / searchable-scan / 500-page / encrypted cells the repo set
lacks.

| GT class (n=47) | n | Cells |
|---|---|---|
| TextBased | 27 | digital ×6, multipage ×6, table ×3, Arabic digital ×3, repairable-truncated (70% bytes) ×5, searchable scan ×3, 500-page ×1 |
| Scanned | 10 | image-only fixtures ×5, generated 3-page scans ×5 |
| ImageBased | 5 | generated slide-deck pages (3 figures + <100-char caption) ×5 |
| Mixed | 5 | generated 4-page docs (text pages 0/2, raster pages 1/3) ×5 |
| Probes (n=23) | — | zero-byte ×5, HTML-as-PDF ×5, truncated-15% ×4, micro-text ×4, blank-page ×4, AES-256 encrypted ×1 |

Contestable labels, decided and documented rather than hidden: **searchable scan =
TextBased** (a complete invisible text layer is present and extractable — the doc does
not need OCR; this is precisely the PRD's "classic hard case" and both tools were scored
against that label); **image-only render = Scanned** (full-page raster of rendered text,
needs OCR); **repairable-truncated = TextBased** (PyMuPDF repairs it and full text
extracts). Micro-text and blank-page have no defensible 4-class label → probes.

### 3.1 Why PARTIAL — exact deviations from the PRD corpus spec

1. **47 labeled docs vs the PRD's 60 (20/20/20 target).** Actual mix 27/10/5/5 —
   TextBased-heavy, thin ImageBased and Mixed cells (5 each; single-doc misses in those
   cells move per-class accuracy by 20 points — see §8 sensitivity).
2. **The 10-doc opendataloader reproducibility probe was not materialized** — the
   container's egress is a pip-scoped proxy; the public corpus was not fetchable. The
   0.875 claim therefore remains unverified (§2).
3. **No real-world documents.** All 70 docs are PyMuPDF-generated synthetics (repo
   fixtures included). Homogeneous producer = real generalization risk: arXiv-style
   academic layouts, CID-font docs, scanner-noise scans, and photographed pages are
   untested. The PRD's arXiv / Internet Archive gap-filling was not possible offline.
4. **Page-level OCR ground truth** exists only on the generated hybrid cells + image-only
   fixtures (46 pages), not on a broader hybrid set.
5. Single machine, single OS (Linux/x86_64), as the PRD allows.

## 4. Environment (identical for both tools)

| | |
|---|---|
| Machine | Intel Xeon @ 2.10 GHz, 4 vCPU, 15 GiB RAM (ephemeral Linux container, kernel 6.18.44, glibc 2.39) |
| Python | 3.11.15 (venv at `eval/pdf-inspector-eval/venv`) |
| pdf-inspector | 1.17.0 (PyPI wheel) |
| pdfmux | clone `main` @ c9cb5f5, v1.8.7; **imported from `src/` via a stub package skipping `pdfmux/__init__`** (extraction-side deps not installed; `detect.py`, `errors.py`, `pdf_cache.py`, `arabic.py` execute unmodified) |
| PyMuPDF | 1.28.2 — **inside** pdfmux's pin `>=1.24.0,<1.29` |
| Timing | `time.perf_counter` around the classify call only; pdfmux's `pdf_cache` cleared after every call so every measurement includes the document open; run 1 cold-ish, runs 2–3 warm (OS page cache); tools run in separate loops, same process |

Caveat: not pdfmux's full locked resolve (`pymupdf4llm` etc. not installed — not imported
by the detect path). PyMuPDF, the only dependency `detect.py` actually exercises, is
within the production pin.

## 5. Accuracy (4-class doc-level, n=47)

| Metric | pdfmux `detect.py` | pdf-inspector 1.17.0 | Δ |
|---|---|---|---|
| **Macro accuracy (decision-rule metric)** | **0.500** | **0.926** | **+42.6 pts** |
| Micro accuracy (all docs pooled) | 0.787 | 0.830 | +4.3 pts |
| TextBased per-class (n=27) | 1.000 | 0.704 | −29.6 |
| Scanned per-class (n=10) | 1.000 | 1.000 | 0 |
| ImageBased per-class (n=5) | 0.000 | 1.000 | +100 |
| Mixed per-class (n=5) | 0.000 | 1.000 | +100 |
| **Cost-weighted silent failures (Scanned/Mixed → TextBased)** | **0 (weighted 0)** | **0 (weighted 0)** | tie |
| Wasted-OCR docs (TextBased → OCR-routed class) | 0 | 3 | pdfmux better |
| Hard errors on classifiable docs | 0 | 5 | pdfmux better |

Confusion matrices (rows = ground truth, run-1 predictions; all three runs were
prediction-stable for both tools — zero instability observed):

```
pdfmux                                   pdf-inspector
            TB   Sc   IB   Mx  ERR                  TB   Sc   IB   Mx  ERR
TextBased   27    -    -    -    -       TextBased   19    -    3    -    5
Scanned      -   10    -    -    -       Scanned      -   10    -    -    -
ImageBased   -    5    -    -    -       ImageBased   -    -    5    -    -
Mixed        -    -    5    -    -       Mixed        -    -    -    5    -
```

### 5.1 Every disagreement between the tools, one line each (18 classifiable docs, 4 families)

| Docs | GT | pdfmux said | pdf-inspector said | Note |
|---|---|---|---|---|
| `edge-trunc-70pct-00…04` (5) | TextBased | TextBased (repairs via PyMuPDF, extracts full text) | **`ValueError: Invalid PDF structure`** | pdf-inspector refuses damaged-but-repairable files pdfmux handles — a real robustness gap for a "self-healing extraction" pipeline |
| `gen-mixed-00…04` (5) | Mixed | ImageBased (raw `graphical` — see bug B1) | Mixed ✓ | pdfmux's error is between OCR-routed siblings: routing consequence is mistral_ocr/rapidocr chain instead of the mixed multi-pass chain — suboptimal, not silent |
| `gen-imagedeck-00…04` (5) | ImageBased | Scanned (raw `scanned` — see bug B2) | ImageBased ✓ | again an OCR-routed sibling; text under 50 chars/page + images trips detect.py's scanned-page rule |
| `gen-searchable-00…02` (3) | TextBased | TextBased ✓ (reads the text layer) | **ImageBased** (routes both pages to OCR) | **the PRD's classic hard case — pdf-inspector fails it, pdfmux passes.** Wasted OCR on already-searchable scans; at fleet scale this is the expensive error direction pdf-inspector's own pitch ("skip OCR when not needed") targets |

All other 29 classifiable docs: both tools agree with GT and each other. Probe-side
disagreements are in §7.

### 5.2 Page-level OCR routing (46 labeled pages: gen-mixed, gen-scanned, image-only, searchable)

| Metric | pdfmux (`scanned_pages ∪ graphical_pages`) | pdf-inspector (`pages_needing_ocr`) |
|---|---|---|
| Page accuracy | **1.000** (46/46) | 0.870 (40/46) |
| Missed OCR pages (silent direction) | 0 | 0 |
| Unneeded OCR pages (wasted direction) | 0 | 6 (both pages of all 3 searchable scans) |

## 6. Speed (classification-only, wall clock, 3 runs × 70 docs each)

| | pdfmux `detect.py` | pdf-inspector `classify_pdf` |
|---|---|---|
| Median, run 1 (cold) | 8.74 ms | 0.52 ms |
| Median, run 2 (warm) | 10.19 ms | 0.52 ms |
| Median, run 3 (warm) | 8.06 ms | 0.33 ms |
| **Median, pooled (n=210)** | **8.99 ms** | **0.48 ms** — **18.7× faster** |
| **p95, pooled** | **37.89 ms** | **29.68 ms** — no worse |
| Mean, pooled | 18.31 ms | 7.11 ms |
| Max (the 500-page doc, both tools) | 542 ms | 128 ms |
| 500-page doc, 3 dedicated reps | 597 / 632 / 513 ms | 50 / 38 / 29 ms (~13× faster) |

Cold vs warm was negligible for both (files are small and page-cached after generation).
Both tools classified the 500-page doc correctly (TextBased); pdfmux's `_detect_tables`
sampling dominates its large-doc cost. Absolute numbers are container-specific; the
relative comparison is the metric (PRD §9).

## 7. Robustness probes (23 docs, no 4-class GT — behavior recorded, not scored)

| Probe | pdfmux | pdf-inspector | Read |
|---|---|---|---|
| zero-byte ×5 | `FileError` (clean, typed) | `ValueError: Not a PDF: file is empty` | both reject cleanly |
| HTML-as-PDF ×5 | **classifies as digital/TextBased** (downstream audit layer is the catch) | **rejects**: `Not a PDF: file appears to be HTML` | pdf-inspector strictly better at the classify layer |
| truncated-15% ×4 | `FileError` (clean) | `ValueError: Invalid PDF structure` | both reject |
| micro-text ×4 | TextBased | TextBased | agree |
| blank-page ×4 | TextBased (documented all-empty→digital rule, conf 0.5) | **Scanned** | pdf-inspector would route empty pages to OCR — wasted-OCR direction on degenerate input |
| AES-256 encrypted ×1 | **raises bare `ValueError: document closed or encrypted`** — violates `classify()`'s documented FileError contract (bug B3) | Scanned (misleading but non-fatal; `process_pdf_with_ocr` accepts `password=`) | neither handles it well; pdfmux's is a contract bug |
| Crashes/segfaults, either tool, any of 70 docs × 3 runs | **none** | **none** | criterion-3 input |

## 8. Decision rule, applied mechanically (PRD §5)

| # | Criterion | Result | Detail |
|---|---|---|---|
| 1 | Accuracy: ≥5 pts macro, or eliminates a ≥3-doc silent-failure class | **PASS** (on the macro arm) | +42.6 pts macro (0.926 vs 0.500). Silent-failure arm does NOT apply: both tools have zero silent failures — there was no silent-failure class to eliminate |
| 2 | Speed: ≥2× median, p95 no worse | **PASS** | 18.7× median (0.48 vs 8.99 ms); p95 29.68 vs 37.89 ms |
| 3 | Bindings work on our corpus; integration ≤2 agent-days | **PASS** | Installed and ran all 210 doc-runs with zero crashes (11 docs raised clean typed errors, 5 of them classifiable — regression handled by mandatory fallback in the estimate below); path (a) estimate 1.5–2 agent-days |
| 4 | MIT at pin; no relicensing signal | **PASS** | MIT in wheel metadata at 1.17.0 and in repo; no relicensing signal observed |

**All four hold → `SWAP-PROPOSE`.** This files a new S1 item; it does not authorize a swap.

**Material caveats for the S1/CEO stage (honesty over tidiness):**

- **The macro win is concentrated where the corpus is thinnest.** The +42.6-pt macro delta
  comes almost entirely from the ImageBased and Mixed cells (5 synthetic docs each, 20
  pts/doc granularity). On pooled micro accuracy the gap is **+4.3 pts — under the 5-pt
  bar the rule sets for macro**. The rule says macro; the rule fired; but a 60-doc corpus
  with real-world docs could plausibly move either side by more than the margin in those
  cells.
- **On the cost-weighted axis the baseline is cleaner.** pdfmux: 0 silent failures, 0
  wasted-OCR docs, 0 errors, page-level OCR routing 46/46. pdf-inspector: 0 silent
  failures, but 3 wasted-OCR docs (all searchable scans — the PRD's named hard case), 6
  wasted-OCR pages, and 5 hard errors on repairable files. Every pdfmux miss lands in an
  OCR-capable chain (text is never silently lost); pdf-inspector's misses waste money or
  refuse the file outright.
- Both caveats cut against the swap; neither is a criterion under the rule as written.

**Either-outcome obligations (PRD §5):** the build-vs-buy note for the acquisition package
should be drafted from §5–§7 (headline: *pdfmux's classifier has zero silent failures and
zero wasted OCR on this corpus and uniquely survives repairable-corrupt and searchable-scan
inputs; pdf-inspector is 18.7× faster at the median and sharper on Mixed/ImageBased
separation; and classification is commodity routing either way — pdfmux's differentiation,
verified extraction, has no counterpart in pdf-inspector*). And the found-bug backlog items:

- **B1 (P2):** `detect.py`/`_classify_to_page_type` — genuinely mixed text+raster docs
  report `graphical`, because both flags fire and graphical outranks mixed; the `mixed`
  route is effectively unreachable whenever >25% of pages are also image-heavy. Files:
  `src/pdfmux/detect.py` (graphical threshold), `src/pdfmux/pipeline.py:540`.
- **B2 (P3):** image-heavy pages with captions under 50 chars classify as scanned pages
  (`detect.py` page rule), so slide decks read as Scanned doc-level. Routing still reaches
  OCR; label is wrong on the shared axis.
- **B3 (P2):** `classify()` on an encrypted PDF leaks a bare `ValueError: document closed
  or encrypted` from page access instead of the documented `FileError` (`detect.py`
  wraps only the open, not page iteration).
- **B4 (P3):** HTML-as-PDF passes classification as digital; only the downstream audit
  catches it. pdf-inspector demonstrates cheap magic-bytes strictness at the classify layer.

## 9. Integration-cost estimate (estimate only — nothing was integrated)

| Path | Estimate | Notes from this run |
|---|---|---|
| (a) Official PyPI bindings behind a feature flag | **1.5–2 agent-days** | Wheel installs and runs cleanly (criterion 3). Scope: pinned dep + flag + 4-class→7-class shim + **mandatory error-fallback to `detect.py`** (pdf-inspector hard-fails 5/47 classifiable docs and misroutes searchable scans — the fallback is not optional) + tests. Note: `detect.py` **cannot be deleted in any swap** — Arabic/tables/academic sub-typing has no pdf-inspector counterpart and drives the routing matrix, so a swap adds a native dependency without removing code |
| (b) Subprocess wrapper around `detect-pdf` CLI | **3+ agent-days** (raised from PRD's 2–3) | The PyPI wheel ships **no CLI**; path (b) would mean cargo-building and vendoring a Rust binary per platform |
| (c) Custom PyO3 bindings | 5+ days | Out of the question, unchanged |

## 10. Strategic-dependency note (verbatim from PRD §4 — mandatory S1 input)

> a swap puts a fast-moving (492 commits), well-funded competitor's library at the front
> of the core pipeline — supply-chain surface via native wheels, version churn, and an
> acquisition-narrative cost (an acquirer diligencing pdfmux finds Firecrawl code at the
> routing layer). This note goes in the report verbatim as S1 input for any swap proposal.

Corroborated at eval time: native abi3 wheel (supply-chain surface confirmed), and README
0.2.6 → PyPI 1.17.0 in ~5 weeks (version churn confirmed).

## 11. Reproducibility — full runner source + corpus manifest

Container is ephemeral; these embedded copies are canonical for the Mac-side commit to
`eval/pdf-inspector-eval/` in the pdfmux code repo. File SHA-256s as run:
`runner.py` e5a25ebf0b76afb2da91933300bb79eebd543907ccd4c93a81258aca60395222 ·
`build_corpus_extra.py` 157d47bc995224188622a4a9b444990ab944d75e063fa4078e9cf00a014f86ce ·
`manifest.csv` 4e672e745814c42838d74e5b132431dafa21c6992142eafd21c5b34d90902749.

Re-run: `python3 -m venv venv && venv/bin/pip install pdf-inspector==1.17.0 pymupdf==1.28.2 python-bidi && venv/bin/python build_corpus_extra.py && venv/bin/python runner.py` from `eval/pdf-inspector-eval/` in the pdfmux clone.

### 11.1 `eval/pdf-inspector-eval/build_corpus_extra.py`

```python
"""Generate supplemental corpus fixtures for the pdf-inspector eval.

The repo's eval/fixtures set (51 docs, labels.csv) covers TextBased and
image-only/Scanned cells plus robustness probes, but has no Mixed,
ImageBased (image-heavy deck), searchable-scan, 500+ page, or encrypted
docs. This script generates those cells deterministically with PyMuPDF so
every ground-truth label is set BY CONSTRUCTION, independent of both
systems under test (same principle as eval/build_fixtures.py).

Output: ./corpus-extra/*.pdf + ./manifest.csv covering BOTH the repo
fixtures (referenced in place) and the generated ones.

Idempotent, deterministic (fixed seed, fixed creation metadata).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import fitz  # PyMuPDF

SEED = 20260902
HERE = Path(__file__).parent
EXTRA = HERE / "corpus-extra"
FIXTURES = HERE.parent / "fixtures"
REPO_LABELS = HERE.parent / "labels.csv"
MANIFEST = HERE / "manifest.csv"

PARAS = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. How vexingly quick daft zebras jump.",
    "Quarterly logistics throughput rose 14 percent as the Jebel Ali "
    "corridor cleared its customs backlog ahead of schedule.",
    "PDF extraction quality depends on the source document structure: "
    "digital text exports cleanly, scanned images need OCR, and tables "
    "require dedicated extractors for fidelity.",
    "When extraction fails silently, downstream RAG pipelines retrieve "
    "empty chunks and confidently hallucinate answers from missing data.",
    "Verification is the differentiator: a classifier only routes, but a "
    "verifier certifies that what came out matches what went in.",
]


def _text_doc(pages: int, offset: int = 0) -> fitz.Document:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        body = "\n\n".join(PARAS[(i + offset + j) % len(PARAS)] for j in range(3))
        page.insert_text((72, 72), f"Section {i + 1}\n\n{body}", fontsize=11)
    return doc


def _page_png(offset: int, dpi: int = 110) -> bytes:
    """Render one deterministic text page to PNG bytes (a 'scan')."""
    src = _text_doc(1, offset=offset)
    pix = src[0].get_pixmap(dpi=dpi)
    data = pix.tobytes("png")
    src.close()
    return data


def _small_png(color: tuple[float, float, float]) -> bytes:
    """A small colored-figure PNG for slide-deck style pages."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(10, 10, 290, 190), color=color, fill=color)
    page.draw_circle(fitz.Point(150, 100), 60, color=(0, 0, 0), width=3)
    pix = page.get_pixmap(dpi=96)
    data = pix.tobytes("png")
    doc.close()
    return data


def make_scanned(path: Path, pages: int, offset: int) -> None:
    """Full-page raster pages, no text layer — a classic scan."""
    doc = fitz.open()
    for i in range(pages):
        png = _page_png(offset + i)
        page = doc.new_page()
        page.insert_image(page.rect, stream=png)
    doc.save(str(path))
    doc.close()


def make_mixed(path: Path, offset: int) -> None:
    """4 pages: digital text on 0 and 2, full-page raster on 1 and 3."""
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        if i % 2 == 0:
            body = "\n\n".join(PARAS[(offset + i + j) % len(PARAS)] for j in range(3))
            page.insert_text((72, 72), f"Chapter {i + 1}\n\n{body}", fontsize=11)
        else:
            page.insert_image(page.rect, stream=_page_png(offset + i))
    doc.save(str(path))
    doc.close()


def make_imagedeck(path: Path, offset: int) -> None:
    """4 slide-style pages: 3 figure images each + a short caption (<100 chars)."""
    colors = [(0.9, 0.3, 0.2), (0.2, 0.5, 0.9), (0.3, 0.8, 0.4)]
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        for k, c in enumerate(colors):
            r = fitz.Rect(60 + k * 170, 120, 200 + k * 170, 260)
            page.insert_image(r, stream=_small_png(c))
        page.insert_text((72, 80), f"Figure deck slide {i + 1} — chart {offset}", fontsize=14)
    doc.save(str(path))
    doc.close()


def make_searchable_scan(path: Path, offset: int) -> None:
    """Full-page raster + invisible text layer (render_mode=3) — a searchable scan."""
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_image(page.rect, stream=_page_png(offset + i))
        body = "\n\n".join(PARAS[(offset + i + j) % len(PARAS)] for j in range(3))
        page.insert_text(
            (72, 72), f"Section {i + 1}\n\n{body}", fontsize=11, render_mode=3
        )
    doc.save(str(path))
    doc.close()


def make_huge(path: Path, pages: int = 500) -> None:
    doc = _text_doc(pages)
    doc.save(str(path))
    doc.close()


def make_encrypted(path: Path) -> None:
    doc = _text_doc(2)
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# Ground-truth mapping for repo fixtures (labels.csv category -> 4-class axis or probe).
# Set by construction semantics documented in eval/build_fixtures.py.
REPO_GT = {
    "digital_single": ("TextBased", "classifiable"),
    "digital_multi": ("TextBased", "classifiable"),
    "digital_table": ("TextBased", "classifiable"),
    "arabic": ("TextBased", "classifiable"),  # digital Arabic text; sub-type has no 4-class counterpart
    "truncated_light": ("TextBased", "classifiable"),  # repairable digital text
    "image_only": ("Scanned", "classifiable"),  # full-page raster, needs OCR
    "zero_byte": ("invalid", "probe"),
    "html_as_pdf": ("invalid", "probe"),
    "truncated_heavy": ("invalid", "probe"),
    "micro_text": ("degenerate", "probe"),  # valid PDF, 1-char body — GT contestable
    "blank_page": ("degenerate", "probe"),  # valid PDF, zero content — no 4-class GT
}


def main() -> None:
    EXTRA.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []

    # 1. Repo fixtures, referenced in place.
    with open(REPO_LABELS, newline="") as f:
        for r in csv.DictReader(f):
            gt, tier = REPO_GT[r["category"]]
            p = FIXTURES / r["fixture"]
            rows.append(
                dict(
                    file=str(p.relative_to(HERE.parent.parent)),
                    source="repo-fixture (eval/fixtures, label by construction)",
                    category=r["category"],
                    gt_4class=gt,
                    tier=tier,
                    sha256=sha256(p),
                    note=r["note"],
                )
            )

    # 2. Generated fixtures.
    gen: list[tuple[str, str, str, str, str]] = []
    for i in range(5):
        p = EXTRA / f"gen-scanned-{i:02d}.pdf"
        make_scanned(p, pages=3, offset=i)
        gen.append((p.name, "gen_scanned", "Scanned", "classifiable", "3 full-page raster pages, no text layer"))
    for i in range(5):
        p = EXTRA / f"gen-mixed-{i:02d}.pdf"
        make_mixed(p, offset=i)
        gen.append((p.name, "gen_mixed", "Mixed", "classifiable", "4 pages: text on 0/2, raster on 1/3"))
    for i in range(5):
        p = EXTRA / f"gen-imagedeck-{i:02d}.pdf"
        make_imagedeck(p, offset=i)
        gen.append((p.name, "gen_imagedeck", "ImageBased", "classifiable", "4 slide pages, 3 figures + <100 char caption each"))
    for i in range(3):
        p = EXTRA / f"gen-searchable-{i:02d}.pdf"
        make_searchable_scan(p, offset=i)
        gen.append((p.name, "gen_searchable_scan", "TextBased", "classifiable", "raster + invisible text layer (render_mode=3); complete text extractable"))
    p = EXTRA / "gen-huge-500p.pdf"
    make_huge(p)
    gen.append((p.name, "gen_huge", "TextBased", "classifiable", "500 digital text pages — sampling/speed probe"))
    p = EXTRA / "gen-encrypted.pdf"
    make_encrypted(p)
    gen.append((p.name, "gen_encrypted", "encrypted", "probe", "AES-256, user password set — error-behavior probe"))

    for name, cat, gt, tier, note in gen:
        p = EXTRA / name
        rows.append(
            dict(
                file=str(p.relative_to(HERE.parent.parent)),
                source="generated (build_corpus_extra.py, label by construction)",
                category=cat,
                gt_4class=gt,
                tier=tier,
                sha256=sha256(p),
                note=note,
            )
        )

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_class = sum(1 for r in rows if r["tier"] == "classifiable")
    n_probe = sum(1 for r in rows if r["tier"] == "probe")
    print(f"manifest: {len(rows)} docs ({n_class} classifiable, {n_probe} probes)")
    from collections import Counter

    print(Counter(r["gt_4class"] for r in rows if r["tier"] == "classifiable"))


if __name__ == "__main__":
    main()
```

### 11.2 `eval/pdf-inspector-eval/runner.py`

```python
"""Head-to-head classifier benchmark: pdfmux detect.py vs pdf-inspector.

Runs both classifiers over the labeled corpus in manifest.csv, three timed
runs each, and writes results.json with per-doc predictions, per-run
latencies, confusion matrices on the shared 4-class axis, cost-weighted
error counts, page-level OCR-routing accuracy, and probe behavior.

Comparison axis (per PRD 2026-09-02-pdf-inspector-eval):
  shared 4-class doc axis: TextBased / Scanned / ImageBased / Mixed.
  pdfmux 7-class -> shared axis uses the production precedence from
  pipeline._classify_to_page_type (arabic > scanned > tables > graphical
  > mixed > digital) then maps arabic/tables/digital -> TextBased,
  scanned -> Scanned, graphical -> ImageBased, mixed -> Mixed.

Timing protocol:
  - classification only (pdfmux: detect.classify; pdf-inspector:
    classify_pdf — its documented lightweight classification entry point).
  - 3 runs x all docs x both tools, same process, same machine.
  - pdfmux's pdf_cache is cleared after every classify call so every
    measurement includes the document open (no cross-run cache leak).
  - run 1 is cold-ish (first touch of each file this process; OS page
    cache may hold fixtures), runs 2-3 are warm. Reported per-run.

pdfmux is imported straight from the read-only clone's src/ via a stub
package (skipping pdfmux/__init__.py, which pulls extraction-side deps
not needed to classify). detect.py itself and its imports (errors,
pdf_cache, arabic) execute unmodified.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent  # pdfmux clone root
MANIFEST = HERE / "manifest.csv"
RESULTS = HERE / "results.json"
N_RUNS = 3

# --- pdfmux import without package __init__ (read-only clone, no install) ---
_pkg = types.ModuleType("pdfmux")
_pkg.__path__ = [str(REPO / "src" / "pdfmux")]
sys.modules["pdfmux"] = _pkg
from pdfmux.detect import classify as pdfmux_classify  # noqa: E402
from pdfmux.pdf_cache import close_all as pdfmux_cache_clear  # noqa: E402

import pdf_inspector  # noqa: E402


def pdfmux_page_type(c) -> str:
    """Reimplements pipeline._classify_to_page_type (production precedence)."""
    if c.is_arabic:
        return "arabic"
    if c.is_scanned:
        return "scanned"
    if c.has_tables:
        return "tables"
    if c.is_graphical:
        return "graphical"
    if c.is_mixed:
        return "mixed"
    return "digital"


PDFMUX_TO_4CLASS = {
    "arabic": "TextBased",
    "tables": "TextBased",
    "digital": "TextBased",
    "scanned": "Scanned",
    "graphical": "ImageBased",
    "mixed": "Mixed",
}

PI_TO_4CLASS = {
    "text_based": "TextBased",
    "scanned": "Scanned",
    "image_based": "ImageBased",
    "mixed": "Mixed",
}


def run_pdfmux(path: str) -> dict:
    t0 = time.perf_counter()
    try:
        c = pdfmux_classify(path)
        dt = time.perf_counter() - t0
        pt = pdfmux_page_type(c)
        return dict(
            ok=True,
            latency_s=dt,
            raw_class=pt,
            pred=PDFMUX_TO_4CLASS[pt],
            confidence=c.confidence,
            page_count=c.page_count,
            ocr_pages=sorted(set(c.scanned_pages) | set(c.graphical_pages)),
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        return dict(ok=False, latency_s=dt, error=f"{type(e).__name__}: {e}"[:200])
    finally:
        try:
            pdfmux_cache_clear()
        except Exception:
            pass


def run_pi(path: str) -> dict:
    t0 = time.perf_counter()
    try:
        c = pdf_inspector.classify_pdf(path)
        dt = time.perf_counter() - t0
        return dict(
            ok=True,
            latency_s=dt,
            raw_class=c.pdf_type,
            pred=PI_TO_4CLASS.get(c.pdf_type, c.pdf_type),
            confidence=c.confidence,
            page_count=c.page_count,
            ocr_pages=sorted(c.pages_needing_ocr),
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        return dict(ok=False, latency_s=dt, error=f"{type(e).__name__}: {e}"[:200])


# Page-level OCR-needed ground truth, by construction (hybrid/scanned cells only).
def page_ocr_gt(category: str, page_count: int) -> list[int] | None:
    if category in ("gen_scanned", "image_only"):
        return list(range(page_count))
    if category == "gen_mixed":
        return [1, 3]
    if category == "gen_searchable_scan":
        return []  # complete invisible text layer present — text extractable
    return None


def summarize(latencies: list[float]) -> dict:
    s = sorted(latencies)
    p95 = s[max(0, int(round(0.95 * len(s))) - 1)]
    return dict(
        n=len(s),
        median_ms=round(statistics.median(s) * 1000, 2),
        p95_ms=round(p95 * 1000, 2),
        mean_ms=round(statistics.fmean(s) * 1000, 2),
        max_ms=round(max(s) * 1000, 2),
    )


def main() -> None:
    docs = list(csv.DictReader(open(MANIFEST, newline="")))
    for d in docs:
        d["abspath"] = str(REPO / d["file"])

    tools = {"pdfmux": run_pdfmux, "pdf_inspector": run_pi}
    per_doc: dict[str, dict] = {d["file"]: dict(d) for d in docs}
    latencies: dict[str, list[list[float]]] = {t: [[] for _ in range(N_RUNS)] for t in tools}

    for run in range(N_RUNS):
        for tool, fn in tools.items():
            for d in docs:
                r = fn(d["abspath"])
                latencies[tool][run].append(r["latency_s"])
                if run == 0:
                    per_doc[d["file"]][tool] = r
                else:
                    prev = per_doc[d["file"]][tool]
                    now_pred = r.get("pred", r.get("error"))
                    prev_pred = prev.get("pred", prev.get("error"))
                    if now_pred != prev_pred:
                        per_doc[d["file"]].setdefault("instability", []).append(
                            f"{tool} run{run}: {prev_pred} -> {now_pred}"
                        )

    # --- accuracy on classifiable tier ---
    classifiable = [d for d in docs if d["tier"] == "classifiable"]
    classes = ["TextBased", "Scanned", "ImageBased", "Mixed"]
    acc: dict[str, dict] = {}
    for tool in tools:
        conf: dict[str, Counter] = defaultdict(Counter)
        correct_by_class: Counter = Counter()
        total_by_class: Counter = Counter()
        silent = []  # gt Scanned/Mixed -> pred TextBased (silent bad extraction)
        wasted = []  # gt TextBased -> pred needs-OCR class (wasted OCR)
        errors = []
        for d in classifiable:
            r = per_doc[d["file"]][tool]
            gt = d["gt_4class"]
            pred = r.get("pred", "ERROR") if r.get("ok") else "ERROR"
            conf[gt][pred] += 1
            total_by_class[gt] += 1
            if pred == gt:
                correct_by_class[gt] += 1
            if gt in ("Scanned", "Mixed") and pred == "TextBased":
                silent.append(dict(file=d["file"], gt=gt, weight=3 if gt == "Scanned" else 2))
            if gt == "TextBased" and pred in ("Scanned", "ImageBased", "Mixed"):
                wasted.append(dict(file=d["file"], pred=pred))
            if pred == "ERROR":
                errors.append(dict(file=d["file"], error=r.get("error")))
        per_class_acc = {
            c: round(correct_by_class[c] / total_by_class[c], 4) if total_by_class[c] else None
            for c in classes
        }
        macro = round(
            statistics.fmean(v for v in per_class_acc.values() if v is not None), 4
        )
        micro = round(sum(correct_by_class.values()) / len(classifiable), 4)
        acc[tool] = dict(
            n=len(classifiable),
            per_class_accuracy=per_class_acc,
            macro_accuracy=macro,
            micro_accuracy=micro,
            confusion={g: dict(c) for g, c in conf.items()},
            silent_failures=silent,
            silent_failure_weighted=sum(x["weight"] for x in silent),
            wasted_ocr=wasted,
            errors_on_classifiable=errors,
        )

    # --- page-level OCR routing on hybrid/scanned cells ---
    page_ocr: dict[str, dict] = {}
    for tool in tools:
        tp = fp = fn_ = tn = 0
        details = []
        for d in classifiable:
            r = per_doc[d["file"]][tool]
            if not r.get("ok"):
                continue
            gt_pages = page_ocr_gt(d["category"], r["page_count"])
            if gt_pages is None:
                continue
            got = set(r["ocr_pages"])
            want = set(gt_pages)
            allp = set(range(r["page_count"]))
            tp += len(got & want)
            fp += len(got - want)
            fn_ += len(want - got)
            tn += len(allp - got - want)
            if got != want:
                details.append(dict(file=d["file"], want=sorted(want), got=sorted(got)))
        total_pages = tp + fp + fn_ + tn
        page_ocr[tool] = dict(
            pages=total_pages,
            accuracy=round((tp + tn) / total_pages, 4) if total_pages else None,
            missed_ocr_pages=fn_,
            unneeded_ocr_pages=fp,
            disagreements=details,
        )

    # --- probe behavior ---
    probes = []
    for d in docs:
        if d["tier"] != "probe":
            continue
        row = dict(file=d["file"], category=d["category"])
        for tool in tools:
            r = per_doc[d["file"]][tool]
            row[tool] = r.get("pred") if r.get("ok") else r.get("error")
        probes.append(row)

    # --- speed ---
    speed = {
        tool: {
            f"run{i+1}": summarize(latencies[tool][i]) for i in range(N_RUNS)
        }
        | {"pooled": summarize([x for run in latencies[tool] for x in run])}
        for tool in tools
    }

    env = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu=subprocess.run(
            ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
            capture_output=True, text=True,
        ).stdout.strip(),
        nproc=subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(),
        mem=subprocess.run(
            ["sh", "-c", "free -h | awk '/^Mem/{print $2}'"], capture_output=True, text=True
        ).stdout.strip(),
        pdf_inspector_version=version_of("pdf_inspector"),
        pymupdf_version=version_of("pymupdf"),
        pdfmux_git_sha=subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        pdfmux_version=subprocess.run(
            ["sh", "-c", f"grep -m1 '^version' {REPO}/pyproject.toml"],
            capture_output=True, text=True,
        ).stdout.strip(),
    )

    out = dict(
        env=env,
        n_docs=len(docs),
        n_classifiable=len(classifiable),
        gt_composition=dict(Counter(d["gt_4class"] for d in classifiable)),
        accuracy=acc,
        page_ocr_routing=page_ocr,
        speed=speed,
        probes=probes,
        per_doc={
            k: {
                t: {kk: vv for kk, vv in v[t].items() if kk != "latency_s"}
                for t in tools if t in v
            }
            | dict(gt=v["gt_4class"], tier=v["tier"], instability=v.get("instability", []))
            for k, v in per_doc.items()
        },
    )
    RESULTS.write_text(json.dumps(out, indent=1))

    print(json.dumps(dict(env=env, gt=out["gt_composition"]), indent=1))
    for tool in tools:
        a = acc[tool]
        print(
            f"\n{tool}: macro={a['macro_accuracy']} micro={a['micro_accuracy']} "
            f"per-class={a['per_class_accuracy']} "
            f"silent={len(a['silent_failures'])} (weighted {a['silent_failure_weighted']}) "
            f"wasted_ocr={len(a['wasted_ocr'])} errors={len(a['errors_on_classifiable'])}"
        )
        print(f"  speed: {json.dumps(speed[tool])}")
        print(f"  page_ocr: {json.dumps({k: v for k, v in page_ocr[tool].items() if k != 'disagreements'})}")


def version_of(mod: str) -> str:
    import importlib.metadata

    for name in (mod, mod.replace("_", "-")):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "?"


if __name__ == "__main__":
    main()
```

### 11.3 `eval/pdf-inspector-eval/manifest.csv` (corpus manifest, SHA-256 per doc)

```csv
file,source,category,gt_4class,tier,sha256,note
eval/fixtures/good-digital-00.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_single,TextBased,classifiable,d1ac4e99a0496053936c5c7e4ba47e28ce746efcab8e5263378d5e6c0e260b37,"1 page, clean digital text"
eval/fixtures/good-digital-01.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_single,TextBased,classifiable,d2f1336e58a050dbc4e2609c7675d8779574f899750eaa3398db3ff6c604f479,"1 page, clean digital text"
eval/fixtures/good-digital-02.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_single,TextBased,classifiable,f25244f5ab9d6caab542eca332f5a19c0b2d80df3385707a0e4b8bcbd3b9aac1,"1 page, clean digital text"
eval/fixtures/good-digital-03.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_single,TextBased,classifiable,74aa74f53da5dcb87f56c6da0706df2b87f4f1e0372fbfd2525fdbfd6f176856,"1 page, clean digital text"
eval/fixtures/good-digital-04.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_single,TextBased,classifiable,e423e21e67aaf647b3ec8c2e74283be4af79b8bc69f2f430b203b88e227adfc9,"1 page, clean digital text"
eval/fixtures/good-digital-05.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_single,TextBased,classifiable,01ab5cce2d731353d5dc9d494e24bbb766167b405053648ba5b7f364f168bf34,"1 page, clean digital text"
eval/fixtures/good-multipage-00.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_multi,TextBased,classifiable,33bb96c14c183e7be5d740e9395c6bcab234d62ea023d9de0a029359358735a1,"3 pages, clean digital text"
eval/fixtures/good-multipage-01.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_multi,TextBased,classifiable,d62982614b464a58dc8524a2b503aa0900656b196fd7dd2f453757ef67b71343,"3 pages, clean digital text"
eval/fixtures/good-multipage-02.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_multi,TextBased,classifiable,f53f8428e7e2505b8baf3ecf7bc1458874e04acec7adb636f77a7a370d8b7d55,"3 pages, clean digital text"
eval/fixtures/good-multipage-03.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_multi,TextBased,classifiable,e3a13c46ae835d10d41f075f04d9df1debfa0d1c7835b3dd2eea9ee5c3399501,"3 pages, clean digital text"
eval/fixtures/good-multipage-04.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_multi,TextBased,classifiable,1a53e1f1f8c07f216c195162ea8f82229768b969c72ab0c4f03a8b75e1fc5ab7,"3 pages, clean digital text"
eval/fixtures/good-multipage-05.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_multi,TextBased,classifiable,2a4639c7e84a3e4f593690448c8e63ea86676b8f02251a748bcda56aa6782ce3,"3 pages, clean digital text"
eval/fixtures/good-table-00.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_table,TextBased,classifiable,62db7ab375f98f9bcbcd1428aaf652a7ce6d2a7962fe0c1fb0e1062587cae8d6,table-heavy markdown body
eval/fixtures/good-table-01.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_table,TextBased,classifiable,7180616abd54388cd4976c00ca7447ea51538ab20bca1f86c6833f9a123bbcc1,table-heavy markdown body
eval/fixtures/good-table-02.pdf,"repo-fixture (eval/fixtures, label by construction)",digital_table,TextBased,classifiable,12e1bf42503ba44cc0a7c05d7107794834d638049a72e3d9038cc9e8db890b07,table-heavy markdown body
eval/fixtures/good-arabic-00.pdf,"repo-fixture (eval/fixtures, label by construction)",arabic,TextBased,classifiable,c22888bfc87508265274a53f481b4d6f1f18f7e26535ea7ab3ba2e7d97a0963d,Arabic-only content
eval/fixtures/good-arabic-01.pdf,"repo-fixture (eval/fixtures, label by construction)",arabic,TextBased,classifiable,4227cdc547b1085e7112d30ec44b6d4bd9e1060f59f0cf99cb115e506712830a,Arabic-only content
eval/fixtures/good-arabic-02.pdf,"repo-fixture (eval/fixtures, label by construction)",arabic,TextBased,classifiable,e15ac09a2ee2054cc21c7b9aa84ff82890dd24b28b968011d38cd14910f988f6,Arabic-only content
eval/fixtures/bad-zero-byte-00.pdf,"repo-fixture (eval/fixtures, label by construction)",zero_byte,invalid,probe,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,0-byte file
eval/fixtures/bad-zero-byte-01.pdf,"repo-fixture (eval/fixtures, label by construction)",zero_byte,invalid,probe,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,0-byte file
eval/fixtures/bad-zero-byte-02.pdf,"repo-fixture (eval/fixtures, label by construction)",zero_byte,invalid,probe,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,0-byte file
eval/fixtures/bad-zero-byte-03.pdf,"repo-fixture (eval/fixtures, label by construction)",zero_byte,invalid,probe,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,0-byte file
eval/fixtures/bad-zero-byte-04.pdf,"repo-fixture (eval/fixtures, label by construction)",zero_byte,invalid,probe,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,0-byte file
eval/fixtures/bad-html-as-pdf-00.pdf,"repo-fixture (eval/fixtures, label by construction)",html_as_pdf,invalid,probe,4c941569074074d56fdc8d3093a0f907b0b3a431f462802ecaff8050158a5119,"HTML body, .pdf extension"
eval/fixtures/bad-html-as-pdf-01.pdf,"repo-fixture (eval/fixtures, label by construction)",html_as_pdf,invalid,probe,4c941569074074d56fdc8d3093a0f907b0b3a431f462802ecaff8050158a5119,"HTML body, .pdf extension"
eval/fixtures/bad-html-as-pdf-02.pdf,"repo-fixture (eval/fixtures, label by construction)",html_as_pdf,invalid,probe,4c941569074074d56fdc8d3093a0f907b0b3a431f462802ecaff8050158a5119,"HTML body, .pdf extension"
eval/fixtures/bad-html-as-pdf-03.pdf,"repo-fixture (eval/fixtures, label by construction)",html_as_pdf,invalid,probe,4c941569074074d56fdc8d3093a0f907b0b3a431f462802ecaff8050158a5119,"HTML body, .pdf extension"
eval/fixtures/bad-html-as-pdf-04.pdf,"repo-fixture (eval/fixtures, label by construction)",html_as_pdf,invalid,probe,4c941569074074d56fdc8d3093a0f907b0b3a431f462802ecaff8050158a5119,"HTML body, .pdf extension"
eval/fixtures/bad-trunc-15pct-00.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_heavy,invalid,probe,7a7365e6f1804099420c83b4acfcf02fd6128ae42b6f4a9c994c59ca5826f209,15% bytes kept
eval/fixtures/bad-trunc-15pct-01.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_heavy,invalid,probe,7a7365e6f1804099420c83b4acfcf02fd6128ae42b6f4a9c994c59ca5826f209,15% bytes kept
eval/fixtures/bad-trunc-15pct-02.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_heavy,invalid,probe,7a7365e6f1804099420c83b4acfcf02fd6128ae42b6f4a9c994c59ca5826f209,15% bytes kept
eval/fixtures/bad-trunc-15pct-03.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_heavy,invalid,probe,7a7365e6f1804099420c83b4acfcf02fd6128ae42b6f4a9c994c59ca5826f209,15% bytes kept
eval/fixtures/bad-micro-text-00.pdf,"repo-fixture (eval/fixtures, label by construction)",micro_text,degenerate,probe,e6d1aab37b41cac9265e4c5f90d102b9e96239dedc51013fa42e39744b9c1d35,single character body
eval/fixtures/bad-micro-text-01.pdf,"repo-fixture (eval/fixtures, label by construction)",micro_text,degenerate,probe,66f08702ecc64afe8b2b54743b6543199f1f023150023be256031713bfebd1e6,single character body
eval/fixtures/bad-micro-text-02.pdf,"repo-fixture (eval/fixtures, label by construction)",micro_text,degenerate,probe,a3a7fa1a32c80b1a2110a77a1984705f082e60b3a9ebc18686e39de6981cc043,single character body
eval/fixtures/bad-micro-text-03.pdf,"repo-fixture (eval/fixtures, label by construction)",micro_text,degenerate,probe,ef25fa49fc90d603aecd5f700fdd25ba76c4346a331d6e18075ea7bf69565dc2,single character body
eval/fixtures/edge-trunc-70pct-00.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_light,TextBased,classifiable,9ea9ed5d5b145d8df92f1e6178ad4ddf99a57115273e351b50a6c7fe96249c66,70% bytes kept — PyMuPDF repair
eval/fixtures/edge-trunc-70pct-01.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_light,TextBased,classifiable,9ea9ed5d5b145d8df92f1e6178ad4ddf99a57115273e351b50a6c7fe96249c66,70% bytes kept — PyMuPDF repair
eval/fixtures/edge-trunc-70pct-02.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_light,TextBased,classifiable,9ea9ed5d5b145d8df92f1e6178ad4ddf99a57115273e351b50a6c7fe96249c66,70% bytes kept — PyMuPDF repair
eval/fixtures/edge-trunc-70pct-03.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_light,TextBased,classifiable,9ea9ed5d5b145d8df92f1e6178ad4ddf99a57115273e351b50a6c7fe96249c66,70% bytes kept — PyMuPDF repair
eval/fixtures/edge-trunc-70pct-04.pdf,"repo-fixture (eval/fixtures, label by construction)",truncated_light,TextBased,classifiable,1b99a572a14c512d4244c0d48d4f880e308817beecc6b5a62606fdd8ec626328,70% bytes kept — PyMuPDF repair
eval/fixtures/edge-image-only-00.pdf,"repo-fixture (eval/fixtures, label by construction)",image_only,Scanned,classifiable,13531ec5e9678b36874caae4b6f980076f47e9956533016675763d72b9ef1c41,"rendered raster, needs OCR"
eval/fixtures/edge-image-only-01.pdf,"repo-fixture (eval/fixtures, label by construction)",image_only,Scanned,classifiable,f1dd9ef44cf1094e0d8d0227c5628275301471e7d7d5b253d0fdba4ff7878e94,"rendered raster, needs OCR"
eval/fixtures/edge-image-only-02.pdf,"repo-fixture (eval/fixtures, label by construction)",image_only,Scanned,classifiable,0677a4f899a40a563930fea81f6e087df29e69310e5c1306a8eb3ef7d45db246,"rendered raster, needs OCR"
eval/fixtures/edge-image-only-03.pdf,"repo-fixture (eval/fixtures, label by construction)",image_only,Scanned,classifiable,6798cd0e4b7b1cab31687d78816ad4b361c99318ebe2297a51829fafea740fb5,"rendered raster, needs OCR"
eval/fixtures/edge-image-only-04.pdf,"repo-fixture (eval/fixtures, label by construction)",image_only,Scanned,classifiable,ce626a2294ad546028f5e6b93f5cc99e599add24a0af3cda3021ef90790b4c69,"rendered raster, needs OCR"
eval/fixtures/bad-blank-page-00.pdf,"repo-fixture (eval/fixtures, label by construction)",blank_page,degenerate,probe,af04875dcb3ab63cf3c81dd30693d43fbfcf803191f6112d6920ca76dc9d3b51,"structurally valid, zero text"
eval/fixtures/bad-blank-page-01.pdf,"repo-fixture (eval/fixtures, label by construction)",blank_page,degenerate,probe,5a4696ea7fa8da0140d94ef73bb48eab0998650a5d9d7fda782a7486d40c6427,"structurally valid, zero text"
eval/fixtures/bad-blank-page-02.pdf,"repo-fixture (eval/fixtures, label by construction)",blank_page,degenerate,probe,48cf6b89d491a1d92d3e79375452d988f23e92dc278c3bfea34158fbd12f5123,"structurally valid, zero text"
eval/fixtures/bad-blank-page-03.pdf,"repo-fixture (eval/fixtures, label by construction)",blank_page,degenerate,probe,0f84504047ad00fc79a35bee3578c1b501b72609cf01dc77b58f03ca2d811366,"structurally valid, zero text"
eval/pdf-inspector-eval/corpus-extra/gen-scanned-00.pdf,"generated (build_corpus_extra.py, label by construction)",gen_scanned,Scanned,classifiable,64e86a68561b8d46cd19787dcdcfdee21504eab9d776faad4f81f9ff77bcee2e,"3 full-page raster pages, no text layer"
eval/pdf-inspector-eval/corpus-extra/gen-scanned-01.pdf,"generated (build_corpus_extra.py, label by construction)",gen_scanned,Scanned,classifiable,1143dc671d95547ec679c9b41f671b85bd4770a6eb4568d13b199bbd5a7dca20,"3 full-page raster pages, no text layer"
eval/pdf-inspector-eval/corpus-extra/gen-scanned-02.pdf,"generated (build_corpus_extra.py, label by construction)",gen_scanned,Scanned,classifiable,6c1e25ecf9b42f959360a91a4e519e76d37bd5a032106fa1650368082e120be1,"3 full-page raster pages, no text layer"
eval/pdf-inspector-eval/corpus-extra/gen-scanned-03.pdf,"generated (build_corpus_extra.py, label by construction)",gen_scanned,Scanned,classifiable,a3374dda89edbbaaee0344472c352d7100b096e38c50b60e02d1766131d830aa,"3 full-page raster pages, no text layer"
eval/pdf-inspector-eval/corpus-extra/gen-scanned-04.pdf,"generated (build_corpus_extra.py, label by construction)",gen_scanned,Scanned,classifiable,8420781ddf9968851210f2de733466a079302d6ba3102268ec55d75a89fd9a85,"3 full-page raster pages, no text layer"
eval/pdf-inspector-eval/corpus-extra/gen-mixed-00.pdf,"generated (build_corpus_extra.py, label by construction)",gen_mixed,Mixed,classifiable,a15a3c2ebb66d63d0d485cf1cbb520555a38be7135bdb94a2c833cfb02bc19af,"4 pages: text on 0/2, raster on 1/3"
eval/pdf-inspector-eval/corpus-extra/gen-mixed-01.pdf,"generated (build_corpus_extra.py, label by construction)",gen_mixed,Mixed,classifiable,68fa236e77cfa1fa353b3c0c9cc54f6d2c5a7c33d35e27241a4349f917bb07ec,"4 pages: text on 0/2, raster on 1/3"
eval/pdf-inspector-eval/corpus-extra/gen-mixed-02.pdf,"generated (build_corpus_extra.py, label by construction)",gen_mixed,Mixed,classifiable,9055266f379a9910ff22e5c1beb37bcfcffdf8a7a16474f425407590b75c46f5,"4 pages: text on 0/2, raster on 1/3"
eval/pdf-inspector-eval/corpus-extra/gen-mixed-03.pdf,"generated (build_corpus_extra.py, label by construction)",gen_mixed,Mixed,classifiable,e674b813dc4df0438128a1142f88ebfaf19e5038adbe3f07d4bab69eeae9b4a2,"4 pages: text on 0/2, raster on 1/3"
eval/pdf-inspector-eval/corpus-extra/gen-mixed-04.pdf,"generated (build_corpus_extra.py, label by construction)",gen_mixed,Mixed,classifiable,7153fe5369a3ee9ed317aa1dd687cc2b44f48ebc5918f1f0a459a3a467d60e37,"4 pages: text on 0/2, raster on 1/3"
eval/pdf-inspector-eval/corpus-extra/gen-imagedeck-00.pdf,"generated (build_corpus_extra.py, label by construction)",gen_imagedeck,ImageBased,classifiable,6d1416c00bf5dce57c98f346e2babf8c41afe01ddcee01c2a3ab767626fb1af1,"4 slide pages, 3 figures + <100 char caption each"
eval/pdf-inspector-eval/corpus-extra/gen-imagedeck-01.pdf,"generated (build_corpus_extra.py, label by construction)",gen_imagedeck,ImageBased,classifiable,535a30c5e47799b8afd11fba34a3b66376abaca2c0fbb8efdd1c8b370eec632e,"4 slide pages, 3 figures + <100 char caption each"
eval/pdf-inspector-eval/corpus-extra/gen-imagedeck-02.pdf,"generated (build_corpus_extra.py, label by construction)",gen_imagedeck,ImageBased,classifiable,bdeeb83d98233860cb6e7b5f5b4cd03461ded8e089fd801c97abcf0b02852173,"4 slide pages, 3 figures + <100 char caption each"
eval/pdf-inspector-eval/corpus-extra/gen-imagedeck-03.pdf,"generated (build_corpus_extra.py, label by construction)",gen_imagedeck,ImageBased,classifiable,f831e7d59a9b72e84343c1a2466b434a642c4518535edabe6a496dbf496d70c2,"4 slide pages, 3 figures + <100 char caption each"
eval/pdf-inspector-eval/corpus-extra/gen-imagedeck-04.pdf,"generated (build_corpus_extra.py, label by construction)",gen_imagedeck,ImageBased,classifiable,b137331051025d54ee584eb9abd6993bd7217a316059edecefe27e93f74c4433,"4 slide pages, 3 figures + <100 char caption each"
eval/pdf-inspector-eval/corpus-extra/gen-searchable-00.pdf,"generated (build_corpus_extra.py, label by construction)",gen_searchable_scan,TextBased,classifiable,d5240fa4ac1b3c101bc01fb3884f9fb6acd6a631d175c4a817a9b654e922d782,raster + invisible text layer (render_mode=3); complete text extractable
eval/pdf-inspector-eval/corpus-extra/gen-searchable-01.pdf,"generated (build_corpus_extra.py, label by construction)",gen_searchable_scan,TextBased,classifiable,228fc205beb7aa78a15f9dc8c33855f3ec757178a7e6506145dd93c7d2d74111,raster + invisible text layer (render_mode=3); complete text extractable
eval/pdf-inspector-eval/corpus-extra/gen-searchable-02.pdf,"generated (build_corpus_extra.py, label by construction)",gen_searchable_scan,TextBased,classifiable,29daae60aabce2d29923ca6faf20ee49ad024c7887470d863a976f8c7278b8c9,raster + invisible text layer (render_mode=3); complete text extractable
eval/pdf-inspector-eval/corpus-extra/gen-huge-500p.pdf,"generated (build_corpus_extra.py, label by construction)",gen_huge,TextBased,classifiable,3e17a70217a171357a9c00c1dde9496417e884794a8c7129914fcc99a1a7dfaf,500 digital text pages — sampling/speed probe
eval/pdf-inspector-eval/corpus-extra/gen-encrypted.pdf,"generated (build_corpus_extra.py, label by construction)",gen_encrypted,encrypted,probe,40734c2820e5b6fa1d123b30b4f6debcce8e66d22c0e5d1c52f37a385e03eaf0,"AES-256, user password set — error-behavior probe"
```

---

*Report generated 2026-09-02 by cto-agent dispatch 7d3b8f6b27cb1630 (evolve-product S3
BUILD). Not committed from the container per dispatch constraints — the conductor commits
this report; a Mac-side follow-up materializes §11 into the pdfmux code repo at
`eval/pdf-inspector-eval/` per PRD §2/§6.*
