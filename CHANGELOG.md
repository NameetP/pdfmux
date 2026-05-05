# Changelog

## 1.6.4 (2026-05-05)

Additive release. Two new things, no breaking changes, no defaults change. This is the OSS half of the **Verified Extraction Manifest (VEM)** standard — the `audit` command produces the comparison artifact that VEM 1.0 standardizes.

### Added

- **`pdfmux audit --against <other.csv|other.json> --on <pdf-dir>`** — diff your current extractor's output against pdfmux on the same PDFs.
  - Reads `--against` as either CSV (columns `filename,text` or aliases `file,content`) or JSON (`{filename: text}`).
  - Runs pdfmux on every PDF named in `--against` that exists in `--on`.
  - Computes per-document **word-set Jaccard overlap** between the two extractors.
  - Flags documents with `overlap < --overlap-threshold` (default `0.70`) **OR** `pdfmux_confidence < --confidence-threshold` (default `0.50`).
  - Writes a 7-column CSV: `filename, my_extractor_chars, pdfmux_chars, jaccard_overlap, pdfmux_confidence, recommendation, error`.
  - Exit codes: `0` everything clean, `2` usage error, `3` anything flagged (matches `--strict` convention).

  Use case: "diff our output against your current extractor on 100 of your own PDFs — if we agree on every document, you don't need us. If we disagree on more than 2%, those are the silent failures already in your pipeline." This is the public artifact behind the [VEM 1.0 spec](https://verifiedextraction.org/spec/v1).

- **OSS → cloud funnel-upsell line** on `convert` completion. A single `[dim]`-styled line prints to stdout after a successful conversion, pointing to the free tier and the open VEM spec. Suppress with `PDFMUX_NO_UPSELL=1`. Skipped automatically when stdout isn't a TTY (so it never pollutes piped output) and when writing to stdout via `--output -`.

### Tests

- `tests/test_audit_cli.py` — 8 tests covering CSV/JSON inputs, alternate column names, unsupported extensions, no-filename-overlap, empty directory, exit-code contracts, and the documented column set.

Test count: 678 passing (up from 670).

## 1.6.3 (2026-05-02)

Correctness patch — the bug behind the silent-failure incident that prompted 1.6.1. **No defaults change.** Every existing flag and CLI invocation behaves identically. Confidence numbers are now correct on documents where they were previously inflated.

### Fixed

- **`audit.compute_document_confidence` was returning 1.0 on documents with empty extractions.** The function did a content-weighted average of the per-page `confidence` value the extractor wrote at yield time — always `1.0` with the comment "audit will reassess". The reassessment never happened, so blank pages, HTML files renamed to `.pdf`, single-character bodies, and image-only pages with no OCR all returned document confidence `1.0`.

  Fix in `src/pdfmux/audit.py`: re-score every page with `score_page(p.text, p.image_count)` before averaging, and stop flooring the per-page weight at `1` (which let blank pages register full weight in the denominator).

  This is the bug behind the silent failures in [the 433-PDF batch retro](https://pdfmux.com/blog/when-our-own-pdf-extractor-failed-silently/). `--strict --min-confidence 0.20` (shipped in 1.6.1) could not catch the eleven silent failures because the audit didn't know the pages were empty.

### Added

- **`eval/` directory at the project root** — a self-contained confidence calibration harness:
  - `eval/build_fixtures.py` generates 50 labeled PDFs from a fixed seed (clean digital, multi-page, table-heavy, Arabic, 0-byte, HTML-as-PDF, heavily/lightly truncated, blank pages, micro-text, image-only-no-OCR).
  - `eval/run_eval.py` runs pdfmux on each fixture and writes `outputs/raw_scores.csv`.
  - `eval/calibrate.py` computes ROC and recommends thresholds at fixed precision targets (P >= 0.95 for the strict gate, P >= 0.80 for the warning gate).
  - `eval/README.md` documents the workflow and the May 2026 calibration result that drives the 1.7 default.

  The eval set was the instrument that surfaced the audit bug — without it, the bug was invisible. The first calibration run produced precision flat at 0.683 across every threshold, which is the smoking gun.

### Calibration headline (informational, not yet a default)

After the audit fix, on the 50-fixture eval set:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.50 | 0.821 | 0.821 | 0.821 |
| **0.75** | **1.000** | **0.714** | **0.833** |

`0.75` is the recommended default for `--min-confidence` when 1.7 ships breaking-default-strict on `pdfmux convert <dir>`. Not enabled by default in this release.

## 1.6.2 (2026-05-01)

Regression-guard release. No code behavior changes. Adds 11 contract tests for the real-world failure modes seen in the 433-PDF batch run that prompted 1.6.1.

### Added
- `tests/test_real_world_failures.py` — 11 behavioral-contract tests covering:
  - **Truncated PDFs** (the four `pypdf: Stream has ended unexpectedly` cases from the v1 batch). Pdfmux must either recover or raise — never silently return empty Markdown.
  - **Non-ASCII filenames** (CJK + full-width punctuation, e.g. `（原版）`). `extract_text` and `batch_extract` must accept these without shell-quoting issues.
  - **Arabic-only PDFs.** The BiDi pipeline must not crash on RTL text.
  - **0-byte files.** Must raise a named `PdfmuxError`, never silently return empty.
  - **HTML files renamed to `.pdf`.** Must error cleanly OR return text without HTML markup — never pass through `<html>...</html>` as if it were content.
  - **Missing files.** Must raise `FileError`, not bare `FileNotFoundError`.
  - **Batch isolation.** A bad file in `batch_extract` must yield an exception for that file without poisoning the rest of the batch.

Test count: 670 passing (up from 659).

## 1.6.1 (2026-05-01)

Field-driven patch release. Triggered by a real-world 433-PDF batch run where the first invocation silently dropped 16 documents — the exact failure mode pdfmux's brand promises to prevent. All changes are additive; no breaking defaults. The retro and full plan are at <https://github.com/NameetP/pdfmux/blob/main/CHANGELOG.md> and the corresponding [blog post](https://pdfmux.com/blog/).

### Added — surface signals that already exist
- **`pdfmux convert --strict --min-confidence FLOAT`** — exits with code `3` if any document confidence falls below the threshold. Exit codes are now documented: `0` ok, `1` runtime error, `2` usage error, `3` strict gate failed.
- **stderr WARNING line** for every document with confidence < 0.50, regardless of `--strict`. Makes silent low-quality batches visible in CI logs.
- **`manifest.json`** written to the output directory at the end of every batch `convert <dir>` run. Includes per-document confidence, extractor used, OCR pages, cost, warnings, and a confidence breakdown (high ≥0.80 / medium 0.50–0.80 / low <0.50). Schema v1.0.
- **`pdfmux.batch_extract(paths, **kwargs)`** — public Python API over `process_batch`. Use this instead of shelling out to `pdfmux convert` in a loop.
- **`pdfmux doctor --check <dir>`** — samples PDFs from a directory, classifies them, and recommends missing extras. Catches "23% of your batch is scanned, install `pdfmux[ocr]`" before you waste a batch run.
- **RapidOCR warnings translated** into pdfmux-namespaced INFO messages with file + page context. The bare `[RapidOCR] main.py:132: The text detection result is empty` lines are gone.

### Removed
- **`pdfmux.ml_headings` and `models/heading_classifier.pkl`.** The ML heading classifier required `sklearn` (not a base dep), printed `Failed to load ML heading model` 24+ times per batch, and produced no measurable lift over the heuristic font-size fallback. Net: -250 LOC, no behavior change on real-world PDFs.

### Fixed
- `pdfmux.__version__` was stale at `1.5.1`; now matches `pyproject.toml`.

### Docs
- README leads Python users with `batch_extract` for batch use cases.
- `pdfmux[ocr]` promoted from "optional extra" to recommended-default for any real batch.
- Note added: don't wrap pdfmux with your own pypdf/pdfplumber fallback — PyMuPDF tolerates malformed PDFs that pypdf rejects.

## 1.6.0 (2026-04-30)

### Added — Extraction backends
- **Mistral OCR** as a paid extraction backend ($0.002/page, 96.6% table accuracy on internal benches). Optional dep: `pdfmux[llm-mistral]`.
- **Marker** neural extractor (`pdfmux[marker]`) — strong on academic papers and dense layouts. Models cached as module-level singletons to amortize warm-up.
- **Gemma 4 27B IT** as a vision LLM provider via the GeminiAPI OpenAI-compat endpoint. Reuses `GEMINI_API_KEY`. Native Arabic OCR.

### Added — Arabic / RTL support
- **BiDi post-processing** for Arabic and Hebrew using `python-bidi`. Markdown-aware: preserves heading prefixes, list markers, and code fences while reordering RTL text.
- **Arabic detection** in the document classifier (samples first 20 pages). New `arabic` page type wired into the routing matrix → Gemma fallback chain.
- `has_arabic` flag on `DocumentResult`.

### Added — Caching & streaming
- **Smart result cache** keyed by `(file_hash, quality, format, schema)` at `~/.cache/pdfmux/results/`. 30-day TTL, 1 GB max, LRU eviction. New `--no-cache` and `--clear-cache` flags.
- **Streaming extraction** as NDJSON events (`classified` → `page` → `warning` → `complete`). New `pdfmux stream` CLI command and `extract_streaming` MCP tool.

### Added — Developer experience
- **Configuration profiles** at `~/.config/pdfmux/profiles.yaml`. Built-ins: `invoices`, `receipts`, `papers`, `contracts`, `bulk-rag`. New `pdfmux profiles list/show/save/delete` sub-commands and `--profile` flag on `convert`.
- **Watch mode**: `pdfmux watch <dir>` auto-converts new and changed PDFs (via `watchdog`).
- **Cost estimation**: `pdfmux estimate <pdf>` predicts cost before running.
- **Diff command**: `pdfmux diff a.pdf b.pdf` produces a Levenshtein-style comparison of two extractions.
- **Better error messages** — every `PDFMuxError` carries `.user_message`, `.suggestion`, and a copy-pasteable `.reproduce_cmd`.
- **Retry with backoff** — `@with_retry(max_attempts=3, backoff_base=2.0)` decorator applied to every LLM provider's `extract_page()`. Honors `Retry-After` headers.

### Tests
- 9 new test modules. **659 tests passing** (3 skipped) — up from 481.

## 1.5.2 (2026-04-26)

GEO metadata refresh: keywords, description, README opener tuned for AI-engine surfacing. No code changes.

## 1.5.1 (2026-04-17)

### Added
- MCP Registry ownership marker (`mcp-name: io.github.NameetP/pdfmux`) in README so the package can be claimed on registry.modelcontextprotocol.io. Unblocks aggregator ingest across PulseMCP and other downstream MCP directories.

### Notes
- Metadata-only release. No runtime, API, or benchmark changes vs. 1.5.0.

## 1.5.0 (2026-04-05)

### Changed
- **Benchmark score: 0.900 -> 0.905** (+0.5%) on opendataloader-bench. Now within 0.4% of the paid #1.
- **Table score (TEDS): 0.887 -> 0.911** (+2.7%) — image table OCR extracts tables embedded as images using RapidOCR with spatial clustering for row/column reconstruction.
- **Heading score (MHS): 0.844 -> 0.852** (+0.9%) — ML heading classifier (sklearn GradientBoosting, 212KB) as fallback when font-size heuristics find nothing. Consecutive heading merge ("# III." + "# Regulatory..." → "# III. Regulatory..."). Digit page-number filter using y-position.
- **Reading order (NID): 0.918 -> 0.920** — minor improvements from heading cleanup.
- 98 docs improved, 3 regressed across the 200-doc benchmark.

### Added
- `src/pdfmux/image_table_ocr.py` — OCR-based table extraction from image regions. Renders image at 300 DPI, runs RapidOCR, clusters text boxes into rows/columns, outputs markdown pipe table. Safety filters: 50%+ fill rate, 50%+ numeric density.
- `src/pdfmux/ml_headings.py` — ML heading classifier using sklearn GradientBoosting on 12 font features. Used as fallback when heuristic detection finds zero headings.
- `src/pdfmux/models/heading_classifier.pkl` — trained model (212KB, doc-level cross-validated).

## 1.4.0 (2026-03-23)

### Added
- **OpenDataLoader extractor** — integrate OpenDataLoader-PDF (by Hancom) as a backend extractor. Best-in-class reading order (0.94 NID) and table accuracy (0.93 TEDS). Auto-selected in standard mode when installed. Install with `pip install pdfmux[opendataloader]`. Requires Java 11+.
- New `pdfmux doctor` check for OpenDataLoader availability.
- New `pdfmux bench` row for OpenDataLoader comparison.
- pdfmux now acts as an **orchestrator** — automatically routes to the best available extractor per document type rather than being a single extraction engine.

## 1.3.0 (2026-03-21)

### Changed
- **Benchmark score: 0.867 -> 0.900** (+3.7%) on opendataloader-bench (200 real-world PDFs). Now #2 overall, beating docling (0.877) and every other open-source extractor.
- **Heading detection overhaul** — TOC page cleanup strips false headings from Contents pages. Sentence-ending filter prevents body text promotion. Soft heading fallback for docs with subtle font-size signals. Equation and running-header filters. Page-number filter.
- **Heading score (MHS): 0.739 -> 0.844** (+14.2%) — beats all extractors including paid tiers.
- **Reading order (NID): 0.910 -> 0.918** — smart quote normalization, italic/link stripping, accent normalization, Unicode cleanup.
- **Table score (TEDS): 0.884 -> 0.887** — relaxed table block threshold.
- **Postprocess hardening** — normalize smart quotes/dashes to ASCII, strip markdown links and footnote markers, strip italic underscores, remove zero-width chars, normalize heading levels to H1.
- Bold-line promotion now validates candidates (rejects sentences, figure captions).
- 95 docs improved, 2 regressed across the 200-doc benchmark.

## 1.2.0 (2026-03-15)

_Structured extraction, parallel OCR, Docling table overlay._

## 1.0.0 (2026-03-05)

### Added
- **LangChain loader** (`integrations/langchain.py`) — `PDFMuxLoader(path, quality)` wraps `load_llm_context()`, returns `list[Document]` with metadata. Lazy import of langchain-core.
- **LlamaIndex reader** (`integrations/llamaindex.py`) — `PDFMuxReader(quality)` wraps `load_llm_context()`, returns `list[Document]`. Lazy import of llama-index-core.
- **Security hardening** — file size limit (500MB), page count limit (10,000), configurable via `PDFMUX_MAX_FILE_SIZE_MB` and `PDFMUX_MAX_PAGES` env vars.
- **API stability tests** — signature-level tests lock function parameters, JSON schema fields, and `__all__` exports for the 1.x series.
- **New optional extras**: `pdfmux[langchain]`, `pdfmux[llamaindex]`, updated `pdfmux[all]`.
- 19 new tests: API stability (10 tests), integrations (6 tests), security (3 tests).

### Changed
- **Interface locked** — public API signatures frozen for 1.x (additive-only keyword args allowed). JSON schema frozen at 1.0.0.
- Classifier upgraded to "Production/Stable".
- JSON schema version bumped to `1.0.0`.
- Total test count: 151.

## 0.9.0 (2026-03-05)

### Added
- **Region OCR** (`regions.py`) — surgical OCR of image regions within pages. `detect_weak_regions()` finds images without overlapping text. `ocr_region()` clips and OCRs just the image area. `merge_region_text()` inserts results at correct y-position. "Bad" pages now preserve good text while recovering image content.
- **Enhanced MCP server** — 2 new tools: `analyze_pdf` (quick triage — classify + audit without extraction) and `batch_convert` (process entire directories). Error responses now include structured error codes.
- **`WeakRegion` type** (`types.py`) — frozen dataclass with `page_num`, `bbox`, `reason`. Exported from `pdfmux`.
- **Example scripts** (`examples/`) — `basic_usage.py`, `batch_processing.py`, `mcp_agent.py` with self-contained usage examples.
- 14 new tests: region OCR (8 tests) and enhanced MCP (6 tests).

### Changed
- Multi-pass pipeline now tries region OCR on "bad" pages before falling back to full-page OCR.
- JSON schema version bumped to `0.9.0`.
- Total test count: 132.

## 0.8.0 (2026-03-05)

### Added
- **Column detection** (`detect.py`) — `detect_layout(page)` detects multi-column PDFs by clustering text block x-positions with gap detection. Returns `PageLayout` with column count, boundaries, and reading order.
- **Layout-aware extraction** (`fast.py`) — `_needs_reorder()` samples first 5 pages; if multi-column detected, `_extract_with_layout()` reorders blocks column-by-column. Single-column PDFs take existing fast path with zero overhead.
- **Block-level scoring** (`audit.py`) — `score_block(text)` applies 3 lightweight quality checks (alphabetic ratio, word structure, encoding quality) at individual text block granularity.
- **`PageLayout` type** (`types.py`) — frozen dataclass with `columns`, `column_boundaries`, `reading_order`. Exported from `pdfmux`.
- 8 new tests: layout detection (5 tests) and block scoring (3 tests).

### Changed
- JSON schema version bumped to `0.8.0`.
- Total test count: 118.

## 0.7.0 (2026-03-05)

### Added
- **Structured error codes** — every exception now has a `.code` class attribute (`PDF_NOT_FOUND`, `PDF_CORRUPTED`, `EXTRACTION_ERROR`, `PARTIAL_EXTRACTION`, `NO_EXTRACTOR`, `FORMAT_ERROR`, `AUDIT_ERROR`, `OCR_TIMEOUT`). Backward-compatible: existing catch blocks still work.
- **`OCRTimeoutError`** — new exception for OCR timeout scenarios. Exported from `pdfmux`.
- **Provenance on chunks** — `Chunk` dataclass now carries `extractor` and `ocr_applied` fields. Propagated through `chunk_by_sections()` and `format_llm()` output.
- **JSON `error_code` field** — JSON output includes `error_code` (null on success). Per-page `ocr` boolean flag in pages array.
- **CLI logging** — `--verbose` (INFO), `--debug` (DEBUG), `--quiet` (ERROR only) flags on `pdfmux convert`.
- 17 new tests: error codes (13 tests) and provenance (4 tests).

### Changed
- `FileError` and `ExtractionError` accept optional `code=` keyword for specific error codes.
- LLM format output now includes `extractor` and `ocr_applied` per chunk.
- JSON schema version bumped to `0.7.0`.
- Total test count: 110.

## 0.6.0 (2026-03-05)

### Added
- **Parallel OCR dispatch** (`parallel.py`) — OCR re-extraction now runs across 4 threads via `ThreadPoolExecutor`. ONNX runtime releases the GIL during inference, giving real parallelism. Per-page timing and error isolation via `PageOCRResult` frozen dataclass.
- **OCR budget control** — Standard mode caps OCR at 30% of document pages. Prioritizes "bad" pages (some text) over "empty" pages. `quality=high` ignores the budget. Override with `PDFMUX_OCR_BUDGET` env var.
- **Windowed audit** — `audit_document()` now processes pages in windows of 50 instead of loading the entire document at once. Bounds memory on 500+ page PDFs.
- 7 new tests: parallel OCR dispatch (4 tests) and budget control logic (3 tests).

### Changed
- Multi-pass pipeline uses parallel OCR dispatch instead of serial page-by-page loop.
- JSON schema version bumped to `0.6.0`.
- Total test count: 93.

## 0.5.0 (2026-03-05)

### Added
- **Typed architecture** — 6 frozen dataclasses and enums in `types.py` (`Quality`, `OutputFormat`, `PageQuality`, `PageResult`, `DocumentResult`, `Chunk`). Every data flow in the pipeline now passes through typed, immutable objects.
- **Error hierarchy** — flat exception tree in `errors.py`: `PdfmuxError` base with `FileError`, `ExtractionError`, `ExtractorNotAvailable`, `FormatError`, `AuditError`. All exported from `import pdfmux`.
- **Streaming extractors** — all 5 extractors yield `Iterator[PageResult]`, one page at a time. Memory stays bounded even on 500-page PDFs (~135MB peak vs unbounded before).
- **Extractor protocol + registry** — `Extractor` Protocol class + `@register(name, priority)` decorator for auto-registration. Programmatic access via `get_extractor()`, `available_extractors()`, `extractor_names()`.
- **5-check confidence scoring** — per-page confidence computed from character density, alphabetic ratio, word structure, whitespace sanity, and encoding quality (mojibake detection). Content-weighted document average replaces the old heuristic scorer.
- **Concurrent batch processing** — `process_batch()` with `ThreadPoolExecutor`. Error isolation per file — one failure doesn't stop the batch. Used by CLI batch conversion.

### Changed
- JSON schema version bumped to `0.5.0`.
- All public types and errors exported from the top-level `pdfmux` package (`pdfmux.PageResult`, `pdfmux.FileError`, etc.).
- Extractors conform to a common `Extractor` protocol and register via decorator instead of hardcoded lookup.
- Confidence scoring is now deterministic and auditable — 5 named checks with individual scores, content-weighted aggregation.
- Extractor names simplified: `"pymupdf4llm (fast)"` → `"pymupdf4llm"`, `"docling (tables)"` → `"docling"`, etc.
- `detect.py` now raises `FileError` instead of `FileNotFoundError`/`ValueError` for consistency with the error hierarchy.

### Fixed
- Memory usage no longer scales linearly with page count during extraction (streaming architecture).

## 0.4.0 (2026-03-04)

### Added
- **Public Python API** — three importable functions: `extract_text()`, `extract_json()`, `load_llm_context()`. No more CLI-only usage.
- **Section-aware chunking** (`chunking.py`) — splits Markdown at heading boundaries with per-chunk page tracking and token estimates (chars/4). Powers `load_llm_context()` and `--format llm`.
- **LLM output format** — `pdfmux report.pdf -f llm` outputs chunked JSON with `{title, text, page_start, page_end, tokens, confidence}` per section. Designed for RAG pipelines and context windows.
- **`pdfmux analyze`** — per-page extraction breakdown showing page type (digital/graphical/scanned), quality (good/bad/empty), char count, confidence, and extractor used.
- **Locked JSON schema** — JSON output now includes `schema_version: "0.4.0"` and `ocr_pages` field for downstream stability.

### Changed
- **JSON output** now includes `schema_version` and `ocr_pages` fields in every response.
- **`--format` option** accepts `llm` in addition to `markdown`, `json`, `csv`.

## 0.3.0 (2026-03-04)

### Added
- **Multi-pass extraction** — fast extract → per-page audit → selective OCR → merge. All standard-mode PDFs now go through this pipeline. Zero overhead when all pages are good.
- **RapidOCR extractor** — lightweight OCR using PaddleOCR v4 models via ONNX runtime. ~200MB install, CPU-only, Apache 2.0 license. Replaces Surya as default `pdfmux[ocr]`.
- **Per-page quality auditing** (`audit.py`) — classifies each page as "good", "bad", or "empty" based on text density and image presence. Drives selective re-extraction.
- **Smart OCR comparison** — for "bad" pages (some text), only uses OCR if it extracts MORE text than fast extraction. For "empty" pages, any OCR text >10 chars is accepted.
- **OCR fallback chain** — RapidOCR → Surya → Gemini Flash LLM. Each step only processes pages the previous step couldn't recover.
- **`ocr_pages` tracking** — `ConversionResult` now reports which pages were re-extracted with OCR.
- **Multi-pass in bench** — `pdfmux bench` now includes a "Multi-pass" row showing the full pipeline result.
- **RapidOCR in doctor** — `pdfmux doctor` now checks for RapidOCR installation.

### Changed
- **`pdfmux[ocr]`** now installs RapidOCR + onnxruntime (~200MB) instead of Surya (~5GB). Surya moved to `pdfmux[ocr-heavy]`.
- **Routing simplified** — removed `_handle_graphical_pdf()` and `_handle_mixed_pdf()`. Multi-pass handles all PDF types uniformly.
- **Graphical + tables routing** — graphical PDFs no longer route to Docling even if table heuristics trigger. Multi-pass OCR is more valuable than table formatting for image-heavy content.
- **Confidence scoring** — OCR-recovered pages get a small penalty for OCR noise (max 15%) instead of the large "extraction_limited" penalty.

### Fixed
- Pitch decks and slide exports now get 70-85% confidence with OCR installed (was 30-55%)
- Digital PDFs maintain identical confidence and zero overhead through multi-pass fast path
- RapidOCR logging noise suppressed (model paths, engine info no longer pollute output)

## 0.2.2 (2026-03-04)

### Added
- **Graphical PDF detection** — detects image-heavy PDFs (pitch decks, infographics) and routes to OCR/LLM instead of fast extraction
- **Honest confidence scoring** — confidence now reflects actual extraction quality, not just text presence. Graphical PDFs with missing image content score lower.
- **Actionable warnings** — clear messages when extraction is limited, with specific `pip install pdfmux[ocr]` or `pdfmux[llm]` suggestions
- **MCP server quality metadata** — AI agents now receive confidence score and warnings alongside extracted text
- **Spaced-text cleanup** — fixes common PDF artifact where text renders as "W i t h  o v e r" → "With over"

### Fixed
- Detection no longer classifies image-heavy "digital" PDFs as fully extractable
- Confidence no longer reports 100% on graphical PDFs where image content was missed
- FastExtractor now falls back to raw fitz when pymupdf4llm returns empty (fixes certain PDF encodings)
- bench command now shows honest confidence that matches pipeline routing

## 0.2.1 (2026-03-04)

### Added
- **`pdfmux doctor`** — check installed extractors, versions, and API keys
- **`pdfmux bench`** — benchmark all available extractors on a PDF side by side

### Fixed
- Suppressed upstream pymupdf4llm "Consider using pymupdf_layout" noise from all commands

## 0.2.0 (2026-03-03)

First public release.

### Added
- **Smart routing** — auto-detect PDF type and pick the best extractor
- **PyMuPDF extractor** — digital PDFs at 0.01s/page
- **Docling extractor** — 97.9% table accuracy (optional: `pdfmux[tables]`)
- **Surya OCR extractor** — scanned PDF support (optional: `pdfmux[ocr]`)
- **Gemini Flash extractor** — complex layout fallback (optional: `pdfmux[llm]`)
- **Mixed PDF handling** — digital pages + scanned pages merged automatically
- **Output formats** — Markdown, JSON, CSV
- **Quality presets** — fast, standard, high
- **Batch conversion** — convert entire directories
- **MCP server** — built-in stdio server for AI agents
- **Confidence scoring** — text completeness, encoding quality, structure checks
- **Graceful fallback** — missing extractors fall back silently to next best
