# Changelog

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
