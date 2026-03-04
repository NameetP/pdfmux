# Changelog

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
