# Changelog

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
- **`pdfmux doctor`** — check installed extractors and API keys
- **`pdfmux bench`** — benchmark all extractors on a file side by side
- **Graceful fallback** — missing extractors fall back silently to next best
