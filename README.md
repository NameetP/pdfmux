# pdfmux

[![CI](https://github.com/NameetP/pdfmux/actions/workflows/ci.yml/badge.svg)](https://github.com/NameetP/pdfmux/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pdfmux)](https://pypi.org/project/pdfmux/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/pdfmux)](https://pypi.org/project/pdfmux/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/pdfmux)](https://pypi.org/project/pdfmux/)

PDF extraction that checks its own work. Built for LLM pipelines.

```
PDF ──→ pdfmux ──→ Markdown / JSON
         │
         ├─ fast extract every page
         ├─ audit each page (good / bad / empty)
         ├─ re-extract bad pages with OCR
         ├─ merge → clean → confidence score
         └─ extract tables, key-values, normalize dates/amounts
```

Most PDF extractors run once and hope for the best. pdfmux extracts, audits every page, and re-extracts the ones that came out wrong — automatically.

## Quick Start

```bash
pip install pdfmux

pdfmux invoice.pdf
# ✓ invoice.pdf → invoice.md (2 pages, 95% confidence, via pymupdf4llm)
```

No config, no flags, no API keys needed.

## Install

```bash
# core — handles digital PDFs instantly (the vast majority)
pip install pdfmux

# add OCR for scanned/image-heavy pages (~200MB, CPU-only)
pip install "pdfmux[ocr]"

# add table extraction (Docling — 97.9% table accuracy)
pip install "pdfmux[tables]"

# add LLM fallback for hardest cases (Gemini Flash)
pip install "pdfmux[llm]"

# everything
pip install "pdfmux[all]"
```

Requires Python 3.11+.

## How It Works

### Multi-pass extraction

Every PDF goes through a multi-pass pipeline. This is what makes pdfmux different.

```
Pass 1 — Fast extract + audit
  For each page:
    ├─ Extract text with PyMuPDF (instant)
    ├─ Count characters + images
    └─ Classify: "good" / "bad" / "empty"

  All pages good? → done. Zero overhead.

Pass 2 — Selective OCR (only bad pages)
  For each bad/empty page:
    ├─ Try RapidOCR  (~200MB, CPU, Apache 2.0)
    ├─ Try Surya OCR  (fallback, heavier)
    └─ Try Gemini LLM (fallback, API)

  Smart comparison:
    ├─ "bad" page (some text): only use OCR if it got MORE text
    └─ "empty" page (no text): accept any OCR result >10 chars

Pass 3 — Merge + score
  ├─ Combine good pages + OCR'd pages in order
  ├─ Clean text (broken words, control chars, spacing)
  └─ Confidence score (honest — reflects actual quality)
```

**The fast path is free.** Digital PDFs pass through in ~0.01s/page with zero OCR overhead. The audit step adds negligible cost. You only pay for OCR on pages that actually need it.

### Detection

pdfmux opens each PDF with PyMuPDF and classifies it:

```
Per page:
  ├─ Has >50 chars of text?             → digital
  ├─ Has images but no/little text?     → graphical (image-heavy)
  └─ No text, no images?                → empty

Document level:
  ├─ ≥80% digital pages                 → digital PDF
  ├─ ≥80% scanned pages                 → scanned PDF
  ├─ Image-heavy pages detected         → graphical PDF
  └─ Mix of types                       → mixed PDF

Table detection:
  ├─ Ruled line patterns (≥3 horiz + ≥2 vert lines)
  └─ Tab-separated or aligned text patterns
```

### Routing

```
classify(pdf)
  │
  ├─ quality=fast     → PyMuPDF only (instant, free)
  ├─ quality=high     → Gemini Flash → OCR → PyMuPDF
  │
  └─ quality=standard (default):
       ├─ has tables (not graphical) → Docling → PyMuPDF fallback
       └─ everything else            → multi-pass pipeline
```

If an optional extractor isn't installed, pdfmux silently falls back to the next best option. No errors, no config.

### Extractors

| Tier | Extractor | What it handles | Speed | Size | Install |
|------|-----------|----------------|-------|------|---------|
| Fast | PyMuPDF / pymupdf4llm | Digital PDFs with clean text | 0.01s/page | Base | Base |
| OCR | RapidOCR (PaddleOCR v4) | Scanned / image-heavy pages | 0.5-2s/page | ~200MB | `pdfmux[ocr]` |
| Tables | Docling | Table-heavy documents | 0.3-3s/page | ~500MB | `pdfmux[tables]` |
| OCR Heavy | Surya OCR | Scanned PDFs (legacy, GPU) | 1-5s/page | ~5GB | `pdfmux[ocr-heavy]` |
| LLM | Gemini 2.5 Flash | Complex layouts, handwriting | 2-5s/page | API | `pdfmux[llm]` |

### Confidence scoring

Every result includes an honest confidence score:

- **95-100%** — clean digital text, fully extractable
- **80-95%** — good extraction, minor OCR noise on some pages
- **50-80%** — partial extraction, some pages couldn't be recovered
- **<50%** — significant content missing, warnings included

When confidence is below 80%, pdfmux tells you exactly what went wrong and how to fix it (e.g., "Install `pdfmux[ocr]` for better results on 6 image-heavy pages").

## Python API

```python
import pdfmux

# Simple text extraction → Markdown string
text = pdfmux.extract_text("report.pdf")
print(text[:200])

# Structured extraction → dict with locked schema
data = pdfmux.extract_json("report.pdf")
print(f"{data['page_count']} pages, {data['confidence']:.0%}")
print(f"OCR pages: {data['ocr_pages']}")

# LLM-ready chunks → list of dicts with token estimates
chunks = pdfmux.load_llm_context("report.pdf")
for c in chunks:
    print(f"{c['title']}: {c['tokens']} tokens (pages {c['page_start']}-{c['page_end']})")
```

All three functions accept `quality="fast"`, `"standard"` (default), or `"high"`.

### Types & Errors

Every object in the pipeline is typed and immutable. All types and errors are exported from the top-level package.

```python
from pdfmux import (
    # Enums
    Quality,              # FAST, STANDARD, HIGH
    OutputFormat,         # MARKDOWN, JSON, CSV, LLM
    PageQuality,          # GOOD, BAD, EMPTY

    # Data objects (frozen dataclasses)
    PageResult,           # Single page: text, page_num, confidence, quality, extractor
    DocumentResult,       # Full document: pages, source, confidence, extractor_used
    Chunk,                # Section-aware chunk: title, text, page_start, page_end, tokens

    # Errors
    PdfmuxError,          # Base — catch this to handle all pdfmux errors
    FileError,            # File not found, unreadable, not a PDF
    ExtractionError,      # Extraction failed
    ExtractorNotAvailable,# Requested extractor not installed
    FormatError,          # Invalid output format
    AuditError,           # Audit could not complete
)
```

Catch broad or narrow:

```python
try:
    text = pdfmux.extract_text("report.pdf")
except pdfmux.ExtractorNotAvailable as e:
    print(f"Missing dependency: {e}")
except pdfmux.PdfmuxError as e:
    print(f"pdfmux error: {e}")
```

Stream pages with bounded memory:

```python
from pdfmux.extractors import get_extractor

ext = get_extractor("fast")
for page in ext.extract("large-500-pages.pdf"):  # Iterator[PageResult]
    process(page.text)  # bounded memory, even on 500-page PDFs
```

## CLI Usage

### Convert a single file

```bash
pdfmux invoice.pdf
# ✓ invoice.pdf → invoice.md (2 pages, 95% confidence, via pymupdf4llm)
```

### With OCR installed (image-heavy PDFs)

```bash
pdfmux pitch-deck.pdf
# ✓ pitch-deck.pdf → pitch-deck.md (12 pages, 85% confidence, 6 pages OCR'd, via pymupdf4llm + rapidocr)
```

### Output location

```bash
pdfmux report.pdf -o ./converted/report.md
```

### Batch convert

```bash
pdfmux ./docs/ -o ./output/
# Converting 12 PDFs from ./docs/...
#   ✓ invoice.pdf → invoice.md (95%)
#   ✓ contract.pdf → contract.md (92%)
#   ✓ scan.pdf → scan.md (87%)
# Done: 12 converted, 0 failed
```

### Output formats

```bash
# markdown (default)
pdfmux report.pdf

# json — structured output with metadata
pdfmux report.pdf -f json

# llm — section-aware chunks with token estimates
pdfmux report.pdf -f llm

# csv — extracts tables only
pdfmux data.pdf -f csv
```

### Quality presets

```bash
# fast — PyMuPDF only, no ML, no audit (instant, free)
pdfmux report.pdf -q fast

# standard — multi-pass pipeline (default)
pdfmux report.pdf -q standard

# high — use LLM for everything (slow, costs ~$0.01/doc)
pdfmux report.pdf -q high
```

### Diagnostics

```bash
# check what's installed
pdfmux doctor
# ┌──────────────┬─────────────┬─────────┬──────────────────────────────┐
# │ Extractor    │ Status      │ Version │ Install                      │
# ├──────────────┼─────────────┼─────────┼──────────────────────────────┤
# │ PyMuPDF      │ ✓ installed │ 1.25.3  │                              │
# │ RapidOCR     │ ✓ installed │ 3.0.6   │                              │
# │ Docling      │ ✗ missing   │ —       │ pip install pdfmux[tables]   │
# └──────────────┴─────────────┴─────────┴──────────────────────────────┘

# benchmark all extractors on a file
pdfmux bench report.pdf
# ┌──────────────┬────────┬────────────┬─────────────┬──────────────────────┐
# │ Extractor    │   Time │ Confidence │      Output │ Status               │
# ├──────────────┼────────┼────────────┼─────────────┼──────────────────────┤
# │ PyMuPDF      │  0.02s │        95% │ 3,241 chars │ ✓                    │
# │ Multi-pass   │  0.03s │        95% │ 3,241 chars │ ✓ all pages good     │
# │ RapidOCR     │  4.20s │        88% │ 2,891 chars │ ✓                    │
# └──────────────┴────────┴────────────┴─────────────┴──────────────────────┘
```

### Analyze a PDF

```bash
pdfmux analyze report.pdf
# report.pdf — 12 pages
#
# ┌──────┬────────────┬────────────────────────┬────────┐
# │ Page │ Type       │ Quality                │  Chars │
# ├──────┼────────────┼────────────────────────┼────────┤
# │    1 │ digital    │ good → fast extraction │  1,204 │
# │    2 │ graphical  │ bad → needs OCR        │     42 │
# │    3 │ digital    │ good → fast extraction │  2,108 │
# └──────┴────────────┴────────────────────────┴────────┘
#
#   Confidence: 91%
#   OCR pages:  2
#   Extractor:  pymupdf4llm + rapidocr (1 page re-extracted)
```

### Other options

```bash
# show confidence score in output
pdfmux report.pdf --confidence

# print to stdout instead of file
pdfmux report.pdf --stdout
```

### All CLI options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | Same dir, `.md` ext | Output file or directory |
| `--format` | `-f` | `markdown` | Output format: `markdown`, `json`, `csv`, `llm` |
| `--quality` | `-q` | `standard` | Quality: `fast`, `standard`, `high` |
| `--schema` | `-s` | `none` | JSON schema file or preset for structured extraction |
| `--confidence` | | `false` | Include confidence score in output |
| `--stdout` | | `false` | Print to stdout instead of writing file |

## Output Formats

### Markdown (default)

Clean markdown optimized for LLM consumption:

```markdown
# Quarterly Report

Revenue for Q3 increased by 15% year-over-year...

## Financial Summary

| Metric | Q3 2025 | Q3 2024 |
|--------|---------|---------|
| Revenue | $12.3M | $10.7M |
| Profit | $3.1M | $2.4M |
```

### JSON

Structured output with metadata:

```json
{
  "source": "report.pdf",
  "converter": "pdfmux",
  "extractor": "pymupdf4llm + rapidocr (3 pages re-extracted)",
  "page_count": 12,
  "confidence": 0.91,
  "ocr_pages": [2, 5, 8],
  "warnings": [],
  "content": "# Quarterly Report\n\nRevenue for Q3...",
  "pages": [
    { "page": 1, "content": "# Quarterly Report..." },
    { "page": 2, "content": "## Financial Summary..." }
  ]
}
```

### LLM (chunked JSON)

Section-aware chunks with token estimates, designed for RAG pipelines:

```json
{
  "document": "report.pdf",
  "chunks": [
    {
      "title": "Quarterly Report",
      "text": "Revenue for Q3 increased by 15%...",
      "page_start": 1,
      "page_end": 2,
      "tokens": 312,
      "confidence": 0.95
    },
    {
      "title": "Financial Summary",
      "text": "| Metric | Q3 2025 | Q3 2024 |...",
      "page_start": 3,
      "page_end": 3,
      "tokens": 156,
      "confidence": 0.95
    }
  ]
}
```

### CSV

Extracts tables from the document:

```csv
Metric,Q3 2025,Q3 2024
Revenue,$12.3M,$10.7M
Profit,$3.1M,$2.4M
```

Raises an error if no tables are found.

## Structured Extraction

*New in v1.1.0.* Extract structured data from invoices, bank statements, and forms — no LLM, no cloud, no cost.

pdfmux auto-detects key-value pairs (colon-separated, whitespace-aligned, dot-leader patterns), extracts tables as typed JSON, and normalizes dates, amounts, and rates into clean values.

### CLI

```bash
# JSON output with auto-detected structure
pdfmux statement.pdf -f json

# Schema-guided extraction — map to your own fields
pdfmux invoice.pdf --schema invoice.schema.json

# Use a built-in preset
pdfmux statement.pdf --schema bank-statement
```

When `--schema` is provided, the format auto-switches to JSON. Fields are matched using fuzzy string similarity — no exact key names required.

### Python API

```python
import pdfmux

data = pdfmux.extract_json("statement.pdf")
# data["pages"][0]["key_values"]  → extracted label: value pairs
# data["pages"][0]["tables"]      → headers + rows as structured JSON
```

### What gets extracted

**Key-value pairs** — detected from `Label: Value`, `Label    Value` (whitespace-aligned), and `Label.......Value` (dot-leader) patterns:

```json
{"key": "Statement Date", "value": "2026-02-28", "page_num": 0}
```

**Tables** — headers and rows as typed arrays:

```json
{
  "headers": ["Date", "Description", "Amount"],
  "rows": [["2026-02-01", "Payment received", "1,234.50"]]
}
```

**Normalized values** — dates become ISO 8601, amounts get parsed with currency and direction, rates get period detection:

```json
{
  "amount": 1234.50,
  "direction": "debit",
  "currency": "AED"
}
```

### Schema-guided mapping

Pass a JSON Schema and pdfmux maps extracted data to your fields using fuzzy matching + type coercion. Array fields map from tables, scalar fields map from key-value pairs. No LLM required.

```json
{
  "properties": {
    "invoice_date": {"type": "string", "format": "date"},
    "total_amount": {"type": "number"},
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "amount": {"type": "number"}
        }
      }
    }
  }
}
```

## MCP Server

pdfmux includes a built-in MCP (Model Context Protocol) server so AI agents can read PDFs natively. Agents receive confidence scores, warnings, and structured extraction data (key-value pairs, tables, normalized values) alongside the text.

```bash
pdfmux serve
```

### Claude Desktop / Cursor

Add to your config:

```json
{
  "mcpServers": {
    "pdfmux": {
      "command": "pdfmux",
      "args": ["serve"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add pdfmux -- pdfmux serve
```

### Tools

The server exposes three tools:

```json
{
  "name": "convert_pdf",
  "description": "Convert a PDF to AI-readable Markdown",
  "parameters": {
    "file_path": "string — absolute path to the PDF",
    "format": "string — markdown (default)",
    "quality": "string — fast | standard | high (default: standard)"
  }
}
```

When confidence is below 80% or there are warnings, the response includes extraction metadata (confidence score, extractor used, OCR page numbers, actionable warnings).

## Examples

See the [`examples/`](examples/) directory for runnable scripts:

- [`basic_usage.py`](examples/basic_usage.py) — extract_text, extract_json, load_llm_context
- [`batch_processing.py`](examples/batch_processing.py) — directory processing with progress
- [`mcp_agent.py`](examples/mcp_agent.py) — MCP config and tool examples

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Only for `pdfmux[llm]` | Google Gemini API key for LLM extraction |
| `GOOGLE_API_KEY` | Alternative | Alternative env var for Gemini API key |

No environment variables are needed for the base install or the `tables`/`ocr` extras.

## Why Not Just Use X?

| Tool | Good at | Limitation |
|------|---------|-----------|
| Marker | GPU ML extraction | Overkill for digital PDFs, needs GPU |
| Docling | Tables (97.9% accuracy) | Slow on non-table documents |
| pymupdf4llm | Fast digital text | Can't handle scanned or image-heavy layouts |
| MinerU | Full ML pipeline | Heavy, complex setup |
| MarkItDown | Wide format support | Not optimized for any specific PDF type |
| **pdfmux** | **Self-healing extraction** | Audits every page, re-extracts bad ones |

pdfmux doesn't compete with these tools — it orchestrates them. The key insight: no single extractor wins on everything. pdfmux routes each page to the right one, verifies the result, and re-extracts if needed.

## Project Structure

```
src/pdfmux/
├── __init__.py         # Public API: extract_text, extract_json, load_llm_context + type/error re-exports
├── py.typed            # PEP 561 marker — mypy/pyright recognize pdfmux as typed
├── types.py            # Frozen dataclasses + enums: Quality, OutputFormat, PageResult, DocumentResult, Chunk
├── errors.py           # Exception hierarchy: PdfmuxError → FileError, ExtractionError, FormatError, AuditError
├── pipeline.py         # Multi-pass routing + merge + process_batch() + security limits
├── detect.py           # PDF type classification + layout detection
├── audit.py            # 5-check per-page confidence scoring + quality classification
├── regions.py          # Region OCR — surgical image extraction for bad pages
├── parallel.py         # Parallel OCR dispatch with thread pool
├── chunking.py         # Section-aware splitting + token estimation
├── kv_extract.py       # Key-value pair extraction (colon, whitespace, dot-leader)
├── normalize.py        # Date/amount/rate normalization (pure Python)
├── schema.py           # Schema-guided extraction (fuzzy matching, type coercion)
├── postprocess.py      # Text cleanup
├── mcp_server.py       # MCP server (stdio JSON-RPC) with path restrictions
├── cli.py              # Typer CLI (convert, analyze, serve, doctor, bench, version)
├── extractors/
│   ├── __init__.py     # Extractor protocol + @register decorator + priority-ordered registry
│   ├── fast.py         # PyMuPDF — handles 90% of PDFs (priority 10)
│   ├── rapid_ocr.py    # RapidOCR — lightweight OCR (~200MB, priority 20)
│   ├── tables.py       # Docling — table-heavy docs (priority 40)
│   ├── ocr.py          # Surya — legacy heavy OCR (priority 30)
│   └── llm.py          # Gemini Flash — hardest cases (priority 50)
├── integrations/
│   ├── langchain.py    # PDFMuxLoader for LangChain
│   └── llamaindex.py   # PDFMuxReader for LlamaIndex
└── formatters/
    ├── markdown.py     # Markdown output
    ├── json_fmt.py     # JSON + LLM chunked output
    └── csv_fmt.py      # CSV output (tables only)
```

## Development

```bash
git clone https://github.com/NameetP/pdfmux.git
cd pdfmux
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run tests (151 tests)
pytest

# lint
ruff check src/ tests/
ruff format src/ tests/
```

## License

[MIT](LICENSE)
