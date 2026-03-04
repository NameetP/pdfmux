# pdfmux

[![CI](https://github.com/NameetP/pdfmux/actions/workflows/ci.yml/badge.svg)](https://github.com/NameetP/pdfmux/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pdfmux)](https://pypi.org/project/pdfmux/)

The smart PDF-to-Markdown router. One command, zero config.

```
PDF ──→ pdfmux ──→ Markdown
         │
         ├─ digital?  → PyMuPDF     (0.01s/pg, free)
         ├─ tables?   → Docling     (0.3s/pg, free)
         ├─ scanned?  → Surya OCR   (1-5s/pg, free)
         └─ complex?  → Gemini Flash (2-5s/pg, ~$0.01)
```

We don't convert PDFs. We route them to whichever tool converts them best.

90% of PDFs are digital — converted in milliseconds, for free.

## Quick Start

```bash
pip install pdfmux

pdfmux invoice.pdf
# ✓ invoice.pdf → invoice.md (2 pages, 95% confidence, via pymupdf4llm)
```

That's it. No config, no flags, no API keys needed.

## Install

```bash
# core (handles digital PDFs — the vast majority)
pip install pdfmux

# add table extraction (Docling — 97.9% table accuracy)
pip install pdfmux[tables]

# add scanned PDF support (Surya OCR)
pip install pdfmux[ocr]

# add LLM fallback for hardest cases (Gemini Flash)
pip install pdfmux[llm]

# everything
pip install pdfmux[all]
```

Requires Python 3.11+.

## Usage

### Convert a single file

```bash
pdfmux invoice.pdf
# ✓ invoice.pdf → invoice.md (2 pages, 95% confidence, via pymupdf4llm)
```

Output is written to the same directory with a `.md` extension by default.

### Specify output location

```bash
pdfmux report.pdf -o ./converted/report.md
```

### Batch convert a directory

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

# csv — extracts tables only
pdfmux data.pdf -f csv
```

### Quality presets

```bash
# fast — PyMuPDF only, no ML (instant, free)
pdfmux report.pdf -q fast

# standard — auto-detect and route (default)
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
# │ Docling      │ ✗ missing   │ —       │ pip install pdfmux[tables]   │
# └──────────────┴─────────────┴─────────┴──────────────────────────────┘

# benchmark all extractors on a file
pdfmux bench report.pdf
# ┌──────────────┬────────┬────────────┬─────────────┬───────────────┐
# │ Extractor    │   Time │ Confidence │      Output │ Status        │
# ├──────────────┼────────┼────────────┼─────────────┼───────────────┤
# │ PyMuPDF      │  0.02s │       100% │ 3,241 chars │ ✓             │
# │ Docling      │      — │          — │           — │ not installed │
# └──────────────┴────────┴────────────┴─────────────┴───────────────┘
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
| `--format` | `-f` | `markdown` | Output format: `markdown`, `json`, `csv` |
| `--quality` | `-q` | `standard` | Quality: `fast`, `standard`, `high` |
| `--confidence` | | `false` | Include confidence score in output |
| `--stdout` | | `false` | Print to stdout instead of writing file |

## How It Works

### Detection

pdfmux opens each PDF with PyMuPDF and classifies it by inspecting every page:

```
For each page:
  ├─ Has >50 chars of extractable text?  → digital
  ├─ Has embedded images but no text?    → scanned
  └─ Empty or minimal content?           → digital (empty page)

Classification:
  ├─ ≥80% digital pages  → digital PDF
  ├─ ≥80% scanned pages  → scanned PDF
  └─ Otherwise            → mixed PDF

Table detection:
  ├─ Check for ruled line patterns (≥3 horizontal + ≥2 vertical lines)
  └─ Check for tab-separated or multi-space aligned text patterns
```

### Routing

Based on classification, pdfmux picks the best extractor:

```
classify(pdf)
  │
  ├─ quality=fast? ────────────────→ PyMuPDF (always)
  ├─ quality=high? ────────────────→ Gemini Flash → Surya → PyMuPDF
  │
  └─ quality=standard (default):
       ├─ digital, no tables ──────→ PyMuPDF
       ├─ has tables ──────────────→ Docling → PyMuPDF fallback
       ├─ scanned ─────────────────→ Surya OCR → PyMuPDF fallback
       ├─ mixed ───────────────────→ PyMuPDF (digital pgs) + Surya (scanned pgs)
       └─ default ─────────────────→ PyMuPDF
```

If an optional extractor isn't installed, pdfmux silently falls back to the next best option. No errors, no config.

### Post-processing

After extraction, every result goes through:

1. **Cleanup** — remove control characters, fix broken hyphenation, normalize blank lines
2. **Confidence scoring** — text completeness, encoding quality, structure preservation, whitespace sanity
3. **Formatting** — heading normalization, list marker standardization, optional YAML frontmatter

### Extractors

| Tier | Extractor | What it handles | Speed | Cost | Install |
|------|-----------|----------------|-------|------|---------|
| Fast | PyMuPDF / pymupdf4llm | Digital PDFs with clean text | 0.01s/page | Free | Base |
| Tables | Docling | Table-heavy documents | 0.3-3s/page | Free | `pdfmux[tables]` |
| OCR | Surya | Scanned / image-based PDFs | 1-5s/page | Free | `pdfmux[ocr]` |
| LLM | Gemini 2.5 Flash | Complex layouts, handwriting, edge cases | 2-5s/page | ~$0.01/doc | `pdfmux[llm]` |

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

Structured output with metadata, useful for pipelines:

```json
{
  "source": "report.pdf",
  "converter": "pdfmux",
  "extractor": "pymupdf4llm (fast)",
  "page_count": 5,
  "confidence": 0.95,
  "warnings": [],
  "content": "# Quarterly Report\n\nRevenue for Q3...",
  "pages": [
    { "page": 1, "content": "# Quarterly Report..." },
    { "page": 2, "content": "## Financial Summary..." }
  ]
}
```

### CSV

Extracts tables from the document into CSV format:

```csv
Metric,Q3 2025,Q3 2024
Revenue,$12.3M,$10.7M
Profit,$3.1M,$2.4M
```

Raises an error if no tables are found in the document.

## MCP Server

pdfmux includes a built-in MCP (Model Context Protocol) server so AI agents can read PDFs natively.

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

### Tool

The server exposes a single `convert_pdf` tool over stdio:

```json
{
  "name": "convert_pdf",
  "description": "Convert a PDF to Markdown/JSON/CSV",
  "parameters": {
    "file_path": "string — path to the PDF file",
    "format": "string — markdown | json | csv (default: markdown)",
    "quality": "string — fast | standard | high (default: standard)"
  }
}
```

Your agent calls it, gets the extracted text back. No setup required.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Only for `pdfmux[llm]` | Google Gemini API key for LLM extraction |
| `GOOGLE_API_KEY` | Alternative | Alternative env var for Gemini API key |

No environment variables are needed for the base install or the `tables`/`ocr` extras.

## Why Not Just Use X?

| Tool | Good at | Limitation |
|------|---------|-----------|
| Marker | GPU ML extraction | Overkill for simple digital PDFs, needs GPU |
| Docling | Tables (97.9% accuracy) | Slow on non-table documents |
| pymupdf4llm | Fast digital text | Can't handle scanned or complex layouts |
| MinerU | Full ML pipeline | Heavy, complex setup |
| MarkItDown | Microsoft tool, wide format support | Not optimized for any specific PDF type |
| **pdfmux** | Picking the right tool automatically | — |

pdfmux uses these tools. It doesn't compete with them — it orchestrates them.

The key insight: no single extractor wins on everything. PyMuPDF is 100x faster on digital PDFs. Docling is better at tables. Surya handles scans. Gemini catches what everything else misses. pdfmux routes each document to the right one.

## Project Structure

```
src/pdfmux/
├── cli.py              # Typer CLI (convert, serve, version)
├── pipeline.py         # Tiered routing logic
├── detect.py           # PDF type detection
├── postprocess.py      # Cleanup + confidence scoring
├── mcp_server.py       # MCP server (stdio JSON-RPC)
├── extractors/
│   ├── fast.py         # PyMuPDF — handles 90% of PDFs
│   ├── tables.py       # Docling — table-heavy docs
│   ├── ocr.py          # Surya — scanned PDFs
│   └── llm.py          # Gemini Flash — hardest cases
└── formatters/
    ├── markdown.py     # Markdown output
    ├── json_fmt.py     # JSON output
    └── csv_fmt.py      # CSV output (tables only)
```

## Development

```bash
git clone https://github.com/NameetP/pdfmux.git
cd pdfmux
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run tests
pytest

# lint
ruff check src/ tests/
ruff format src/ tests/
```

## License

[MIT](LICENSE)
