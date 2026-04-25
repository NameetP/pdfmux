# pdfmux

[![CI](https://github.com/NameetP/pdfmux/actions/workflows/ci.yml/badge.svg)](https://github.com/NameetP/pdfmux/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pdfmux)](https://pypi.org/project/pdfmux/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/pdfmux)](https://pypi.org/project/pdfmux/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/pdfmux)](https://pypi.org/project/pdfmux/)

**Self-healing PDF extraction with per-page confidence scoring.** Open-source LlamaParse alternative for RAG pipelines, MCP server for Claude Desktop, LangChain + LlamaIndex loaders. Ranked #2 on opendataloader-bench (0.900).

The only PDF extractor that audits its own output. Catches blank pages, scrambled columns, broken tables — re-extracts them with a stronger backend. So your LLM gets clean data, not silent garbage. Routes each page to the best of 5 rule-based backends + BYOK LLM fallback (Gemini / Claude / GPT-4o / Ollama). One CLI. One API. Zero config.

<p align="center">
  <img src="demo.svg" alt="pdfmux terminal demo" width="700" />
</p>

```
PDF ──> pdfmux router ──> best extractor per page ──> audit ──> re-extract failures ──> Markdown / JSON / chunks
            |
            ├─ PyMuPDF         (digital text, 0.01s/page)
            ├─ OpenDataLoader  (complex layouts, 0.05s/page)
            ├─ RapidOCR        (scanned pages, CPU-only)
            ├─ Docling          (tables, 97.9% TEDS)
            ├─ Surya            (heavy OCR fallback)
            └─ YOUR LLM        (Gemini / Claude / GPT-4o / Ollama — BYOK via 5-line YAML)
```

## Install

```bash
pip install pdfmux
```

That's it. Handles digital PDFs out of the box. Add backends for harder documents:

```bash
pip install "pdfmux[ocr]"             # RapidOCR — scanned/image pages (~200MB, CPU-only)
pip install "pdfmux[tables]"          # Docling — table-heavy docs (~500MB)
pip install "pdfmux[opendataloader]"  # OpenDataLoader — complex layouts (Java 11+)
pip install "pdfmux[llm]"            # LLM fallback — Gemini, Claude, GPT-4o, Ollama
pip install "pdfmux[all]"            # everything
```

Requires Python 3.11+.

## Quick Start

### CLI

```bash
# zero config — just works
pdfmux convert invoice.pdf
# invoice.pdf -> invoice.md (2 pages, 95% confidence, via pymupdf4llm)

# RAG-ready chunks with token limits
pdfmux convert report.pdf --chunk --max-tokens 500

# cost-aware extraction with budget cap
pdfmux convert report.pdf --mode economy --budget 0.50

# schema-guided structured extraction (5 built-in presets)
pdfmux convert invoice.pdf --schema invoice

# BYOK any LLM for hardest pages
pdfmux convert scan.pdf --llm-provider claude

# batch a directory
pdfmux convert ./docs/ -o ./output/
```

### Python

```python
import pdfmux

# text -> markdown
text = pdfmux.extract_text("report.pdf")

# structured data -> dict with tables, key-values, metadata
data = pdfmux.extract_json("report.pdf")

# RAG chunks -> list of dicts with token estimates
chunks = pdfmux.chunk("report.pdf", max_tokens=500)
```

## Architecture

```
                           ┌─────────────────────────────┐
                           │     Segment Detector         │
                           │  text / tables / images /    │
                           │  formulas / headers per page │
                           └─────────────┬───────────────┘
                                         │
                    ┌────────────────────────────────────────┐
                    │            Router Engine                │
                    │                                        │
                    │   economy ── balanced ── premium        │
                    │   (minimize $)  (default)  (max quality)│
                    │   budget caps: --budget 0.50            │
                    └────────────────────┬───────────────────┘
                                         │
          ┌──────────┬──────────┬────────┴────────┬──────────┐
          │          │          │                  │          │
     PyMuPDF   OpenData    RapidOCR           Docling     LLM
     digital   Loader      scanned            tables    (BYOK)
     0.01s/pg  complex     CPU-only           97.9%    any provider
               layouts                        TEDS
          │          │          │                  │          │
          └──────────┴──────────┴────────┬────────┴──────────┘
                                         │
                    ┌────────────────────────────────────────┐
                    │           Quality Auditor               │
                    │                                        │
                    │   4-signal dynamic confidence scoring   │
                    │   per-page: good / bad / empty          │
                    │   if bad -> re-extract with next backend│
                    └────────────────────┬───────────────────┘
                                         │
                    ┌────────────────────────────────────────┐
                    │           Output Pipeline               │
                    │                                        │
                    │   heading injection (font-size analysis)│
                    │   table extraction + normalization      │
                    │   text cleanup + merge                  │
                    │   confidence score (honest, not inflated)│
                    └────────────────────────────────────────┘
```

### Key design decisions

- **Router, not extractor.** pdfmux does not compete with PyMuPDF or Docling. It picks the best one per page.
- **Agentic multi-pass.** Extract, audit confidence, re-extract failures with a stronger backend. Bad pages get retried automatically.
- **Segment-level detection.** Each page is classified by content type (text, tables, images, formulas, headers) before routing.
- **4-signal confidence.** Dynamic quality scoring from character density, OCR noise ratio, table integrity, and heading structure. Not hardcoded thresholds.
- **Document cache.** Each PDF is opened once, not once per extractor. Shared across the full pipeline.
- **Data flywheel.** Local telemetry tracks which extractors win per document type. Routing improves with usage.

## Features

| Feature | What it does | Command |
|---------|-------------|---------|
| Zero-config extraction | Routes to best backend automatically | `pdfmux convert file.pdf` |
| RAG chunking | Section-aware chunks with token estimates | `pdfmux convert file.pdf --chunk --max-tokens 500` |
| Cost modes | economy / balanced / premium with budget caps | `pdfmux convert file.pdf --mode economy --budget 0.50` |
| Schema extraction | 5 built-in presets (invoice, receipt, contract, resume, paper) | `pdfmux convert file.pdf --schema invoice` |
| BYOK LLM | Gemini, Claude, GPT-4o, Ollama, any OpenAI-compatible API | `pdfmux convert file.pdf --llm-provider claude` |
| Benchmark | Eval all installed extractors against ground truth | `pdfmux benchmark` |
| Doctor | Show installed backends, coverage gaps, recommendations | `pdfmux doctor` |
| MCP server | AI agents read PDFs via stdio or HTTP | `pdfmux serve` |
| Batch processing | Convert entire directories | `pdfmux convert ./docs/` |
| Streaming | Bounded-memory page iteration for large files | `for page in ext.extract("500pg.pdf")` |

## CLI Reference

### `pdfmux convert`

```bash
pdfmux convert <file-or-dir> [options]

Options:
  -o, --output PATH          Output file or directory
  -f, --format FORMAT        markdown | json | csv | llm (default: markdown)
  -q, --quality QUALITY      fast | standard | high (default: standard)
  -s, --schema SCHEMA        JSON schema file or preset (invoice, receipt, contract, resume, paper)
  --chunk                    Output RAG-ready chunks
  --max-tokens N             Max tokens per chunk (default: 500)
  --mode MODE                economy | balanced | premium (default: balanced)
  --budget AMOUNT            Max spend per document in USD
  --llm-provider PROVIDER    LLM backend: gemini | claude | openai | ollama
  --confidence               Include confidence score in output
  --stdout                   Print to stdout instead of file
```

### `pdfmux serve`

Start the MCP server for AI agent integration.

```bash
pdfmux serve              # stdio mode (Claude Desktop, Cursor)
pdfmux serve --http 8080  # HTTP mode
```

### `pdfmux doctor`

```bash
pdfmux doctor
# ┌──────────────────┬─────────────┬─────────┬──────────────────────────────────┐
# │ Extractor        │ Status      │ Version │ Install                          │
# ├──────────────────┼─────────────┼─────────┼──────────────────────────────────┤
# │ PyMuPDF          │ installed   │ 1.25.3  │                                  │
# │ OpenDataLoader   │ installed   │ 0.3.1   │                                  │
# │ RapidOCR         │ installed   │ 3.0.6   │                                  │
# │ Docling          │ missing     │ --      │ pip install pdfmux[tables]       │
# │ Surya            │ missing     │ --      │ pip install pdfmux[ocr-heavy]    │
# │ LLM (Gemini)     │ configured  │ --      │ GEMINI_API_KEY set               │
# └──────────────────┴─────────────┴─────────┴──────────────────────────────────┘
```

### `pdfmux benchmark`

```bash
pdfmux benchmark report.pdf
# ┌──────────────────┬────────┬────────────┬─────────────┬──────────────────────┐
# │ Extractor        │   Time │ Confidence │      Output │ Status               │
# ├──────────────────┼────────┼────────────┼─────────────┼──────────────────────┤
# │ PyMuPDF          │  0.02s │        95% │ 3,241 chars │ all pages good       │
# │ Multi-pass       │  0.03s │        95% │ 3,241 chars │ all pages good       │
# │ RapidOCR         │  4.20s │        88% │ 2,891 chars │ ok                   │
# │ OpenDataLoader   │  0.12s │        97% │ 3,310 chars │ best                 │
# └──────────────────┴────────┴────────────┴─────────────┴──────────────────────┘
```

## Python API

### Text extraction

```python
import pdfmux

text = pdfmux.extract_text("report.pdf")                    # -> str (markdown)
text = pdfmux.extract_text("report.pdf", quality="fast")    # PyMuPDF only, instant
text = pdfmux.extract_text("report.pdf", quality="high")    # LLM-assisted
```

### Structured extraction

```python
data = pdfmux.extract_json("report.pdf")
# data["page_count"]   -> 12
# data["confidence"]   -> 0.91
# data["ocr_pages"]    -> [2, 5, 8]
# data["pages"][0]["key_values"]  -> [{"key": "Date", "value": "2026-02-28"}]
# data["pages"][0]["tables"]      -> [{"headers": [...], "rows": [...]}]
```

### RAG chunking

```python
chunks = pdfmux.chunk("report.pdf", max_tokens=500)
for c in chunks:
    print(f"{c['title']}: {c['tokens']} tokens (pages {c['page_start']}-{c['page_end']})")
```

### Schema-guided extraction

```python
data = pdfmux.extract_json("invoice.pdf", schema="invoice")
# Uses built-in invoice preset: extracts date, vendor, line items, totals
# Also accepts a path to a custom JSON Schema file
```

### Streaming (bounded memory)

```python
from pdfmux.extractors import get_extractor

ext = get_extractor("fast")
for page in ext.extract("large-500-pages.pdf"):  # Iterator[PageResult]
    process(page.text)  # constant memory, even on 500-page PDFs
```

### Types and errors

```python
from pdfmux import (
    # Enums
    Quality,              # FAST, STANDARD, HIGH
    OutputFormat,         # MARKDOWN, JSON, CSV, LLM
    PageQuality,          # GOOD, BAD, EMPTY

    # Data objects (frozen dataclasses)
    PageResult,           # page: text, page_num, confidence, quality, extractor
    DocumentResult,       # document: pages, source, confidence, extractor_used
    Chunk,                # chunk: title, text, page_start, page_end, tokens

    # Errors
    PdfmuxError,          # base -- catch this for all pdfmux errors
    FileError,            # file not found, unreadable, not a PDF
    ExtractionError,      # extraction failed
    ExtractorNotAvailable,# requested backend not installed
    FormatError,          # invalid output format
    AuditError,           # audit could not complete
)
```

## Framework Integrations

### LangChain

```bash
pip install langchain-pdfmux
```

```python
from langchain_pdfmux import PDFMuxLoader

loader = PDFMuxLoader("report.pdf", quality="standard")
docs = loader.load()  # -> list[Document] with confidence metadata
```

### LlamaIndex

```bash
pip install llama-index-readers-pdfmux
```

```python
from llama_index.readers.pdfmux import PDFMuxReader

reader = PDFMuxReader(quality="standard")
docs = reader.load_data("report.pdf")  # -> list[Document]
```

### MCP Server (AI Agents)

Listed on [mcpservers.org](https://mcpservers.org). One-line setup:

```json
{
  "mcpServers": {
    "pdfmux": {
      "command": "npx",
      "args": ["-y", "pdfmux-mcp"]
    }
  }
}
```

Or via Claude Code:

```bash
claude mcp add pdfmux -- npx -y pdfmux-mcp
```

Tools exposed: `convert_pdf`, `analyze_pdf`, `extract_structured`, `get_pdf_metadata`, `batch_convert`.

## BYOK LLM Configuration

pdfmux supports any LLM via 5 lines of YAML. Bring your own keys -- nothing leaves your machine unless you configure it to.

```yaml
# ~/.pdfmux/llm.yaml
provider: claude          # gemini | claude | openai | ollama | any OpenAI-compatible
model: claude-sonnet-4-20250514
api_key: ${ANTHROPIC_API_KEY}
base_url: https://api.anthropic.com  # optional, for custom endpoints
max_cost_per_page: 0.02   # budget cap
```

Supported providers:

| Provider | Models | Local? | Cost |
|----------|--------|--------|------|
| Gemini | 2.5 Flash, 2.5 Pro | No | ~$0.01/page |
| Claude | Sonnet, Opus | No | ~$0.015/page |
| GPT-4o | GPT-4o, GPT-4o-mini | No | ~$0.01/page |
| Ollama | Any local model | Yes | Free |
| Custom | Any OpenAI-compatible API | Configurable | Varies |

## Benchmark

Tested on [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench) -- 200 real-world PDFs across financial reports, legal filings, academic papers, and scanned documents.

| Engine | Overall | Reading Order | Tables (TEDS) | Headings | Requires |
|--------|---------|---------------|---------------|----------|----------|
| opendataloader hybrid | 0.909 | 0.935 | 0.928 | 0.828 | API calls ($) |
| **pdfmux** | **0.905** | **0.920** | **0.911** | **0.852** | **CPU only, $0** |
| docling | 0.877 | 0.900 | 0.887 | 0.802 | ~500MB models |
| marker | 0.861 | 0.890 | 0.808 | 0.796 | GPU recommended |
| opendataloader local | 0.844 | 0.913 | 0.494 | 0.761 | CPU only |
| mineru | 0.831 | 0.857 | 0.873 | 0.743 | GPU + ~2GB models |

#2 overall, #1 among free tools. 99.5% of the paid #1 score at zero cost per page. Best heading detection of any engine tested. Image table OCR extracts tables embedded as images.

## Confidence Scoring

Every result includes a 4-signal confidence score:

- **95-100%** -- clean digital text, fully extractable
- **80-95%** -- good extraction, minor OCR noise on some pages
- **50-80%** -- partial extraction, some pages unrecoverable
- **<50%** -- significant content missing, warnings included

When confidence drops below 80%, pdfmux tells you exactly what went wrong and how to fix it:

```
Page 4: 32% confidence. 0 chars extracted from image-heavy page.
  -> Install pdfmux[ocr] for RapidOCR support on 6 image-heavy pages.
```

## Cost Modes

| Mode | Behavior | Typical cost |
|------|----------|-------------|
| economy | Rule-based backends only. No LLM calls. | $0/page |
| balanced | LLM only for pages that fail rule-based extraction. | ~$0.002/page avg |
| premium | LLM on every page for maximum quality. | ~$0.01/page |

Set a hard budget cap: `--budget 0.50` stops LLM calls when spend reaches $0.50 per document.

## Why pdfmux?

pdfmux is not another PDF extractor. It is the orchestration layer that picks the right extractor per page, verifies the result, and retries failures.

| Tool | Good at | Limitation |
|------|---------|-----------|
| PyMuPDF | Fast digital text | Cannot handle scans or image layouts |
| Docling | Tables (97.9% accuracy) | Slow on non-table documents |
| Marker | GPU ML extraction | Needs GPU, overkill for digital PDFs |
| Unstructured | Enterprise platform | Complex setup, paid tiers |
| LlamaParse | Cloud-native | Requires API keys, not local |
| Reducto | High accuracy | $0.015/page, closed source |
| **pdfmux** | **Orchestrates all of the above** | Routes per page, audits, re-extracts |

Open source Reducto alternative: what costs $0.015/page elsewhere is free with pdfmux's rule-based backends, or ~$0.002/page average with BYOK LLM fallback.

## Development

```bash
git clone https://github.com/NameetP/pdfmux.git
cd pdfmux
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest              # 151 tests
ruff check src/ tests/
ruff format src/ tests/
```

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure `pytest` and `ruff check` pass
5. Open a PR

## License

[MIT](LICENSE)

<!-- mcp-name: io.github.NameetP/pdfmux -->

