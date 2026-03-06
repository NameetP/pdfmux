+++
date = '2026-03-06'
draft = false
title = 'I benchmarked every PDF-to-Markdown tool. Then I built a router.'
description = 'No single PDF extraction tool wins at everything. I tested PyMuPDF, Docling, Marker, Surya, and Gemini Flash across 4 categories — then built pdfmux, a self-healing pipeline that picks the best tool per page.'
tags = ['benchmark', 'pdf-extraction', 'python', 'rag', 'llm']
slug = 'benchmarking-pdf-extractors'
+++

**TL;DR**: No single PDF extraction tool is best at everything. PyMuPDF is 100x faster on digital PDFs. Docling wins on tables. Surya handles scans. So I built pdfmux — a self-healing pipeline that routes each page to the best extractor, scores quality, and re-extracts failures automatically. `pip install pdfmux`.

---

## The problem

Most RAG pipelines fail before they reach the model. The problem is not the LLM. The problem is **document ingestion**.

I was building an AI pipeline that needed to ingest PDFs. There are ~15 tools that convert PDFs to text, and they all have different tradeoffs:

- **PyMuPDF** — blazing fast (0.01s/page) but can't handle scanned docs
- **Marker** — great ML-powered extraction but needs a GPU and is overkill for simple invoices
- **Docling** — 97.9% table accuracy but slow on everything else
- **Surya OCR** — handles scans but pointless for digital PDFs
- **Gemini Flash** — catches everything but costs money and is slow

The kicker: **90% of PDFs are digital** — clean, extractable text. You don't need ML, OCR, or an LLM for those. PyMuPDF does it in 10 milliseconds.

But the other 10%? Those need specialized tools. And you don't know which 10% until you check each page.

## The benchmark

I tested the major tools across four categories.

### Digital PDFs (clean text)

| Tool | Speed (per page) | Accuracy |
|------|-----------------|----------|
| PyMuPDF | 0.01s | 98%+ |
| Marker | 0.5-2s | 98%+ |
| Docling | 0.3-1s | 95%+ |
| Gemini Flash | 2-5s | 99%+ |

For digital PDFs, PyMuPDF is 50-500x faster than everything else and just as accurate. Using anything else is burning time and money.

### Table-heavy documents

| Tool | Table accuracy | Preserves structure |
|------|---------------|-------------------|
| Docling | 97.9% | Yes |
| PyMuPDF | ~60% | Partial |
| Marker | ~85% | Yes |
| Gemini Flash | ~95% | Yes |

Docling wins on tables. It's purpose-built for it.

### Scanned PDFs

| Tool | Works? | Speed |
|------|--------|-------|
| PyMuPDF | No (no text to extract) | — |
| Surya OCR | Yes | 1-5s/page |
| Marker | Yes (with GPU) | 0.5-2s/page |
| Gemini Flash | Yes | 2-5s/page |

You need OCR or vision. PyMuPDF gets nothing from a scanned doc.

### The insight

No tool wins everywhere. The best approach is to **detect what kind of content each page has and route to the right tool**.

Most tools extract once and hope for the best. But what if you could check whether the extraction actually worked?

## The solution: pdfmux

pdfmux is a self-healing extraction pipeline. It doesn't just extract — it **audits every page**, detects broken ones, and re-extracts them automatically.

```
1. Extract     — PyMuPDF on every page (instant)
2. Audit       — 5 quality checks per page: good / bad / empty
3. Region OCR  — surgical OCR on image regions in bad pages
4. Full OCR    — re-extract remaining empty pages completely
5. Merge       — combine good + fixed pages in order
```

The key differentiator: **per-page confidence scoring**. After extraction, pdfmux runs 5 quality checks on every page — character density, alphabetic ratio, word structure, whitespace patterns, and mojibake detection. Each page gets a confidence score from 0 to 1. Pages that score below the threshold get re-extracted with a better tool.

All pages good? Zero OCR overhead. You only pay for what's broken.

### What this looks like in practice

```
Typical single-tool output:
  Page 1: (ok)
  Page 2: (ok)
  Page 3: [empty]
  Page 4: Amoun  Dscriptin  $450  Consltng
  Page 5: (ok)
  → No quality info. No way to know which pages are broken.

pdfmux output:
  Page 1: good  0.98
  Page 2: good  0.96
  Page 3: bad → OCR'd  0.91
  Page 4: bad → OCR'd  0.87
  Page 5: good  0.97
  → ✓ 5 pages, 94% avg confidence, 2 re-extracted with OCR
```

### Usage

Three lines of Python:

```python
import pdfmux

# extract as markdown — auto-audits every page
text = pdfmux.extract_text("report.pdf")

# structured json with locked schema
data = pdfmux.extract_json("report.pdf")

# LLM-ready chunks with token estimates
chunks = pdfmux.load_llm_context("report.pdf")
# → [{title, text, page_start, page_end, tokens, confidence}]
```

Or the CLI:

```bash
pip install pdfmux
pdfmux invoice.pdf          # → invoice.md
pdfmux pitch-deck.pdf       # auto-detects scanned pages, OCRs them
pdfmux analyze report.pdf   # quick quality triage
pdfmux serve                # MCP server for AI agents
```

### The fallback chain

What makes pdfmux practical: if you don't install the optional extractors, it falls back silently.

```bash
# just the base — handles 90% of PDFs
pip install pdfmux

# add table support when you need it
pip install "pdfmux[tables]"

# add OCR for scanned pages
pip install "pdfmux[ocr]"

# add everything
pip install "pdfmux[all]"
```

No errors, no config. If Docling isn't installed and you hit a table-heavy PDF, pdfmux falls back to PyMuPDF and does its best.

## Built for LLM pipelines

pdfmux outputs structured content designed for RAG pipelines, vector databases, agent workflows, and knowledge retrieval systems.

Output includes:
- Section boundaries with page references
- Per-page confidence scoring
- Structured chunks with token estimates
- Locked JSON schema (API frozen for 1.x — your code won't break on updates)

### LangChain and LlamaIndex

```python
# LangChain
from pdfmux.integrations.langchain import PDFMuxLoader
loader = PDFMuxLoader("report.pdf")
docs = loader.load()  # → list[Document]

# LlamaIndex
from pdfmux.integrations.llamaindex import PDFMuxReader
reader = PDFMuxReader()
docs = reader.load_data("report.pdf")  # → list[Document]
```

## MCP server

pdfmux includes a built-in MCP server. Add it to Claude Desktop or Cursor and your AI agent can read PDFs natively:

```json
{
  "mcpServers": {
    "pdfmux": { "command": "pdfmux", "args": ["serve"] }
  }
}
```

Three tools: `convert_pdf` for extraction, `analyze_pdf` for quick quality triage, `batch_convert` for directories.

## Try it

```bash
pip install pdfmux
pdfmux your-file.pdf
```

- [GitHub](https://github.com/NameetP/pdfmux) — source code, docs, examples
- [PyPI](https://pypi.org/project/pdfmux/) — `pip install pdfmux`
- [Website](https://pdfmux.com) — documentation and API reference

MIT licensed. Runs locally. No API keys needed for the base install.

---

*Built by [Nameet Potnis](https://github.com/NameetP). Contributions welcome.*
