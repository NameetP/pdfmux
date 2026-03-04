# Architecture — pdfmux v0.4.0

PDF extraction that checks its own work. This document describes how.

## System Overview

```
pdfmux Python API / CLI / MCP server
    │
    ├─ __init__.py        public API: extract_text, extract_json, load_llm_context
    │
    ├─ detect.py          classify PDF (digital / scanned / graphical / mixed / tables)
    │
    ├─ pipeline.py        route to extractor based on classification + quality
    │   │
    │   ├─ quality=fast     → FastExtractor only
    │   ├─ quality=high     → LLM → OCR → Fast fallback
    │   ├─ has_tables       → TableExtractor → Fast fallback
    │   └─ standard         → multi-pass pipeline (below)
    │
    ├─ audit.py           per-page quality auditing (the core of multi-pass)
    │
    ├─ chunking.py        section-aware splitting + token estimation
    │
    ├─ extractors/
    │   ├─ fast.py          PyMuPDF / pymupdf4llm — 0.01s/page, handles 90%
    │   ├─ rapid_ocr.py     RapidOCR (PaddleOCR v4 + ONNX) — ~200MB, CPU
    │   ├─ ocr.py           Surya OCR — legacy, ~5GB, GPU
    │   ├─ tables.py        Docling — 97.9% table accuracy
    │   └─ llm.py           Gemini 2.5 Flash — API, ~$0.01/doc
    │
    ├─ postprocess.py     text cleanup + confidence scoring
    │
    └─ formatters/        markdown, json, csv, llm output
```

## Multi-Pass Pipeline

This is what makes pdfmux different from other PDF extractors. Instead of running one extractor and hoping for the best, pdfmux verifies every page and re-extracts the ones that came out wrong.

### Flow

```
PDF
 │
 ▼
classify()                              ← detect.py
 │
 ▼
_multipass_extract()                    ← pipeline.py
 │
 ├── Pass 1: Fast extract + audit
 │    │
 │    ├── pymupdf4llm extracts every page (page_chunks=True)
 │    ├── audit.py scores each page:
 │    │    ├── "good":  ≥200 chars, OR ≥50 chars with no images
 │    │    ├── "bad":   <200 chars AND has images (text in images)
 │    │    └── "empty": <20 chars (blank or near-blank)
 │    │
 │    └── All pages good? → return fast text. Done. Zero overhead.
 │
 ├── Pass 2: Selective OCR on bad/empty pages
 │    │
 │    ├── Try RapidOCR (preferred — lightweight, CPU)
 │    │    ├── "bad" pages: use OCR only if it got MORE text than fast
 │    │    └── "empty" pages: accept any OCR result >10 chars
 │    │
 │    ├── Try Surya OCR (fallback — heavier, GPU)
 │    │    └── Same comparison logic for remaining pages
 │    │
 │    └── Try Gemini LLM (last resort — API call)
 │         └── Same comparison logic for remaining pages
 │
 └── Pass 3: Merge + clean + score
      │
      ├── Combine good pages + OCR'd pages in page order
      ├── postprocess.py cleans text
      └── Confidence score reflects actual quality
           ├── OCR'd pages get small penalty (max 15%)
           └── Unrecovered pages get proportional penalty
```

### Why Multi-Pass?

**Problem**: A pitch deck with 12 slides. PyMuPDF extracts text from 6 slides perfectly, but the other 6 have all their text baked into images. A single-pass extractor either:
- Uses fast extraction → misses 50% of the content
- Uses OCR on everything → wastes time on the 6 good pages

**Solution**: Extract fast, audit, OCR only the bad pages. Fast pages stay fast. Bad pages get fixed. Result: 85% confidence instead of 30%.

## Module Details

### `detect.py` — PDF Classification

Opens the PDF with PyMuPDF. Inspects every page for:
- Text content (character count)
- Embedded images (count + coverage area)
- Line patterns (table detection)
- Text alignment patterns (table detection)

Returns a `PDFClassification` dataclass:
```python
@dataclass
class PDFClassification:
    page_count: int
    is_digital: bool
    is_scanned: bool
    is_mixed: bool
    is_graphical: bool        # image-heavy (pitch decks, infographics)
    has_tables: bool
    graphical_pages: list[int]
    page_types: list[str]     # per-page: "digital", "scanned", "graphical"
```

### `audit.py` — Per-Page Quality Auditing

The core of multi-pass. Uses pymupdf4llm with `page_chunks=True` to get per-page text, then classifies each page.

**Thresholds** (hardcoded constants):
- `GOOD_TEXT_THRESHOLD = 200` chars — pages with 200+ chars are reliably extractable
- `MINIMAL_TEXT_THRESHOLD = 50` chars — pages with 50+ chars and no images are fine
- `EMPTY_TEXT_THRESHOLD = 20` chars — below this, page is effectively empty

Returns:
```python
@dataclass(frozen=True)
class PageAudit:
    page_num: int       # 0-indexed
    text: str           # text from fast extraction
    text_len: int
    image_count: int
    quality: str        # "good" | "bad" | "empty"
    reason: str         # human-readable

@dataclass
class DocumentAudit:
    pages: list[PageAudit]
    total_pages: int
    # Properties: good_pages, bad_pages, empty_pages, needs_ocr
```

### `pipeline.py` — Routing + Multi-Pass Orchestration

Central orchestrator. Routes based on quality preset and classification:

```
quality=fast     → FastExtractor (skip audit)
quality=high     → LLM → OCR → Fast
has_tables       → Docling → Fast (skip if graphical)
standard         → _multipass_extract()
```

Key design decision: **graphical PDFs with false-positive table detection skip Docling** and go through multi-pass. Table formatting is less valuable than OCR text recovery for image-heavy content.

`ConversionResult` includes `ocr_pages: list[int]` — which pages were re-extracted with OCR, for transparency.

### `chunking.py` — Section-Aware Splitting

Splits extracted Markdown into chunks at heading boundaries for LLM consumption.

**Strategy:**
1. Build page offset map from `\n\n---\n\n` separators
2. Find all ATX headings (`^#{1,6} `) as section boundaries
3. Map each section to `page_start`/`page_end` via character offsets
4. No headings → fall back to one chunk per page with title "Page N"

**Token estimation:** `len(text.strip()) // 4` — standard GPT-family approximation, no external tokenizer dependency.

```python
@dataclass(frozen=True)
class Chunk:
    title: str           # heading text, or "Page N"
    text: str            # content under this heading
    page_start: int      # 1-indexed
    page_end: int        # 1-indexed
    tokens: int          # estimated token count
    confidence: float    # inherited from document
```

Used by `load_llm_context()` public API and `--format llm` CLI output.

### `__init__.py` — Public Python API

Three thin wrappers around `pipeline.process()`:

```python
extract_text(path, *, quality="standard") → str         # Markdown string
extract_json(path, *, quality="standard") → dict         # dict with locked schema
load_llm_context(path, *, quality="standard") → list[dict]  # chunk dicts with tokens
```

All imports are lazy (inside functions) to avoid circular deps and keep `import pdfmux` fast.

### `postprocess.py` — Cleanup + Confidence

Text cleanup:
- Remove control characters (except newlines/tabs)
- Fix broken hyphenation across lines
- Normalize excessive blank lines
- Fix spaced-out text artifacts ("W i t h" → "With")

Confidence scoring factors:
- Text completeness (chars per page)
- Encoding quality (valid UTF-8 ratio)
- Structure preservation (headings, lists detected)
- Whitespace sanity
- Graphical page penalty (proportional to unrecovered pages)
- OCR noise penalty (max 15% for OCR'd pages)

### `extractors/` — Extractor Protocol

All extractors implement:
```python
class Extractor(Protocol):
    def extract(self, file_path: str | Path, pages: list[int] | None = None) -> str: ...
    @property
    def name(self) -> str: ...
```

**FastExtractor** (`fast.py`): pymupdf4llm → raw fitz fallback. Always available.

**RapidOCRExtractor** (`rapid_ocr.py`): PaddleOCR v4 models via ONNX runtime. ~200MB, CPU-only, Apache 2.0. Renders pages at 200 DPI, runs OCR, returns text. Has `extract_page()` for single-page extraction (used by multi-pass).

**OCRExtractor** (`ocr.py`): Surya OCR. ~5GB, PyTorch, GPU recommended. Legacy — kept for users who need it.

**TableExtractor** (`tables.py`): Docling. Best for table-heavy documents (97.9% accuracy).

**LLMExtractor** (`llm.py`): Gemini 2.5 Flash. API-based, ~$0.01/doc. Last-resort fallback for handwriting, complex layouts.

### `mcp_server.py` — MCP Server

JSON-RPC over stdio. Implements MCP protocol for AI agent integration.

Single tool: `convert_pdf`. When confidence <80% or warnings exist, response includes metadata header (confidence, extractor, OCR pages, warnings) so agents know what they're working with.

## Dependency Model

```
Base (every install):
  pymupdf, pymupdf4llm, typer, rich, mcp

Optional:
  pdfmux[ocr]        → rapidocr + onnxruntime     (~200MB, CPU)
  pdfmux[ocr-heavy]  → surya-ocr                  (~5GB, GPU)
  pdfmux[tables]     → docling                     (~500MB)
  pdfmux[llm]        → google-genai                (API key)
  pdfmux[all]        → tables + ocr + llm
```

## Design Principles

1. **Each version ships independently.** No multi-part refactors.
2. **Don't break existing interfaces.** CLI flags and import paths stay stable.
3. **Deterministic by default.** Same PDF → same output. OCR is the exception (documented).
4. **Base install stays small.** ~30MB. OCR is opt-in.
5. **Flat is better than nested.** No `core/`, `recovery/` folders until complexity justifies it.
6. **Hardest technical risk first.** Multi-pass shipped before API polish.
