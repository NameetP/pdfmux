# Architecture — pdfmux v1.6.0

PDF extraction that checks its own work. This document describes how.

## System Overview

```
pdfmux Python API / CLI / MCP server
    │
    ├─ __init__.py        public API: extract_text, extract_json, load_llm_context + type/error exports
    │
    ├─ types.py           frozen dataclasses + enums: Quality, OutputFormat, PageResult, DocumentResult, Chunk
    ├─ errors.py          exception hierarchy with .user_message / .suggestion / .reproduce_cmd
    ├─ retry.py           @with_retry — exponential backoff, Retry-After aware, skips on auth failures
    │
    ├─ detect.py          classify PDF (digital / scanned / graphical / mixed / tables / arabic / academic)
    │
    ├─ pipeline.py        route to extractor based on classification + quality
    │   │                 (consults result_cache before running, persists on success)
    │   │
    │   ├─ quality=fast     → FastExtractor only
    │   ├─ quality=high     → LLM → OCR → Fast fallback
    │   ├─ has_tables       → TableExtractor → Fast fallback
    │   └─ standard         → multi-pass pipeline (below)
    │
    ├─ audit.py           5-check per-page confidence scoring + quality classification
    │
    ├─ chunking.py        section-aware splitting + token estimation
    │
    ├─ result_cache.py    SHA-256 file hashing → ~/.cache/pdfmux/results/, 30d TTL, 1 GB LRU
    ├─ streaming.py       generator yielding NDJSON StreamEvents (classified/page/warning/complete)
    ├─ profiles.py        ~/.config/pdfmux/profiles.yaml + 5 built-ins
    ├─ arabic.py          BiDi reordering + Arabic detection + canonicalization
    │
    ├─ extractors/
    │   ├─ __init__.py      Extractor protocol + @register decorator + priority-ordered registry
    │   ├─ fast.py          PyMuPDF / pymupdf4llm — 0.01s/page, handles 90% (priority 10)
    │   ├─ rapid_ocr.py     RapidOCR (PaddleOCR v4 + ONNX) — ~200MB, CPU (priority 20)
    │   ├─ mistral_ocr.py   Mistral OCR API — $0.002/page, 96.6% TEDS  (priority 25)
    │   ├─ ocr.py           Surya OCR — legacy, ~5GB, GPU (priority 30)
    │   ├─ marker.py        Marker neural extractor — academic papers (priority 35)
    │   ├─ tables.py        Docling — 97.9% table accuracy (priority 40)
    │   ├─ opendataloader.py Java-backed reading-order extractor (priority 45)
    │   └─ llm.py           BYOK vision LLM — Gemini, Gemma 4, Claude, GPT-4o, Ollama, Mistral (priority 50)
    │
    ├─ providers/
    │   ├─ base.py            LLMProvider protocol + ModelInfo + CostEstimate
    │   ├─ _discovery.py      auto-registers all built-in providers
    │   ├─ gemini.py          Gemini 2.5 Flash / Pro
    │   ├─ gemma.py           Gemma 4 27B IT (Gemini OpenAI-compat endpoint, reuses GEMINI_API_KEY)
    │   ├─ claude.py          Claude Sonnet / Opus
    │   ├─ openai_native.py   GPT-4o family
    │   ├─ openai_compatible.py  Mistral + custom OpenAI-compatible APIs
    │   └─ ollama.py          local models
    │
    ├─ postprocess.py     text cleanup + Arabic BiDi (markdown-aware)
    │
    └─ formatters/        markdown, json, csv, llm output
```

## Type System

All data flowing through the pipeline is represented by frozen dataclasses and enums defined in `types.py`. Nothing is a raw dict. Nothing is mutable after construction.

```python
class Quality(Enum):         # FAST, STANDARD, HIGH
class OutputFormat(Enum):    # MARKDOWN, JSON, CSV, LLM
class PageQuality(Enum):     # GOOD, BAD, EMPTY

@dataclass(frozen=True)
class PageResult:            # page_num, text, confidence, quality, extractor, image_count, ocr_applied
@dataclass(frozen=True)
class DocumentResult:        # pages, source, confidence, extractor_used, format, text, warnings, ocr_pages
@dataclass(frozen=True)
class Chunk:                 # title, text, page_start, page_end, tokens, confidence
```

All types are exported from `import pdfmux`. They are frozen (`@dataclass(frozen=True)`) so they can be hashed, stored in sets, and passed across threads safely.

## Error Hierarchy

Exceptions follow a flat hierarchy in `errors.py`:

```
PdfmuxError                      base — catch this to handle all pdfmux errors
├── FileError                    bad path, unreadable, not a PDF, corrupted
├── ExtractionError              extractor failed to produce output
├── ExtractorNotAvailable        optional dependency not installed (includes install instructions)
├── FormatError                  invalid or unsupported output format
└── AuditError                   per-page quality audit failed (non-fatal in pipeline)
```

Propagation rules:
- `FileError` → raise immediately, nothing to retry
- `ExtractorNotAvailable` → log + try next extractor in registry
- `ExtractionError` → log + try next extractor in registry
- `FormatError` → raise immediately, caller picked bad format
- `AuditError` → log + skip audit, return unaudited result

## Extractor Protocol & Registry

Every extractor implements the `Extractor` protocol:

```python
@runtime_checkable
class Extractor(Protocol):
    @property
    def name(self) -> str: ...
    def extract(self, file_path: str | Path, pages: list[int] | None = None) -> Iterator[PageResult]: ...
    def available(self) -> bool: ...
```

Extractors register themselves with `@register(name, priority)` at class definition time:

```python
@register(name="fast", priority=10)
class FastExtractor:
    ...
```

The registry maintains priority order. Lower priority = tried first. The pipeline queries at runtime:

- `get_extractor("fast")` — get a specific extractor by name
- `available_extractors()` — list all extractors whose dependencies are installed
- `extractor_names()` — list all registered names (installed or not)

| Priority | Registry Name | Backend | Use Case |
|----------|--------------|---------|----------|
| 10 | `fast` | PyMuPDF / pymupdf4llm | Digital text PDFs |
| 20 | `rapidocr` | RapidOCR (PaddleOCR v4) | Lightweight OCR (~200MB, CPU) |
| 25 | `mistral_ocr` | Mistral OCR API | Cloud table OCR — 96.6% TEDS, $0.002/page |
| 30 | `surya` | Surya OCR | Heavy OCR (~5GB, GPU) |
| 35 | `marker` | Marker neural extractor | Academic papers, dense layouts |
| 40 | `docling` | Docling | Table-heavy documents |
| 45 | `opendataloader` | OpenDataLoader (Java) | Complex multi-column reading order |
| 50 | `llm` | BYOK vision LLM | Complex layouts, handwriting, Arabic, charts |

The router (`router/engine.py`) maintains a `ROUTING_MATRIX` keyed by page
type (`digital`, `scanned`, `graphical`, `tables`, `academic`, `arabic`)
and a `quality` preset, plus per-extractor `QUALITY_ESTIMATES` and
`COST_PER_PAGE` to power `pdfmux estimate`.

## Streaming Architecture

All extractors yield `Iterator[PageResult]` — one page object at a time. The pipeline consumes pages lazily. This bounds memory to roughly one page of text plus extractor overhead, regardless of document length.

Measured: ~135MB peak on a 500-page PDF. Before v0.5.0, memory scaled linearly with page count.

```
Extract (streaming) → Audit (per-page) → Re-extract failures (streaming) → Merge → Format
```

## 5-Check Confidence Scoring

Each page receives a confidence score from 0.0 to 1.0, computed from 5 independent checks in `audit.score_page()`:

| Check | What it measures |
|-------|-----------------|
| Character density | Characters per page — catches blank extractions |
| Alphabetic ratio | Proportion of alphabetic characters — catches garbage/binary data |
| Word structure | Average word length and variance — catches joined/split text |
| Whitespace sanity | Whitespace distribution — catches formatting collapse |
| Encoding quality | Mojibake detection via regex — catches encoding failures |

Document-level confidence (`audit.compute_document_confidence()`) is a content-weighted average: longer pages contribute more than shorter ones. Additional penalties for OCR'd pages and unrecovered pages.

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
 │    ├── audit.score_page() scores each page with 5 checks
 │    │    ├── "good":  sufficient text density and quality
 │    │    ├── "bad":   low text with images (text in images)
 │    │    └── "empty": near-blank page
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
      ├── postprocess.clean_text() cleans text
      └── audit.compute_document_confidence() scores the merged result
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

Raises `FileError` for missing files, non-PDFs, or corrupted documents.

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

### `pipeline.py` — Routing + Multi-Pass Orchestration

Central orchestrator. Routes based on quality preset and classification:

```
quality=fast     → FastExtractor (skip audit)
quality=high     → LLM → OCR → Fast
has_tables       → Docling → Fast (skip if graphical)
standard         → _multipass_extract()
```

Key design decision: **graphical PDFs with false-positive table detection skip Docling** and go through multi-pass. Table formatting is less valuable than OCR text recovery for image-heavy content.

`process()` returns a `ConversionResult` (legacy wrapper) with text, confidence, extractor_used, ocr_pages, and warnings.

`process_batch()` uses `ThreadPoolExecutor` for concurrent processing with per-file error isolation.

### `chunking.py` — Section-Aware Splitting

Splits extracted Markdown into chunks at heading boundaries for LLM consumption.

**Strategy:**
1. Build page offset map from `\n\n---\n\n` separators
2. Find all ATX headings (`^#{1,6} `) as section boundaries
3. Map each section to `page_start`/`page_end` via character offsets
4. No headings → fall back to one chunk per page with title "Page N"

**Token estimation:** `len(text.strip()) // 4` — standard GPT-family approximation, no external tokenizer dependency.

Uses `Chunk` from `pdfmux.types`:
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

### `__init__.py` — Public Python API

Three thin wrappers around `pipeline.process()`:

```python
extract_text(path, *, quality="standard") → str            # Markdown string
extract_json(path, *, quality="standard") → dict           # dict with locked schema
load_llm_context(path, *, quality="standard") → list[dict] # chunk dicts with tokens
```

All imports are lazy (inside functions) to avoid circular deps and keep `import pdfmux` fast.

Re-exports all 6 types and all 6 error classes for convenience.

### `postprocess.py` — Cleanup

Text cleanup:
- Remove control characters (except newlines/tabs)
- Fix broken hyphenation across lines
- Normalize excessive blank lines
- Fix spaced-out text artifacts ("W i t h" → "With")

### `mcp_server.py` — MCP Server

JSON-RPC over stdio (or HTTP via `pdfmux serve --http <port>`). Implements
the MCP protocol for AI agent integration.

Tools exposed:

| Tool | Purpose |
|------|---------|
| `convert_pdf` | Full extraction → markdown / json / csv / llm |
| `analyze_pdf` | Quick triage: classification, page count, recommended quality |
| `extract_structured` | Tables and key-values via schema preset |
| `extract_streaming` | NDJSON page-by-page events for very long documents |
| `get_pdf_metadata` | Title, author, page count, embedded metadata |
| `batch_convert` | Run a directory of PDFs through the pipeline |

When confidence < 80% or warnings exist, responses include a metadata
header (confidence, extractor, OCR pages, warnings) so agents know what
they're working with.

## Dependency Model

```
Base (every install):
  pymupdf, pymupdf4llm, typer, rich, python-bidi

Optional:
  pdfmux[ocr]              → rapidocr + onnxruntime         (~200MB, CPU)
  pdfmux[ocr-heavy]        → surya-ocr                      (~5GB, GPU)
  pdfmux[tables]           → docling                        (~500MB)
  pdfmux[opendataloader]   → opendataloader-pdf             (Java 11+)
  pdfmux[marker]           → marker-pdf                     (~2GB models)
  pdfmux[llm]              → google-genai                   (Gemini API key)
  pdfmux[llm-claude]       → anthropic
  pdfmux[llm-openai]       → openai
  pdfmux[llm-ollama]       → ollama
  pdfmux[llm-mistral]      → mistralai
  pdfmux[llm-all]          → all LLM providers (Gemma 4 piggybacks on Gemini key)
  pdfmux[langchain]        → langchain-core
  pdfmux[llamaindex]       → llama-index-core
  pdfmux[serve]            → mcp + uvicorn (MCP server)
  pdfmux[watch]            → watchdog (`pdfmux watch <dir>`)
  pdfmux[all]              → everything
```

## v1.6 Additions

### `result_cache.py` — Smart result cache

Hash every input PDF (SHA-256). Build a cache key from
`(file_hash, quality, format, schema)`. Cache hit returns the previous
`DocumentResult` in milliseconds; miss runs the pipeline and persists the
result. Cache lives at `~/.cache/pdfmux/results/{hash}.json`, expires after
30 days, LRU-evicts at 1 GB. The CLI exposes `--no-cache` (bypass) and
`--clear-cache` (purge then run).

### `streaming.py` — Page-at-a-time NDJSON

Generator that yields `StreamEvent` objects in order:

```
classified  → page → page → ... → warning? → complete
```

Each `page` event ships once a page exits the audit/recovery pipeline so
clients can render output progressively. Wired into the CLI as
`pdfmux stream` and into the MCP server as `extract_streaming`.

### `profiles.py` — Saved configurations

YAML-backed preset store at `~/.config/pdfmux/profiles.yaml`. Built-ins:
`invoices`, `receipts`, `papers`, `contracts`, `bulk-rag`. CRUD via
`pdfmux profiles list/show/save/delete`. `convert --profile <name>` merges
profile values with explicit flags (explicit flags win).

### `arabic.py` — RTL pipeline

- `is_arabic_text` / `arabic_ratio` — fast detection.
- `fix_bidi_order` — Unicode Bidi Algorithm (`python-bidi`) with
  markdown awareness: heading prefixes (`#`), list markers, code fences,
  and pipe-table cells are preserved while inner text is reordered.
- `normalize_arabic` — Tatweel removal, Alef/Yeh/Hamza unification,
  Tashkeel stripping for indexable / embeddable strings.

`detect.py` samples the first 20 pages and sets
`PDFClassification.is_arabic` + `arabic_pages`. The router's
`ROUTING_MATRIX` includes an `arabic` page type that prefers the Gemma 4
provider when available.

### `retry.py` — Decorator on every LLM provider

`@with_retry(max_attempts=3, backoff_base=2.0)` wraps each provider's
`extract_page()`. Honors `Retry-After` headers, returns immediately on
auth failures (401/403), and uses exponential backoff with jitter on
5xx / 429 / network errors.

## Design Principles

1. **Each version ships independently.** No multi-part refactors.
2. **Don't break existing interfaces.** CLI flags and import paths stay stable.
3. **Deterministic by default.** Same PDF → same output. OCR is the exception (documented).
4. **Base install stays small.** ~30MB. OCR is opt-in.
5. **Flat is better than nested.** No `core/`, `recovery/` folders until complexity justifies it.
6. **Types over dicts.** All data flows through frozen dataclasses. No raw dicts in the pipeline.
7. **Hardest technical risk first.** Multi-pass shipped before API polish.
