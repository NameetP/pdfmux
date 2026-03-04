# Multi-Pass Architecture — Feedback

Collecting all feedback before finalizing the plan.

---

## Feedback #1: 8 Reference Documents (OAI-generated architectural specs)

The user shared 8 comprehensive documents covering pdfmux's target architecture, engineering standards, benchmarking, competitive positioning, and production roadmap. Key insights that should inform our plan:

### From Reference Architecture
- **Layered architecture**: CORE → RECOVERY → STRUCTURE → LLM → PERFORMANCE → INTERFACES — each layer can be enabled/disabled independently
- **Separate modules proposed**: `audit.py`, `merge.py`, `confidence.py`, `page_analysis.py` as distinct files in `core/`
- **Region OCR** (not full-page): detect weak blocks → crop bounding box → OCR just that region → merge. Faster and more accurate than OCR'ing entire pages.
- **Block scoring**: evaluate individual text blocks by text length, alphabetic ratio, word structure. Low-scoring blocks get reprocessed.
- **Layout grid detection**: cluster x-positions, group blocks into columns, sort column-by-column for multi-column PDFs
- **Fingerprint cache**: perceptual hash of pages to reuse extraction results for repeated layouts
- **OCR budget control**: `max_ocr_pages = 0.3 * document_pages` — cap how much OCR we do
- **Confidence formula**: `0.5 * text_coverage + 0.3 * OCR_reliability + 0.2 * structure_integrity`
- **Repository structure**: much more modular than current — `core/`, `recovery/`, `structure/`, `llm/`, `performance/`, `cli/`, `mcp/`, `extractors/`, `models/`, `utils/`

### From Engineering Playbook
- **Deterministic behavior**: same input → same output, byte-stable, no randomness in heuristics
- **Minimal dependencies**: base install = Python + PyMuPDF ONLY. Everything else behind extras.
- **Pipeline isolation**: layers must not create circular dependencies, cross-layer imports through interfaces
- **Coverage targets**: core logic 90%, critical pipeline 95%
- **Performance guardrails**: no unnecessary page rasterization, OCR only when needed, avoid copying large arrays
- **Logging**: optional, default level WARN
- **Security**: all PDFs treated as untrusted input

### From Benchmark Dataset Spec
- **Ground truth format**: JSON with sections array `[{type, text, page}, ...]`
- **Evaluation metrics**: text accuracy (normalized edit distance, token overlap), structure accuracy, OCR usage, confidence honesty, runtime
- **Compare against**: pdfminer.six, pdftotext, unstructured, pymupdf4llm
- **Expected accuracy targets**: digital 95-99%, slides 85-92%, scanned 85-90%, papers 90-95%
- **Benchmark automation**: `pdfmux bench benchmarks/` with dataset categories

### From Public Spec
- **3 CLI commands**: convert, analyze, bench
- **JSON output schema** (should be locked):
  ```json
  {
    "document": "file.pdf",
    "pages": 12,
    "confidence": 0.91,
    "ocr_pages": [2, 5],
    "sections": [
      {"type": "heading", "text": "...", "page": 1},
      {"type": "paragraph", "text": "...", "page": 1}
    ]
  }
  ```
- **LLM output format** with chunks: `{title, page_start, page_end, tokens, confidence, text}`
- **Error codes**: PDF_PARSE_ERROR, OCR_FAILURE, INVALID_FILE, UNSUPPORTED_FORMAT
- **MCP tools**: pdf.extract_text, pdf.extract_llm_context, pdf.analyze
- **`--format llm`** flag for LLM-ready chunked output

### From Competitive Analysis
- **pdfmux differentiators vs competition**: deterministic pipeline, selective OCR fallback, section-aware chunking, confidence scoring, LLM-ready schemas, CLI-first, MCP support
- **Feature gap**: pdfmux has confidence scoring, MCP, and LLM chunking that NO competitor has
- **Position**: not "PDF text extractor" — "reliable PDF ingestion for LLM systems"

### From RAG Integration Guide
- **Python API**: `load_llm_context(path)` → returns list of chunks with text, confidence, tokens
- **Integration patterns**: LangChain, LlamaIndex, Haystack, Pinecone, Qdrant, Weaviate
- **Key insight**: chunk boundaries should follow document structure, not arbitrary token counts
- **Confidence per chunk**: stored in vector DB metadata for retrieval filtering

### From Launch Pack
- **Roadmap**:
  - v0.3: core extraction + OCR fallback + CLI + JSON output + analyze command
  - v0.4: parallel processing + fingerprint caching + MCP server
  - v1.0: layout intelligence + region OCR recovery + advanced structure detection
- **Key features to advertise**: deterministic, multi-pass, selective OCR, section-aware chunking, token estimation, confidence scoring

### From Next Steps Engineering Guide
- **13 priorities** in order:
  1. Stabilize public API (3 entry points: `extract_text`, `extract_json`, `load_llm_context`)
  2. Lock output schema
  3. Add provenance metadata (page_start, page_end, section, confidence, bbox)
  4. Guarantee deterministic extraction (sort blocks by y,x)
  5. Large PDF safety (stream pages, don't load all into memory)
  6. Parallel page processing (ThreadPoolExecutor)
  7. Ship benchmark datasets with repo
  8. Simplify CLI
  9. Add runnable example integrations
  10. Publish package (already done!)
  11. Add CI/CD
  12. Ecosystem integrations (LlamaIndex, LangChain loaders)
  13. Scope discipline (stay focused on extraction + chunking + ingestion)
- **Public API should only expose**: `extract_text(path)`, `extract_json(path)`, `load_llm_context(path)`
- **Provenance per chunk**: page_start, page_end, section, tokens, confidence, optional bbox/block_ids
- **Deterministic sorting**: blocks sorted by `(y_position, x_position)`

---

## Feedback #2: Comprehensive Architecture Plan Written

A full versioned rollout plan has been written to the plan file. Key architectural decisions made:

### Decisions
1. **Flat module structure** — no premature restructuring into core/recovery/structure/. Add modules flat until density justifies grouping.
2. **v0.3 does ONE thing** — multi-pass + RapidOCR. No public API, no schema lock, no benchmarks. Ship the hardest technical risk first.
3. **audit.py as separate module** — not inside pipeline.py. The audit logic is complex enough and will grow.
4. **RapidOCR replaces Surya as default** — `pdfmux[ocr]` becomes RapidOCR, Surya becomes `pdfmux[ocr-heavy]`.
5. **Multi-pass for ALL PDFs in standard mode** — not just graphical. Audit overhead is ~0 when all pages are good.
6. **Tables still route to Docling** — multi-pass doesn't handle tables, Docling does.

### Version sequence
- v0.3.0: Multi-pass + RapidOCR (fix the extraction quality problem)
- v0.4.0: Public API + schema lock (make it usable as a library)
- v0.5.0: Benchmarks (prove it with numbers)
- v0.6.0: Performance (parallel, caching)
- v1.0.0: Production stable (locked interfaces, error codes, ecosystem)

---

## Feedback #3: Implementer Review — 7 Issues Found

Tested RapidOCR end-to-end on a real pitch deck (Drumworks, 10 pages). Confirmed it works. Found 7 issues in the plan that need fixing:

### Real test data (Drumworks pitch deck, page_chunks=True)
```
Page  0: text=  83 chars, images=1  → "bad"  (below 200, has images)
Page  1: text=   0 chars, images=5  → "empty"
Page  2: text=   0 chars, images=2  → "empty"
Page  3: text=  22 chars, images=1  → "bad"
Page  4: text=   0 chars, images=2  → "empty"
Page  5: text= 320 chars, images=5  → "good" ✓
Page  6: text= 250 chars, images=5  → "good" ✓
Page  7: text= 293 chars, images=7  → "good" ✓
Page  8: text=   0 chars, images=3  → "empty"
Page  9: text=  47 chars, images=1  → "bad"

Result: 3 good, 3 bad, 4 empty = 7 pages need OCR
```

RapidOCR on same pages: Page 1 got 308 chars from OCR vs 0 from fast. Page 0: OCR=71 vs Fast=79. Speed: 0.7-2.7s/page at 200 DPI.

### Issue 1: Dependency spec is WRONG
`rapidocr[onnxruntime]` does NOT install onnxruntime as an extra. Tested: `pip install "rapidocr[onnxruntime]"` → onnxruntime not found at runtime.

**Fix**: Use separate deps:
```toml
ocr = ["rapidocr>=3.0.0", "onnxruntime>=1.19.0"]
```

### Issue 2: OCR comparison should be smarter
Plan says: `if len(ocr_text.strip()) > 10: ocr_results[page_num] = ocr_text`

Problem: On Page 0, OCR got 71 chars but fast got 83 chars. We'd replace better fast text with worse OCR text.

**Fix**: For "bad" pages (which have SOME fast text), only use OCR if it's longer:
```python
if audit.quality == "empty":
    # Any OCR text is an improvement over nothing
    if len(ocr_text.strip()) > 10:
        ocr_results[page_num] = ocr_text
elif audit.quality == "bad":
    # Only replace if OCR got more than fast extraction
    if len(ocr_text.strip()) > len(audit.text.strip()):
        ocr_results[page_num] = ocr_text
```

### Issue 3: RapidOCR engine must be created ONCE
Creating `RapidOCR()` loads 3 ONNX models (~1s). Plan creates it before the loop but the code structure in `extract_page()` creates a new engine per call.

**Fix**: RapidOCRExtractor should cache the engine as an instance attribute:
```python
class RapidOCRExtractor:
    def __init__(self):
        from rapidocr import RapidOCR
        self._engine = RapidOCR()
```

### Issue 4: RapidOCR logging noise
RapidOCR outputs noisy INFO logs: `[RapidOCR] base.py:22: Using engine_name: onnxruntime`, model paths, etc.

**Fix**: Suppress in extractor init:
```python
logging.getLogger("RapidOCR").setLevel(logging.WARNING)
```

### Issue 5: _handle_mixed_pdf() is now redundant
Multi-pass already handles mixed PDFs — digital pages pass audit as "good", scanned pages get flagged as "bad"/"empty" and OCR'd. No need for separate mixed handler.

**Fix**: Remove `_handle_mixed_pdf()`. Multi-pass subsumes it.

### Issue 6: _try_ocr_extractor() should try RapidOCR first
Currently the `is_scanned` path calls `_try_ocr_extractor()` which imports surya. With multi-pass handling most cases, this path is rarely hit — but when it IS hit (quality="standard", has_tables=False, is_scanned=True), it should try RapidOCR before Surya.

**Fix**: Update `_try_ocr_extractor()` fallback chain: RapidOCR → Surya → Fast.

### Issue 7: fitz doc opened per-page in extract_page()
`extract_page()` opens and closes the fitz doc for each page. With 7 bad pages, that's 7 open/close cycles.

**Fix for v0.3**: Accept — it's simple and correct. 7 opens is negligible vs OCR time.
**Fix for v0.6**: Parallel processing will need a different pattern anyway.

---
