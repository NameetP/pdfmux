# pdfmux v1.1.0 — Product Improvement Brief

**Date:** March 10, 2026
**Author:** Product Agent
**Based on:** Real-world benchmark of v1.0.1 across 11 public documents (1,422 pages)
**Status:** PROPOSED

---

## Executive Summary

pdfmux v1.0.1 performs well on digital-native PDFs: 7% faster than raw pymupdf4llm, 100% confidence across the test corpus, and graceful error recovery (Airbnb deck). However, real-world testing against SEC filings, pitch decks, and government reports exposed 6 gaps — all in edge-case handling rather than core extraction. The gaps cluster into two themes:

1. **Image-heavy documents are under-served** (gaps 1, 5) — the OCR budget is too conservative, and classification should drive budget decisions.
2. **Detection heuristics miss common patterns** (gaps 2, 3, 6) — table detection fails on financial statements, empty pages are silent, and fast mode has no table support.

None of these gaps require architectural changes. All can be addressed with targeted fixes to `detect.py`, `pipeline.py`, and `audit.py`. The recommended v1.1.0 scope delivers 4 quick wins and 1 medium effort, with the Docling-in-fast-mode feature deferred to v1.2.0.

---

## Gap Analysis

### P0 — OCR budget too conservative for image-heavy documents

**Gaps addressed:** #1 (Airbnb deck under-OCR'd) and #5 (30% budget caps 12-page deck at 3 pages)

**User impact:** CRITICAL. Users processing pitch decks, slide presentations, and image-heavy reports get incomplete output. The Airbnb deck — a 12-page presentation where 6 pages are graphical — only got 3 pages OCR'd in standard mode. This is the single most visible failure in the benchmark. Users who encounter this will conclude pdfmux does not work for presentations.

**Root cause:** `pipeline.py` line 39 sets `OCR_BUDGET_RATIO = 0.30` as a flat rate. The budget calculation at line 369 (`max(1, int(classification.page_count * OCR_BUDGET_RATIO))`) does not account for the document's graphical ratio. A 12-page deck with 50%+ graphical pages should get 100% OCR budget, not 30%.

**Current code path:**
```
pipeline.py:_multipass_extract()
  → max_ocr_pages = max(1, int(classification.page_count * OCR_BUDGET_RATIO))
  → 12 pages * 0.30 = 3 pages max OCR
```

**Files to change:** `pipeline.py`

---

### P1 — Table detection returns False for SEC financial statements

**Gap addressed:** #3 (Tesla 10-K, Berkshire AR, EY guide all miss table detection)

**User impact:** HIGH. SEC filings are a top-3 use case for PDF extraction tools. When `has_tables` returns False, the pipeline skips Docling in standard mode, producing lower-quality table output. Users comparing pdfmux against Docling-native tools will see worse table formatting on financial documents.

**Root cause:** `detect.py:_detect_tables()` uses two heuristics, both of which fail on financial PDFs:

1. **Line-based detection** (lines 121-138): Looks for horizontal/vertical drawing lines. Many SEC filings use whitespace-aligned tables, not drawn grid lines. The threshold requires `horizontal_lines >= 3 AND vertical_lines >= 2` on the same page — too strict for borderless tables.

2. **Text-pattern detection** (lines 140-148): Looks for tab characters or 3+ consecutive double-spaces. Only checks the first 5 pages. Financial tables in 10-K filings often start on page 30+ (after the narrative section).

**Files to change:** `detect.py`

---

### P1 — Standard quality mode times out on files >100 pages

**Gap addressed:** #4 (Docling model loading bottleneck)

**User impact:** HIGH. Four documents in the benchmark corpus exceed 100 pages (Tesla 144p, Berkshire 150p, EY 154p, Uber 522p). The standard quality mode with Docling triggers model loading on every invocation, and the transformer model initialization is the bottleneck. Users processing large documents in standard mode with Docling installed will hit the default 300s timeout.

**Root cause:** `extractors/tables.py:TableExtractor.extract()` creates a new `DocumentConverter()` on every call (line 69). Docling's `DocumentConverter` loads transformer models (~2-5 seconds cold start), and for large documents the full-document processing compounds with this overhead. There is no model caching between invocations.

Additionally, `pipeline.py` routes to Docling for the entire document when `has_tables` is True, rather than extracting only the pages that actually contain tables.

**Files to change:** `extractors/tables.py`, `pipeline.py`

---

### P2 — Empty pages not flagged as warnings in output

**Gap addressed:** #2 (Tesla 2 empty, Berkshire 3 empty, Uber 1 empty — all silent)

**User impact:** MEDIUM. Empty pages are correctly detected internally (the audit pipeline classifies them as "empty"), but the information never reaches the user. In standard mode, empty pages go through the OCR recovery path, and if OCR also finds nothing, they are silently dropped from output. Users processing financial documents do not know that pages were skipped.

**Root cause:** `audit.py:compute_document_confidence()` has sparse/empty page warnings at lines 229-234, but they only trigger when empty pages exceed 15% of total pages. In a 144-page document with 2 empty pages (1.4%), the threshold is never hit. There is no per-page empty warning — only a statistical threshold.

**Files to change:** `audit.py`

---

### P3 — No Docling table extraction in fast mode

**Gap addressed:** #6 (fast mode skips Docling entirely for complex tables)

**User impact:** LOW-MEDIUM. Fast mode is explicitly documented as "PyMuPDF only, skip audit" — users selecting fast mode expect speed over quality. However, for SEC filings with complex tables, the quality difference is meaningful. A "tables-only" Docling pass that targets just detected table pages (without full-document processing) would give users a middle ground.

**Root cause:** `pipeline.py:_route_and_extract()` line 251 routes fast mode directly to `FastExtractor` with no table consideration. There is no concept of a "selective Docling" pass that processes only specific pages.

**Files to change:** `pipeline.py`, `extractors/tables.py`

---

## Proposed Solutions

### Solution 1: Adaptive OCR Budget (P0) — QUICK WIN

**Change:** Make OCR budget proportional to the document's graphical ratio. If >50% of pages are graphical, OCR 100% of pages that need it. If 25-50%, OCR 75%. Below 25%, keep the current 30% budget.

**Implementation in `pipeline.py`:**

```python
# Replace flat OCR_BUDGET_RATIO with adaptive function
def _compute_ocr_budget(classification: PDFClassification) -> float:
    """Compute OCR budget based on document classification.

    Image-heavy documents (>50% graphical) get full OCR.
    Mixed documents (25-50% graphical) get 75%.
    Digital-dominant (<25% graphical) keep the default 30%.
    """
    if classification.page_count == 0:
        return 0.30

    graphical_ratio = len(classification.graphical_pages) / classification.page_count

    if graphical_ratio > 0.50:
        return 1.0   # Full OCR for image-heavy docs
    elif graphical_ratio > 0.25:
        return 0.75  # Higher budget for mixed docs
    else:
        return float(os.environ.get("PDFMUX_OCR_BUDGET", "0.30"))
```

Then in `_multipass_extract()`, replace:
```python
max_ocr_pages = max(1, int(classification.page_count * OCR_BUDGET_RATIO))
```
with:
```python
budget = _compute_ocr_budget(classification)
max_ocr_pages = max(1, int(classification.page_count * budget))
```

**Effort:** ~30 minutes. Single function addition + one line change in `_multipass_extract()`.

**Impact:** Fixes the Airbnb deck completely (12 pages * 100% = 12 pages OCR'd) and all similar image-heavy documents.

---

### Solution 2: Improved Table Detection (P1) — QUICK WIN

**Change:** Add three new heuristics to `_detect_tables()`:

1. **Number-dense line detection** — financial tables have lines with 3+ numbers separated by whitespace (e.g., `Revenue 1,234 1,456 1,789`).
2. **Column alignment detection** — check if numbers in consecutive lines share similar x-positions (column alignment without grid lines).
3. **Deeper page sampling** — scan pages at 25%, 50%, 75% of the document, not just the first 5 pages.

**Implementation in `detect.py`:**

```python
def _detect_tables(doc: fitz.Document) -> bool:
    """Heuristic table detection using line analysis + content patterns."""
    # ... existing line-based and text-pattern heuristics ...

    # NEW: Sample pages throughout the document, not just first 5
    sample_pages = _get_sample_pages(doc)

    for page_num in sample_pages:
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split("\n")

        # NEW: Number-dense line detection (financial tables)
        number_dense_lines = 0
        for line in lines:
            # Count numeric tokens (including formatted numbers like 1,234)
            nums = re.findall(r'[\d,]+\.?\d*', line)
            if len(nums) >= 3:
                number_dense_lines += 1

        if number_dense_lines >= 5:
            return True

    return False

def _get_sample_pages(doc: fitz.Document) -> list[int]:
    """Get sample pages spread throughout the document."""
    n = len(doc)
    if n <= 10:
        return list(range(n))
    # First 5 + quartile samples
    samples = set(range(min(5, n)))
    for frac in [0.25, 0.50, 0.75]:
        idx = int(n * frac)
        samples.update(range(idx, min(idx + 3, n)))
    return sorted(samples)
```

**Effort:** ~1 hour. New heuristic functions + updated sampling strategy.

**Impact:** Fixes table detection for Tesla 10-K, Berkshire AR, EY guide, and similar financial documents.

---

### Solution 3: Empty Page Warnings (P2) — QUICK WIN

**Change:** Always emit a warning when any empty pages are detected, regardless of the percentage threshold. Add the specific page numbers to the warning message.

**Implementation in `audit.py:compute_document_confidence()`:**

Replace the current empty-page warning logic (lines 229-234):
```python
# Current: threshold-based
if empty > 0 and empty / len(pages) > 0.15:
    warnings.append(f"{empty} pages appear to have no extractable text")
```

With:
```python
# New: always warn on empty pages, include page numbers
empty_page_nums = [p.page_num + 1 for p in pages if p.char_count < 20]
if empty_page_nums:
    page_list = ", ".join(str(p) for p in empty_page_nums[:10])
    suffix = f" (and {len(empty_page_nums) - 10} more)" if len(empty_page_nums) > 10 else ""
    warnings.append(
        f"{len(empty_page_nums)} empty page(s) detected: {page_list}{suffix}"
    )
```

Keep the sparse-page warning but make it additive (don't duplicate with empty warning):
```python
sparse_non_empty = [p for p in pages if 20 <= p.char_count < 100]
if len(sparse_non_empty) > 0 and len(sparse_non_empty) / len(pages) > 0.25:
    warnings.append(f"{len(sparse_non_empty)} pages have very little text")
```

**Effort:** ~15 minutes. Replace 4 lines in `audit.py`.

**Impact:** Users always know when pages had no extractable content. Helpful for QA on financial documents with intentional blank pages.

---

### Solution 4: Lazy Docling Model Loading (P1) — MEDIUM EFFORT

**Change:** Cache the `DocumentConverter` instance at module level so that model loading happens once per process lifetime, not once per invocation. Add a page-level routing option so Docling only processes pages that the classifier flagged as having tables.

**Implementation in `extractors/tables.py`:**

```python
# Module-level singleton
_CONVERTER: DocumentConverter | None = None
_CONVERTER_LOCK = threading.Lock()

def _get_converter() -> DocumentConverter:
    """Get or create the cached DocumentConverter."""
    global _CONVERTER
    if _CONVERTER is None:
        with _CONVERTER_LOCK:
            if _CONVERTER is None:
                from docling.document_converter import DocumentConverter
                _CONVERTER = DocumentConverter()
    return _CONVERTER
```

And in the `extract()` method:
```python
converter = _get_converter()  # instead of DocumentConverter()
```

**For page-level Docling in `pipeline.py`:** When the document has tables and is large (>50 pages), extract only the pages classified as having tables with Docling, and use PyMuPDF for the rest. This avoids full-document Docling processing on 500+ page SEC filings.

```python
# In _route_and_extract(), replace the current table routing:
if classification.has_tables and not classification.is_graphical:
    if classification.page_count > 50:
        # Large doc: selective Docling on table pages only
        return _selective_table_extract(file_path, classification)
    else:
        pages, name = _try_table_extractor(file_path)
        return pages, name, []
```

**Effort:** ~2-3 hours. Singleton pattern + selective page routing + tests.

**Impact:** Eliminates timeout on large documents. The Uber S-1 (522 pages) and similar large filings will complete within the default 300s timeout. Repeated invocations in the same process (MCP server, batch mode) will skip model loading entirely.

---

### Solution 5: Docling in Fast Mode for Tables (P3) — DEFERRED to v1.2.0

**Change:** Add a "tables-only" mode that runs Docling on just the pages detected as having tables, without full-document processing. This would be available in fast mode as an opt-in flag (`--tables`).

**Rationale for deferral:** This requires significant work on Docling's page-level extraction API (Docling currently processes full documents, not individual pages). The selective table extraction in Solution 4 partially addresses this for standard mode. Fast mode users have explicitly chosen speed; adding Docling would contradict that contract.

**Effort:** ~1-2 days. Requires Docling API investigation + new routing logic.

---

## v1.1.0 Scope

### Ships in v1.1.0 (4 quick wins + 1 medium effort)

| # | Change | Priority | Effort | Files |
|---|--------|----------|--------|-------|
| 1 | Adaptive OCR budget based on graphical ratio | P0 | 30 min | `pipeline.py` |
| 2 | Improved table detection (number-dense lines, deeper sampling) | P1 | 1 hr | `detect.py` |
| 3 | Always emit empty page warnings with page numbers | P2 | 15 min | `audit.py` |
| 4 | Lazy Docling model caching (singleton converter) | P1 | 2 hr | `extractors/tables.py` |
| 5 | Selective Docling for large documents (>50 pages) | P1 | 1 hr | `pipeline.py` |

**Total estimated effort:** ~5 hours of implementation + 2 hours of testing.

### Deferred to v1.2.0

| # | Change | Priority | Reason |
|---|--------|----------|--------|
| 6 | Docling table extraction in fast mode | P3 | Contradicts fast mode contract; needs Docling API work |

---

## Implementation Notes

### File-by-file changes

**`pipeline.py`**
- Add `_compute_ocr_budget(classification)` function (new, ~15 lines)
- Replace `OCR_BUDGET_RATIO` usage in `_multipass_extract()` with adaptive call
- Add `_selective_table_extract()` function for large documents with tables
- Update `_route_and_extract()` table routing to use selective extraction on large docs
- Keep `OCR_BUDGET_RATIO` constant as the default fallback for non-graphical docs
- Keep `PDFMUX_OCR_BUDGET` env var as an escape hatch (still respected for digital-dominant docs)

**`detect.py`**
- Add `_get_sample_pages(doc)` helper for distributed page sampling
- Add number-dense line detection to `_detect_tables()`
- Expand the first-5-pages scan to include quartile samples
- Add `import re` at top of file (currently not imported)

**`audit.py`**
- In `compute_document_confidence()`, replace threshold-based empty warning with always-warn
- Include specific page numbers in the warning message (capped at 10 to avoid noise)
- Separate sparse-page warning from empty-page warning to avoid duplication

**`extractors/tables.py`**
- Add module-level `_CONVERTER` singleton with thread-safe lazy initialization
- Replace `DocumentConverter()` in `extract()` with `_get_converter()` call
- Add `import threading` at top

### Testing strategy

1. **Adaptive OCR budget:** Unit test with mock classification at different graphical ratios (0%, 30%, 60%, 100%). Integration test with the actual Airbnb deck — verify all 12 pages are OCR'd.
2. **Table detection:** Unit test with extracted text from Tesla 10-K page 35 (financial statements). Verify `_detect_tables` returns True. Test with non-table documents to avoid false positives.
3. **Empty page warnings:** Unit test `compute_document_confidence()` with a page list containing 2 empty pages in a 150-page document. Verify warning is emitted with page numbers.
4. **Docling caching:** Integration test that calls `_get_converter()` twice and verifies the same instance is returned. Performance test to verify cold start is ~2-5s but second invocation is <10ms.
5. **Selective Docling:** Integration test with a >50 page document that has tables. Verify only table pages go through Docling and total processing time is reduced.

### Benchmark validation

After implementation, re-run the full 11-document benchmark suite to verify:
- Airbnb deck extracts significantly more content (target: >10,000 chars vs current 2,766)
- Tesla, Berkshire, EY now show `has_tables: True` in classification
- Empty pages in Tesla, Berkshire, Uber emit warnings in output
- Standard mode completes on all 11 documents within 300s timeout
- No regressions in speed or confidence on the existing passing documents

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Adaptive OCR budget increases processing time on image-heavy docs | Medium | Expected and acceptable — users get better output. Add timing info to warnings. |
| Number-dense table detection produces false positives | Low | Threshold of 5+ number-dense lines is conservative. Test against narrative-heavy docs. |
| Docling singleton causes memory issues in long-running MCP server | Low | DocumentConverter is designed for reuse. Add `pdfmux doctor` memory reporting. |
| Selective Docling produces inconsistent formatting (mix of PyMuPDF + Docling) | Medium | Normalize output through existing postprocess.py. Add integration tests. |

---

## Success Criteria for v1.1.0

1. Airbnb pitch deck produces >8,000 chars in standard mode (up from 2,766)
2. Tesla 10-K, Berkshire AR, EY guide all report `has_tables: True`
3. All 6 empty pages across the corpus emit warnings in output
4. Uber S-1 (522 pages) completes in standard mode within 300s
5. No regressions: all 11 benchmark documents maintain 100% confidence
6. Benchmark re-run shows equal or better speed on all documents
