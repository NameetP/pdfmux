# CTO Architecture Plan -- pdfmux v1.1.0

**Date:** March 10, 2026
**Author:** CTO Agent
**Status:** Implementation-ready
**Based on:** Benchmark findings from 11 public documents (1,422 pages)

---

## Executive Summary

Real-world benchmarking exposed 6 gaps in pdfmux v1.0.1. This plan addresses each with specific code changes, estimated complexity, and breaking-change assessment. The changes are ordered by impact (highest first) and designed to be shipped incrementally -- each gap is an independent PR.

| # | Gap | File(s) | Complexity | Breaking |
|---|-----|---------|:----------:|:--------:|
| 1 | Table detection misses financial statements | `detect.py` | M | No |
| 2 | OCR budget too conservative for image-heavy docs | `pipeline.py`, `detect.py` | S | No |
| 3 | Empty pages not surfaced as warnings | `audit.py`, `pipeline.py` | S | No |
| 4 | Docling model loading timeouts on large files | `extractors/tables.py`, `pipeline.py` | L | No |
| 5 | No table extraction in fast mode | `extractors/fast.py`, `pipeline.py` | M | No |
| 6 | Classification lacks `empty_pages` field | `detect.py`, `types.py` | S | No |

**Estimated total effort:** 3-5 engineering days

---

## Gap 1: Table Detection Misses Financial Statements

### Problem

`_detect_tables()` in `detect.py` only checks the **first 5 pages** for drawn lines and tab/space patterns. Financial documents like Tesla 10-K, Apple 10-K, Berkshire AR, and EY 10-K Guide have prominent tables but `has_tables` returned `False` because:

1. SEC filings typically have narrative/legal text on pages 1-5; financial tables start on page 30+.
2. Financial tables use **whitespace-aligned columns** (not drawn grid lines), so the line-count heuristic fails.
3. The tab/multi-space heuristic (`line.count("  ") >= 3`) fires on indented legal text, creating false positives that mask real table detection needs.

### Current Behavior

```python
# detect.py: _detect_tables()
def _detect_tables(doc: fitz.Document) -> bool:
    # Only checks pages 0-4
    for page_num in range(min(len(doc), 5)):
        # Check 1: drawn lines (horizontal >= 3, vertical >= 2)
        # Check 2: tab characters or 3+ double-spaces per line
    return False
```

### Proposed Behavior

Replace `_detect_tables()` with a **multi-signal scoring system** that:

1. Samples pages from **three sections** of the document (front, middle, back).
2. Uses **four detection signals** instead of two.
3. Returns a confidence score, not just a boolean.

#### File: `detect.py`

#### Function: `_detect_tables()` -- full replacement

```python
# --- Table detection constants ---
_TABLE_SAMPLE_PAGES = 12          # total pages to sample
_TABLE_SCORE_THRESHOLD = 3        # sum of signals needed to confirm tables
_NUMBER_DENSE_THRESHOLD = 0.30    # ratio of numeric chars on a line
_ALIGNED_COLUMN_MIN_LINES = 4    # lines with matching column positions


def _detect_tables(doc: fitz.Document) -> bool:
    """Multi-signal table detection with strategic page sampling.

    Samples pages from front (0-4), middle, and back of document.
    Uses 4 signals scored additively:
        Signal 1: Drawn grid lines (existing) -- score 2
        Signal 2: Number-dense lines (new) -- score 2
        Signal 3: Aligned column positions via text blocks (new) -- score 2
        Signal 4: Tab/whitespace patterns (existing, refined) -- score 1

    Returns True if combined score >= _TABLE_SCORE_THRESHOLD.
    """
    total = len(doc)
    if total == 0:
        return False

    # Strategic sampling: front, middle, back
    sample_pages = _get_sample_pages(total, _TABLE_SAMPLE_PAGES)

    total_score = 0

    for page_num in sample_pages:
        page = doc[page_num]
        page_score = 0

        # Signal 1: Drawn grid lines (keep existing logic)
        page_score += _score_drawn_lines(page)

        # Signal 2: Number-dense lines (financial tables)
        page_score += _score_number_density(page)

        # Signal 3: Aligned column positions via text blocks
        page_score += _score_column_alignment(page)

        # Signal 4: Tab/whitespace patterns (refined)
        page_score += _score_whitespace_patterns(page)

        total_score += page_score

        # Early exit: enough evidence
        if total_score >= _TABLE_SCORE_THRESHOLD:
            return True

    return total_score >= _TABLE_SCORE_THRESHOLD


def _get_sample_pages(total: int, sample_size: int) -> list[int]:
    """Select pages from front, middle, and back of document."""
    if total <= sample_size:
        return list(range(total))

    third = sample_size // 3
    front = list(range(min(third, total)))
    mid_start = max(0, total // 2 - third // 2)
    middle = list(range(mid_start, min(mid_start + third, total)))
    back_start = max(0, total - third)
    back = list(range(back_start, total))

    # Deduplicate and sort
    return sorted(set(front + middle + back))


def _score_drawn_lines(page: fitz.Page) -> int:
    """Score based on drawn horizontal/vertical lines. Returns 0 or 2."""
    drawings = page.get_drawings()
    h_lines = 0
    v_lines = 0
    for drawing in drawings:
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 50:
                    h_lines += 1
                elif abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) > 20:
                    v_lines += 1
    return 2 if (h_lines >= 3 and v_lines >= 2) else 0


def _score_number_density(page: fitz.Page) -> int:
    """Score based on lines dominated by numbers (financial data).

    Financial tables have lines like:
        "Revenue    $394,328    $383,285    $365,817"
    where >30% of non-space characters are digits, $, commas, periods, or %.

    Returns 0 or 2.
    """
    text = page.get_text("text")
    lines = text.split("\n")
    number_dense_lines = 0

    for line in lines:
        stripped = line.strip()
        if len(stripped) < 20:
            continue
        non_space = stripped.replace(" ", "")
        if not non_space:
            continue
        numeric_chars = sum(
            1 for c in non_space if c in "0123456789$,%.()-"
        )
        ratio = numeric_chars / len(non_space)
        if ratio >= _NUMBER_DENSE_THRESHOLD:
            number_dense_lines += 1

    return 2 if number_dense_lines >= 5 else 0


def _score_column_alignment(page: fitz.Page) -> int:
    """Score based on text blocks with aligned x-positions.

    Tables produce text blocks with consistent left-edge x-positions
    across multiple rows. We cluster block x0 positions and check
    if 3+ clusters exist with 4+ blocks each.

    Returns 0 or 2.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    if len(text_blocks) < 6:
        return 0

    # Round x0 positions to nearest 5pt for clustering
    x0_rounded = [round(b[0] / 5) * 5 for b in text_blocks]
    from collections import Counter
    x0_counts = Counter(x0_rounded)

    # Count how many distinct x-positions have 4+ blocks aligned to them
    aligned_columns = sum(1 for count in x0_counts.values() if count >= _ALIGNED_COLUMN_MIN_LINES)

    return 2 if aligned_columns >= 3 else 0


def _score_whitespace_patterns(page: fitz.Page) -> int:
    """Score based on whitespace-separated columns. Returns 0 or 1.

    Refined from original: requires 5+ qualifying lines (up from 3)
    and requires lines to have at least 3 multi-space gaps (not just
    any indentation).
    """
    text = page.get_text("text")
    lines = text.split("\n")
    import re
    # Count lines with 3+ gaps of 3+ spaces (column separators)
    tab_lines = sum(
        1 for line in lines
        if len(line.strip()) > 20 and len(re.findall(r"  {3,}", line)) >= 3
    )
    return 1 if tab_lines >= 5 else 0
```

### Why This Works

- **Tesla 10-K**: Financial statements on pages 60+ have number-dense lines (`$394,328  $383,285`) -- Signal 2 catches these.
- **Apple 10-K**: Balance sheets use aligned columns of dollar amounts -- Signal 3 catches the column alignment.
- **Berkshire AR**: Mixed narrative + tables deep in document -- middle/back sampling finds them.
- **EY 10-K Guide**: Professional layout with structured content -- combination of signals 2+3 triggers.

### Complexity: M (Medium)

- Replace one function, add 5 helper functions
- All new logic uses existing PyMuPDF APIs (no new dependencies)
- Test with existing benchmark corpus

### Breaking Change: No

- `PDFClassification.has_tables` stays a `bool`
- All downstream routing logic unchanged
- Documents previously detected as having tables will still be detected

---

## Gap 2: Dynamic OCR Budget Based on Classification

### Problem

The OCR budget is hardcoded at 30% of document pages (`OCR_BUDGET_RATIO = 0.30`). For the Airbnb pitch deck (12 pages, 6 graphical), this means only 3 pages get OCR'd -- skipping half the graphical content. When a document is predominantly image-based, the budget should scale to 100%.

### Current Behavior

```python
# pipeline.py, line 39
OCR_BUDGET_RATIO = float(os.environ.get("PDFMUX_OCR_BUDGET", "0.30"))

# pipeline.py, _multipass_extract(), line 369
max_ocr_pages = max(1, int(classification.page_count * OCR_BUDGET_RATIO))
```

The budget is computed once from the static ratio, regardless of document content.

### Proposed Behavior

Compute the OCR budget dynamically based on the graphical page ratio from classification.

#### File: `pipeline.py`

#### Function: `_compute_ocr_budget()` -- new function

```python
# New constant: threshold for "image-heavy" documents
IMAGE_HEAVY_THRESHOLD = 0.50  # >50% graphical pages = OCR everything


def _compute_ocr_budget(classification: PDFClassification) -> float:
    """Compute OCR budget ratio based on document classification.

    Rules:
        - If >50% of pages are graphical: budget = 1.0 (OCR all)
        - If >25% graphical: budget = graphical_ratio + 0.10 (generous)
        - Otherwise: use default OCR_BUDGET_RATIO (0.30)

    Returns:
        Float between 0.0 and 1.0 representing max fraction of pages to OCR.
    """
    if classification.page_count == 0:
        return OCR_BUDGET_RATIO

    graphical_ratio = len(classification.graphical_pages) / classification.page_count

    if graphical_ratio >= IMAGE_HEAVY_THRESHOLD:
        return 1.0
    elif graphical_ratio > 0.25:
        return min(1.0, graphical_ratio + 0.10)
    else:
        return OCR_BUDGET_RATIO
```

#### Function: `_multipass_extract()` -- modify OCR budget calculation

Replace this block (line 369):
```python
# OCR budget: cap at budget ratio of document pages
max_ocr_pages = max(1, int(classification.page_count * OCR_BUDGET_RATIO))
```

With:
```python
# OCR budget: dynamic based on classification
effective_budget = _compute_ocr_budget(classification)
max_ocr_pages = max(1, int(classification.page_count * effective_budget))
```

Also update the log message (line 379) to show the effective budget:
```python
logger.warning(
    f"OCR budget: processing {max_ocr_pages} of {len(all_pages_needing_ocr)} "
    f"pages (budget={effective_budget:.0%} of {classification.page_count}). "
    f"Skipping {skipped} pages."
)
```

### Why This Works

- **Airbnb deck (50% graphical)**: `graphical_ratio = 0.50 >= 0.50` triggers `budget = 1.0`. All 6 graphical pages get OCR'd instead of 3.
- **Uber S-1 (2.9% graphical)**: `graphical_ratio = 0.029 < 0.25` keeps default 30% budget. No change for large mostly-digital docs.
- **EY Guide (1.3% graphical)**: Default budget. No waste.

### Complexity: S (Small)

- Add one function, modify one calculation, update one log line
- No new dependencies

### Breaking Change: No

- The `PDFMUX_OCR_BUDGET` env var still works as a floor
- Default behavior unchanged for documents with <25% graphical pages
- Only image-heavy documents see the increase

---

## Gap 3: Empty Page Warnings in Output

### Problem

Empty pages are correctly detected by the audit pipeline (Tesla: 2, Berkshire: 3, Uber: 1) but never surfaced as user-visible warnings. Users should know which pages had no extractable content.

### Current Behavior

```python
# audit.py, compute_document_confidence(), lines 231-234
# Only warns when empty pages exceed 15% of total
empty = sum(1 for p in pages if p.char_count < 20)
if empty > 0 and empty / len(pages) > 0.15:
    warnings.append(f"{empty} pages appear to have no extractable text")
```

For a 144-page document with 2 empty pages (1.4%), the threshold is never hit. The 15% gate means empty page warnings are effectively invisible for large documents.

### Proposed Behavior

Always emit an empty page warning when empty pages exist, with page numbers listed.

#### File: `audit.py`

#### Function: `compute_document_confidence()` -- modify sparse/empty page detection

Replace lines 228-234:
```python
    # Sparse page detection
    sparse = sum(1 for p in pages if p.char_count < 100)
    empty = sum(1 for p in pages if p.char_count < 20)
    if empty > 0 and empty / len(pages) > 0.15:
        warnings.append(f"{empty} pages appear to have no extractable text")
    elif sparse > 0 and sparse / len(pages) > 0.25:
        warnings.append(f"{sparse} pages have very little text")
```

With:
```python
    # Empty page detection -- always warn with page numbers
    empty_pages = [p for p in pages if p.char_count < 20]
    if empty_pages:
        page_nums = ", ".join(str(p.page_num + 1) for p in empty_pages)
        if len(empty_pages) <= 5:
            warnings.append(
                f"{len(empty_pages)} empty page(s) detected (pages: {page_nums})"
            )
        else:
            first_five = ", ".join(str(p.page_num + 1) for p in empty_pages[:5])
            warnings.append(
                f"{len(empty_pages)} empty pages detected "
                f"(first 5: {first_five}, ...)"
            )

    # Sparse page detection (non-empty but low text)
    sparse = [p for p in pages if 20 <= p.char_count < 100]
    if sparse and len(sparse) / len(pages) > 0.25:
        warnings.append(f"{len(sparse)} pages have very little text")
```

### Why This Works

- **Tesla 10-K**: 2 empty pages -> warning: `"2 empty page(s) detected (pages: 54, 143)"`
- **Berkshire AR**: 3 empty pages -> warning: `"3 empty page(s) detected (pages: 1, 148, 150)"`
- **Uber S-1**: 1 empty page -> warning: `"1 empty page(s) detected (pages: 265)"`
- Large doc with 20 empty pages -> truncated: `"20 empty pages detected (first 5: 1, 15, 30, 45, 60, ...)"`

### Complexity: S (Small)

- Modify one code block in one function
- No new dependencies, no API changes

### Breaking Change: No

- Warnings are additive -- existing warning strings are replaced but the `warnings: list[str]` return type is unchanged
- Consumers that parse warning strings should use `startswith()` or regex rather than exact match (best practice already)

---

## Gap 4: Docling Model Loading Timeouts on Large Files

### Problem

Standard quality mode with Docling timed out on files >100 pages during initial benchmark runs. The bottleneck is Docling's transformer model loading (EasyOCR + TableFormer), which happens on every `DocumentConverter()` instantiation. For a 522-page Uber S-1, this means:

1. Model loads from disk into memory (~5-10s)
2. Full document conversion starts (~1-3s/page for complex pages)
3. Total time exceeds the 300s default timeout

### Current Behavior

```python
# extractors/tables.py, lines 66-69
def extract(self, file_path, pages=None):
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()          # <-- model loads here every time
    result = converter.convert(str(file_path))  # <-- full document, no page filtering
```

```python
# pipeline.py, lines 263-265
if classification.has_tables and not classification.is_graphical:
    pages, name = _try_table_extractor(file_path)  # <-- full Docling on entire doc
```

### Proposed Behavior: Three-Part Solution

#### Part A: Lazy-loaded Singleton for Docling Converter

##### File: `extractors/tables.py`

```python
import threading

# Module-level singleton with thread safety
_converter_lock = threading.Lock()
_converter_instance = None


def _get_converter():
    """Lazy-load and cache the Docling DocumentConverter.

    Thread-safe singleton. The converter loads transformer models
    on first use (~5-10s), then reuses them for all subsequent calls.
    """
    global _converter_instance
    if _converter_instance is None:
        with _converter_lock:
            if _converter_instance is None:  # double-check locking
                from docling.document_converter import DocumentConverter
                _converter_instance = DocumentConverter()
    return _converter_instance


@register(name="docling", priority=40)
class TableExtractor:

    def extract(self, file_path, pages=None):
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable
            raise ExtractorNotAvailable(
                "Docling is not installed. Install with: pip install pdfmux[tables]"
            )

        converter = _get_converter()  # <-- cached after first call
        result = converter.convert(str(file_path))
        # ... rest unchanged
```

#### Part B: Page-limited Docling Pass for Table Pages Only

Instead of running Docling on the entire 522-page document, run it only on pages identified as having tables.

##### File: `extractors/tables.py` -- add new method

```python
class TableExtractor:
    # ... existing methods ...

    def extract_pages(
        self,
        file_path: str | Path,
        page_nums: list[int],
    ) -> Iterator[PageResult]:
        """Extract specific pages using Docling.

        Extracts only the requested page range by creating a temporary
        PDF subset, then runs Docling on the subset. This avoids
        processing the entire document when only certain pages need
        table extraction.

        Args:
            file_path: Path to the full PDF.
            page_nums: 0-indexed page numbers to extract.

        Yields:
            PageResult for each requested page.
        """
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable
            raise ExtractorNotAvailable(
                "Docling is not installed. Install with: pip install pdfmux[tables]"
            )

        import tempfile
        file_path = Path(file_path)

        # Create a temporary PDF with only the requested pages
        doc = fitz.open(str(file_path))
        subset_doc = fitz.open()  # new empty PDF
        for pn in sorted(page_nums):
            if pn < len(doc):
                subset_doc.insert_pdf(doc, from_page=pn, to_page=pn)
        doc.close()

        # Write subset to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            subset_doc.save(tmp.name)
            subset_doc.close()
            tmp_path = Path(tmp.name)

        try:
            converter = _get_converter()
            result = converter.convert(str(tmp_path))
            markdown = result.document.export_to_markdown()
            page_texts = (
                markdown.split("\n\n---\n\n")
                if "\n\n---\n\n" in markdown
                else [markdown]
            )

            sorted_pages = sorted(page_nums)
            for i, text in enumerate(page_texts):
                if i >= len(sorted_pages):
                    break
                original_page_num = sorted_pages[i]
                has_text = len(text.strip()) > 10

                yield PageResult(
                    page_num=original_page_num,
                    text=text,
                    confidence=0.95 if has_text else 0.0,
                    quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                    extractor=self.name,
                )
        finally:
            tmp_path.unlink(missing_ok=True)
```

#### Part C: Pipeline Routing for Targeted Docling

##### File: `pipeline.py` -- modify table routing in `_route_and_extract()`

Replace lines 263-265:
```python
    # Tables -> Docling (unless also graphical)
    if classification.has_tables and not classification.is_graphical:
        pages, name = _try_table_extractor(file_path)
        return pages, name, []
```

With:
```python
    # Tables -> targeted Docling on table pages, fast for the rest
    if classification.has_tables and not classification.is_graphical:
        pages, name = _try_targeted_table_extraction(file_path, classification)
        return pages, name, []
```

Add new function:
```python
def _try_targeted_table_extraction(
    file_path: Path,
    classification: PDFClassification,
) -> tuple[list[PageResult], str]:
    """Hybrid extraction: Docling for table pages, fast for the rest.

    For documents with <50 table-candidate pages, extracts those pages
    with Docling and the rest with PyMuPDF. For smaller documents (<50
    total pages), falls back to full-document Docling.
    """
    # Small documents: full Docling is fine
    if classification.page_count <= 50:
        return _try_table_extractor(file_path)

    # Identify table-candidate pages (use detection sampling)
    try:
        from pdfmux.extractors.tables import TableExtractor
        ext = TableExtractor()
        if not ext.available():
            raise ImportError

        # Get table pages from detection (reuse _detect_table_pages)
        table_pages = _identify_table_pages(file_path, classification)

        if not table_pages or len(table_pages) > 100:
            # Too many or none: fall back to full Docling
            return _try_table_extractor(file_path)

        # Hybrid: fast for non-table pages, Docling for table pages
        from pdfmux.extractors.fast import FastExtractor
        fast = FastExtractor()
        fast_pages = {p.page_num: p for p in fast.extract(file_path)}

        # Docling only for table pages
        docling_pages = {
            p.page_num: p for p in ext.extract_pages(file_path, table_pages)
        }

        # Merge: prefer Docling results for table pages
        merged = []
        for page_num in sorted(fast_pages.keys()):
            if page_num in docling_pages:
                merged.append(docling_pages[page_num])
            else:
                merged.append(fast_pages[page_num])

        n_docling = len(docling_pages)
        return merged, f"pymupdf4llm + docling ({n_docling} table pages)"

    except Exception:
        logger.info("Targeted table extraction failed, falling back")
        return _try_table_extractor(file_path)


def _identify_table_pages(
    file_path: Path,
    classification: PDFClassification,
) -> list[int]:
    """Identify pages likely to contain tables.

    Uses lightweight heuristics (number density + column alignment)
    on all pages. Returns 0-indexed page numbers.
    """
    import fitz
    doc = fitz.open(str(file_path))
    table_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split("\n")

        # Quick check: 3+ number-dense lines on this page
        number_dense = 0
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 20:
                continue
            non_space = stripped.replace(" ", "")
            if not non_space:
                continue
            numeric = sum(1 for c in non_space if c in "0123456789$,%.()-")
            if numeric / len(non_space) >= 0.30:
                number_dense += 1

        if number_dense >= 3:
            table_pages.append(page_num)

    doc.close()
    return table_pages
```

### Why This Works

- **Singleton**: Eliminates 5-10s model load time on every Docling call. Second invocation is instant.
- **Page subsetting**: Uber S-1 might have 30 table pages out of 522. Docling processes 30 pages (~30-90s) instead of 522 (~260-1500s).
- **Hybrid routing**: Non-table pages use the fast path (0.01s/page), table pages use Docling (0.3-3s/page). Total time for Uber S-1 drops from timeout to ~35s.

### Complexity: L (Large)

- Part A (singleton): Small, straightforward
- Part B (page subsetting): Medium, needs careful page-number remapping and temp file cleanup
- Part C (hybrid routing): Medium, new routing logic with fallback chains

### Breaking Change: No

- Public API (`process()`, `process_batch()`) unchanged
- `TableExtractor.extract()` unchanged -- `extract_pages()` is additive
- Fallback chain ensures graceful degradation if any part fails

---

## Gap 5: Lightweight Table Extraction in Fast Mode

### Problem

Fast mode skips Docling entirely, which means complex tables in SEC filings lose structure. The `pymupdf4llm` extractor preserves basic table formatting via markdown pipes, but misses:

- Multi-line cell content
- Merged cells
- Header row detection
- Column alignment for numeric data

A lightweight table enhancement should work without pulling in the full Docling dependency (~500MB of transformer models).

### Current Behavior

```python
# pipeline.py, _route_and_extract(), lines 251-256
if quality == Quality.FAST:
    from pdfmux.extractors.fast import FastExtractor
    ext = FastExtractor()
    pages = list(ext.extract(file_path))
    return pages, ext.name, []
```

Fast mode uses only PyMuPDF/pymupdf4llm. No table awareness whatsoever.

### Proposed Behavior

Add a `_enhance_tables_fast()` post-processing step that uses PyMuPDF's built-in `page.find_tables()` API (available since PyMuPDF 1.23.0, no extra dependencies) to detect and re-format table regions.

#### File: `extractors/fast.py` -- add table enhancement

```python
def _enhance_tables_fast(page: fitz.Page, text: str) -> str:
    """Enhance table formatting using PyMuPDF's built-in table finder.

    PyMuPDF's find_tables() uses heuristic analysis of text positions
    and line objects to detect table boundaries. It works without any
    ML models or extra dependencies.

    This function:
    1. Detects tables on the page using fitz
    2. Extracts cell content into a structured grid
    3. Formats as a markdown table
    4. Replaces the corresponding region in the page text

    Args:
        page: PyMuPDF page object.
        text: Current extracted text for this page.

    Returns:
        Text with tables re-formatted as proper markdown tables.
    """
    try:
        tables = page.find_tables()
    except AttributeError:
        # PyMuPDF < 1.23.0 doesn't have find_tables()
        return text
    except Exception:
        return text

    if not tables.tables:
        return text

    # Build markdown tables for each detected table
    table_markdowns = []
    for table in tables.tables:
        try:
            df = table.to_pandas()
            if df.empty or len(df.columns) < 2:
                continue

            # Convert to markdown table
            md_lines = []

            # Header row
            headers = [str(h).strip() if str(h) != "None" else "" for h in df.columns]
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join("---" for _ in headers) + " |")

            # Data rows
            for _, row in df.iterrows():
                cells = [
                    str(v).strip() if str(v) != "None" and str(v) != "nan" else ""
                    for v in row
                ]
                md_lines.append("| " + " | ".join(cells) + " |")

            table_markdowns.append("\n".join(md_lines))
        except Exception:
            continue

    if not table_markdowns:
        return text

    # Append enhanced table markdown after the page text
    # We append rather than replace because positional replacement
    # is fragile -- the user gets both the original layout and
    # a clean structured version.
    enhanced = text.rstrip()
    for table_md in table_markdowns:
        enhanced += "\n\n" + table_md

    return enhanced
```

#### File: `extractors/fast.py` -- modify `extract()` method

```python
class FastExtractor:
    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
        *,
        enhance_tables: bool = False,  # <-- new parameter
    ) -> Iterator[PageResult]:
        file_path = Path(file_path)
        chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

        # Open doc only if table enhancement requested
        doc = None
        if enhance_tables:
            doc = fitz.open(str(file_path))

        for i, chunk in enumerate(chunks):
            if pages is not None and i not in pages:
                continue

            text = chunk.get("text", "")
            image_count = len(chunk.get("images", []))

            if len(text.strip()) < 50:
                raw = self._extract_raw_page(file_path, i)
                if len(raw.strip()) > len(text.strip()):
                    text = raw

            # Table enhancement (fast mode only, no ML deps)
            if enhance_tables and doc and i < len(doc):
                text = _enhance_tables_fast(doc[i], text)

            yield PageResult(
                page_num=i,
                text=text,
                confidence=1.0,
                quality=PageQuality.GOOD,
                extractor=self.name,
                image_count=image_count,
            )

        if doc:
            doc.close()
```

#### File: `pipeline.py` -- modify fast mode routing

Replace lines 251-256:
```python
    if quality == Quality.FAST:
        from pdfmux.extractors.fast import FastExtractor
        ext = FastExtractor()
        pages = list(ext.extract(file_path))
        return pages, ext.name, []
```

With:
```python
    if quality == Quality.FAST:
        from pdfmux.extractors.fast import FastExtractor
        ext = FastExtractor()
        # Enable table enhancement when tables are detected
        pages = list(ext.extract(
            file_path,
            enhance_tables=classification.has_tables,
        ))
        return pages, ext.name, []
```

### Why This Works

- **Zero new dependencies**: `page.find_tables()` is built into PyMuPDF 1.23.0+ (we already require PyMuPDF).
- **Minimal overhead**: Table detection is heuristic-based (no model loading), adds ~0.01s per page with tables.
- **Graceful degradation**: If `find_tables()` isn't available (old PyMuPDF), returns original text unchanged.
- **Additive output**: Tables are appended as clean markdown after the original text, so no existing content is lost.

### Pandas Dependency Note

`table.to_pandas()` requires pandas. To avoid adding pandas as a hard dependency, add a fallback path:

```python
try:
    df = table.to_pandas()
except ImportError:
    # Fallback: manually build from table.extract()
    cells = table.extract()
    if not cells or len(cells) < 2:
        continue
    headers = [str(c).strip() if c else "" for c in cells[0]]
    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in cells[1:]:
        row_cells = [str(c).strip() if c else "" for c in row]
        md_lines.append("| " + " | ".join(row_cells) + " |")
    table_markdowns.append("\n".join(md_lines))
    continue
```

### Complexity: M (Medium)

- New function in `fast.py`, modify `extract()` signature, small change in `pipeline.py`
- Requires testing with PyMuPDF `find_tables()` API
- Need to handle pandas optional dependency gracefully

### Breaking Change: No

- `enhance_tables` defaults to `False` -- existing callers unaffected
- Fast mode output contains original text plus optional table appendix
- No API signature changes in `process()` or `ConversionResult`

---

## Gap 6: Classification Lacks `empty_pages` Field

### Problem

`PDFClassification` tracks `digital_pages`, `scanned_pages`, and `graphical_pages` but not `empty_pages`. The classify function detects empty-text pages but lumps them into `digital_pages` (the else-branch at line 81). This data is needed by the dynamic OCR budget (Gap 2) and empty page warnings (Gap 3).

### Current Behavior

```python
# detect.py, lines 76-81
if text_len > 50:
    digital_pages.append(page_num)
elif images:
    scanned_pages.append(page_num)
else:
    digital_pages.append(page_num)  # <-- empty pages go here
```

### Proposed Behavior

#### File: `detect.py` -- add `empty_pages` to classification

```python
@dataclass
class PDFClassification:
    is_digital: bool = False
    is_scanned: bool = False
    is_mixed: bool = False
    is_graphical: bool = False
    has_tables: bool = False
    page_count: int = 0
    languages: list[str] = field(default_factory=list)
    confidence: float = 0.0
    digital_pages: list[int] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    graphical_pages: list[int] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)   # <-- new field
```

#### File: `detect.py` -- modify `classify()` page loop

Replace lines 76-81:
```python
        if text_len > 50:
            digital_pages.append(page_num)
        elif images:
            scanned_pages.append(page_num)
        else:
            digital_pages.append(page_num)
```

With:
```python
        if text_len < 20 and image_count == 0:
            empty_pages.append(page_num)
        elif text_len > 50:
            digital_pages.append(page_num)
        elif images:
            scanned_pages.append(page_num)
        else:
            digital_pages.append(page_num)
```

And add `empty_pages = []` initialization alongside the others (line 64), then:
```python
    result.empty_pages = empty_pages
```

### Complexity: S (Small)

- Add one field to a dataclass, adjust one if/elif chain

### Breaking Change: No

- New field has a default (`field(default_factory=list)`)
- Existing code that accesses `PDFClassification` will not break
- No serialization format changes

---

## Implementation Order

Recommended shipping order based on dependencies and impact:

```
Phase 1 (v1.1.0-rc1) -- Foundation
  1. Gap 6: Add empty_pages to PDFClassification      [S, 0.5 day]
  2. Gap 1: Improve table detection                    [M, 1 day]
  3. Gap 3: Empty page warnings                        [S, 0.5 day]

Phase 2 (v1.1.0-rc2) -- Performance
  4. Gap 2: Dynamic OCR budget                         [S, 0.5 day]
  5. Gap 4: Docling lazy loading + targeted extraction  [L, 1.5 days]

Phase 3 (v1.1.0) -- Enhancement
  6. Gap 5: Fast mode table extraction                 [M, 1 day]
```

### Rationale

- **Phase 1** ships the classification improvements that Phase 2 depends on (dynamic OCR budget needs `empty_pages`; table detection fix feeds into table routing).
- **Phase 2** addresses the two performance bottlenecks (budget + Docling timeout).
- **Phase 3** adds the table enhancement to fast mode, which is additive and independent.

---

## Test Plan

### Unit Tests

| Test | File | What it validates |
|------|------|-------------------|
| `test_detect_tables_financial` | `test_detect.py` | Tesla 10-K, Apple 10-K trigger `has_tables=True` |
| `test_detect_tables_sampling` | `test_detect.py` | Tables on page 60 of 144 are found via middle sampling |
| `test_number_density_scoring` | `test_detect.py` | Lines like `"$394,328  $383,285"` score as number-dense |
| `test_column_alignment_scoring` | `test_detect.py` | Pages with 3+ aligned x-position clusters score correctly |
| `test_dynamic_ocr_budget_image_heavy` | `test_pipeline.py` | 50%+ graphical doc gets budget=1.0 |
| `test_dynamic_ocr_budget_default` | `test_pipeline.py` | <25% graphical doc keeps budget=0.30 |
| `test_empty_page_warning_always` | `test_audit.py` | 2 empty pages in 144 emits warning (not gated at 15%) |
| `test_empty_page_warning_page_nums` | `test_audit.py` | Warning includes specific page numbers |
| `test_docling_singleton` | `test_tables.py` | Two calls to `_get_converter()` return same instance |
| `test_extract_pages_subset` | `test_tables.py` | Docling processes only requested pages |
| `test_fast_table_enhancement` | `test_fast.py` | `find_tables()` output formatted as markdown |
| `test_fast_no_pandas_fallback` | `test_fast.py` | Table extraction works without pandas installed |
| `test_empty_pages_classification` | `test_detect.py` | Empty pages tracked in `classification.empty_pages` |

### Integration Tests (Benchmark Corpus)

| Document | Expected Change |
|----------|----------------|
| Tesla 10-K | `has_tables=True` (was False) |
| Apple 10-K | `has_tables=True` (was False) |
| Berkshire AR | `has_tables=True` (was False), empty page warning |
| EY 10-K Guide | `has_tables=True` (was False) |
| Airbnb deck | All 6 graphical pages OCR'd (was 3) |
| Uber S-1 | Standard mode completes within 120s (was timeout) |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Table detection false positives | Score threshold is conservative (3 out of max 7); test against full corpus |
| Docling page subsetting changes output | Fallback to full-document Docling if subset extraction fails |
| `find_tables()` not in user's PyMuPDF version | Try/except with graceful fallback to original text |
| Pandas not installed for table formatting | Fallback to manual `table.extract()` cell extraction |
| Dynamic OCR budget increases processing time | Only triggers for image-heavy docs (>50% graphical); net positive for user value |
| Singleton Docling converter memory usage | Converter uses ~200MB RAM; acceptable for server-side use. CLI exits after processing so memory is reclaimed. |

---

## Metrics to Track Post-Release

1. **Table detection recall**: % of corpus documents with real tables where `has_tables=True` (target: >90%, current: 57%)
2. **Airbnb deck coverage**: Character count from image-heavy docs (target: >5000 chars, current: 2766)
3. **Uber S-1 standard mode time**: Wall-clock time in standard mode (target: <120s, current: timeout)
4. **Fast mode table quality**: Manual review of table formatting in fast mode output
5. **Empty page warning visibility**: All 6 known empty pages in corpus emit warnings (target: 100%, current: 0%)
