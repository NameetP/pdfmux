# Architecture Note: Column-Aware Reading Order

**Goal**: 0.900 → 0.910+ overall (beat paid #1 at 0.909)
**Primary lever**: NID 0.918 → 0.935 (+0.017)
**Root cause**: 40 docs have correct text but wrong reading order (multi-column interleaving)

---

## Current State

### How reading order works today

```
PDF → pymupdf4llm.to_markdown() → text with baked-in reading order
                                    ↓
                              no reordering anywhere
                                    ↓
                              pipeline merges pages → clean_text → output
```

pymupdf4llm is a **black box** for reading order. It uses PyMuPDF's internal layout analysis, which works well for single-column docs but fails on:
- 2-column academic papers (reads across columns instead of down)
- Financial reports with sidebars
- Multi-column resumes
- Newsletters and magazines

### What already exists (unused)

`detect.py:322-425` has a complete `detect_layout()` function that:
1. Extracts text blocks with bounding boxes via `page.get_text("blocks")`
2. Clusters x0 positions with 50pt gap detection
3. Builds column boundaries
4. Returns a `PageLayout` with column count + reading order

`types.py:129-138` has the `PageLayout` dataclass ready to go.

**This code is defined but NEVER CALLED in the pipeline.**

---

## Proposed Architecture

### The integration point

The fix goes in `extractors/fast.py` and `pipeline.py`. Two strategies depending on confidence:

#### Strategy A: Post-extraction reorder (safer, simpler)

After pymupdf4llm extracts text, detect if the page is multi-column. If so, re-extract using fitz blocks in the correct reading order instead of using pymupdf4llm's text.

```
PDF → pymupdf4llm.to_markdown() → text (possibly wrong order)
  ↓
  page → detect_layout(page)
           ↓
           columns == 1? → keep pymupdf4llm text as-is
           columns >= 2? → re-extract with fitz blocks in correct column order
                            ↓
                          merge block texts: col1 top→bottom, col2 top→bottom
                            ↓
                          use THIS text instead of pymupdf4llm text
```

**Where to wire it**: `fast.py:153-175`, after pymupdf4llm extraction, before heading injection.

```python
# fast.py — after line 153 (pymupdf4llm extraction)
for i, chunk in enumerate(chunks):
    text = chunk.get("text", "")

    # NEW: detect multi-column and reorder if needed
    if i < len(doc):
        layout = detect_layout(doc[i])
        if layout.columns >= 2:
            text = _reorder_by_layout(doc[i], layout)

    # existing: heading injection, quality checks, etc.
    if i < len(doc):
        text = inject_headings(text, doc[i])
    ...
```

#### Strategy B: Parallel extraction with quality comparison (more robust)

Extract text TWO ways, compare, pick the better one:

```
PDF page → pymupdf4llm text  ──→ NID-like self-score ─┐
     ↓                                                  ├→ pick better
     └──→ column-reordered text ──→ NID-like self-score ┘
```

Self-scoring heuristic: compare text against a "reference" built from all text blocks. The extraction with more sequential block-to-block matches wins.

**This is more complex but avoids regressions** — if column reordering makes things worse (e.g., false column detection), the original pymupdf4llm text is preserved.

### Recommendation: Start with Strategy A, add B if regressions appear

---

## The `_reorder_by_layout()` function

This is the new function that needs to be built. It takes a fitz page + PageLayout and returns correctly-ordered text.

### Algorithm

```python
def _reorder_by_layout(page: fitz.Page, layout: PageLayout) -> str:
    """Re-extract page text in column-aware reading order."""

    blocks = page.get_text("blocks")
    text_blocks = [(i, b) for i, b in enumerate(blocks) if b[6] == 0 and b[4].strip()]

    # Use the reading_order from detect_layout
    ordered_texts = []
    for block_idx in layout.reading_order:
        for i, b in text_blocks:
            if i == block_idx:
                ordered_texts.append(b[4].strip())
                break

    return "\n\n".join(ordered_texts)
```

### Key decisions

**1. What text format to use from fitz blocks?**

`page.get_text("blocks")` returns raw text without markdown formatting. pymupdf4llm adds:
- Bold markers (`**text**`)
- Heading markers (`# text`)
- Table formatting (`| cell | cell |`)
- List markers (`- item`)

Options:
- **Option A**: Use raw block text (lose formatting). Heading injection + bold promotion still runs after, so headings will be recovered. Tables are overlaid from Docling anyway.
- **Option B**: Use `page.get_text("dict")` for richer data (font info, spans). More work but preserves more structure.
- **Option C**: Use pymupdf4llm's text but reorder the paragraphs based on the layout analysis. This is tricky because pymupdf4llm's text doesn't have block boundaries.

**Recommendation: Option A** for simplicity. The heading injection pipeline already recovers structural formatting. The main thing we need is correct text ORDER, not formatting.

**2. How to handle spanning elements?**

A full-width heading in a 2-column doc should NOT be split across columns. Detection:
- A block whose x0→x1 spans >60% of page width is a "spanning element"
- Spanning elements are inserted at their y-position in the reading order, between column blocks

```python
# Pseudocode for spanning element handling
for block in text_blocks:
    x0, y0, x1, y1 = block[:4]
    block_width = x1 - x0
    if block_width > page_width * 0.6:
        # This is a spanning element (heading, table, figure caption)
        # Insert at correct y-position, not assigned to any column
        spanning_elements.append((y0, block))
    else:
        # Assign to nearest column
        assign_to_column(block)

# Merge: interleave spanning elements with column text by y-position
```

**3. When to skip reordering?**

Don't reorder when:
- Page has < 6 text blocks (too few to determine columns)
- Page has tables (table blocks span columns and confuse detection)
- Classification says page is graphical/scanned (OCR text has unreliable positions)
- Column detection confidence is low (ambiguous clustering)

**4. What about the `pymupdf_layout` package?**

pymupdf4llm prints a deprecation-style message: "Consider using the pymupdf_layout package for a greatly improved page layout analysis." This is PyMuPDF's official layout analysis library. Worth investigating as an alternative to our custom column detection.

```python
# Check if available
try:
    import pymupdf_layout
    # May provide better column detection than our custom clustering
except ImportError:
    pass
```

---

## Impact on the pipeline

### Files to modify

| File | Change | Risk |
|---|---|---|
| `extractors/fast.py` | Wire `detect_layout()` + `_reorder_by_layout()` after pymupdf4llm extraction | Medium — could regress single-column docs if column detection false-positives |
| `detect.py` | May need tuning of `_COLUMN_GAP_MIN` (currently 50pt). Add spanning element logic. Add confidence score | Low — isolated function |
| `audit.py` | Apply same reordering in the audit path (line 293) | Low — mirrors fast.py change |
| `pipeline.py` | Pass `PageLayout` through to merge step for potential cross-page column awareness | Low |

### Files NOT to modify

| File | Why |
|---|---|
| `headings.py` | Heading injection works on the final text — independent of reading order |
| `postprocess.py` | Text cleaning is order-independent |
| `regions.py` | Region OCR is supplemental — doesn't affect main extraction |

---

## Testing strategy

### Fast validation (6 docs)

Pick 3 known multi-column docs from the benchmark + 3 single-column docs. Run before/after and verify:
- Multi-column NID improves
- Single-column NID doesn't regress

### Full validation (200 docs)

Run the full benchmark. Accept if:
- Overall >= 0.905
- Regressions <= 5
- No single doc drops more than 0.05

### Manual spot-check

For the 40 wrong-order docs, visually compare output vs GT for 5 representative docs. Verify the column interleaving is fixed.

---

## Edge cases to handle

1. **3+ columns** (rare but exists in newspapers): The algorithm already handles N columns — just ensure gap detection works for 3 clusters.

2. **Mixed layout** (first half single-column, second half two-column): Detect per-page, not per-document. Each page gets its own layout analysis.

3. **Marginal notes / sidebars**: These are narrow columns at page edges. The gap detection should catch them, but the text may be ancillary (footnotes, references). Consider ignoring columns < 15% of page width.

4. **Right-to-left text** (Arabic, Hebrew): Column order reverses. Detect via Unicode character analysis of extracted text. Out of scope for v1 but note for future.

5. **Tables spanning columns**: If a table spans the full page width in a 2-column doc, it should be treated as a spanning element, not split.

6. **Headers/footers**: These span the full page width. The spanning element detection (>60% of page width) should catch them, but they should be at the top/bottom of the reading order.

---

## Estimated effort

| Task | Complexity | Time |
|---|---|---|
| Wire `detect_layout()` into fast.py | Simple | 30 min |
| Add spanning element detection | Medium | 1 hour |
| Add `_reorder_by_layout()` function | Medium | 1 hour |
| Tune `_COLUMN_GAP_MIN` threshold | Iteration | 1-2 hours |
| Wire into audit.py path | Simple | 30 min |
| Run benchmark + fix regressions | Iteration | 2-3 hours |
| **Total** | | **5-7 hours** |

---

## Success criteria

- Overall: 0.900 → **0.910+** (beats paid #1 at 0.909)
- NID: 0.918 → **0.930+**
- MHS: no regression (stay >= 0.840)
- TEDS: no regression (stay >= 0.887)
- Zero new dependencies
- < 2ms overhead per page

---

## Experiment Results (2026-03-21)

### What was tested

Three approaches were tested against the 200-doc benchmark:

#### Attempt 1: Raw fitz block replacement (50pt gap)
- Detected 148/200 docs as multi-column (way too many)
- Replaced pymupdf4llm text with raw fitz blocks in column order
- **Result: 0.878 overall (DOWN from 0.900), 45 regressions**
- Root cause: 50pt gap threshold detected single-column docs with indented text as multi-column. Raw fitz blocks lost all markdown formatting (bold, headings, lists), crashing MHS from 0.844 to 0.772.

#### Attempt 2: Raw fitz block replacement (200pt gap)
- Detected 46/200 docs as multi-column (more reasonable)
- Same raw block replacement approach
- **Result: 0.894 overall (DOWN from 0.900), 11 regressions**
- Root cause: Even at 200pt, some docs were being replaced with raw text unnecessarily. MHS still dropped to 0.822.

#### Attempt 3: Paragraph reordering (200pt gap)
- Keep pymupdf4llm text, match paragraphs to fitz blocks by text overlap, reorder paragraphs
- **Result: 0.895 overall (DOWN from 0.900), 18 regressions**
- Root cause: Paragraph-to-block matching was too imprecise (first-50-chars overlap). Docs where pymupdf4llm already had correct order were being scrambled. Major regressions on docs 069 (-0.132 NID), 097 (-0.135 NID), 153 (-0.559 MHS).

### Key findings

1. **pymupdf4llm handles many multi-column docs correctly already.** Blindly reordering everything multi-column causes more harm than good.

2. **Column detection false positives are deadly.** Even at 200pt gap threshold, docs with wide margins, centered text, or indented code blocks get falsely detected as multi-column.

3. **Text replacement destroys formatting.** Any approach that replaces pymupdf4llm text with raw fitz blocks loses heading markers, bold, lists, and table formatting — crashing MHS.

4. **Paragraph matching is fragile.** pymupdf4llm splits text differently than fitz blocks. Character-overlap matching at 50 chars is not reliable enough for correct reordering.

### What must change for v2

The approach needs to be fundamentally different:

1. **Only reorder when provably wrong.** Need a heuristic to detect when pymupdf4llm's order is actually incorrect (e.g., compare consecutive paragraphs — if a paragraph's y-position jumps UP, the order is likely wrong).

2. **Never replace text.** Reorder pymupdf4llm's existing paragraphs, don't substitute with raw fitz text. This preserves all formatting.

3. **Better paragraph-to-block matching.** Use `fuzz.partial_ratio` or longest-common-subsequence instead of first-50-chars overlap. Or use the y-position of each paragraph's first line (available from `page.get_text("dict")`) to establish correspondence.

4. **Conservative column detection.** The gap threshold alone isn't enough. Also require:
   - At least 3 blocks per column (not just 2 x-positions far apart)
   - Blocks in different columns must have comparable y-ranges (actual interleaving)
   - Page must have enough text to warrant reordering (>200 chars per column)

5. **A/B comparison.** Run both orderings, compute a self-consistency score (e.g., do consecutive sentences flow logically?), pick the better one. This makes the feature safe — worst case, it's a no-op.
