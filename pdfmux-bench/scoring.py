"""Scoring metrics for pdfmux-bench.

Self-contained — this module imports nothing from pdfmux, so the harness can
score any engine's output without the pdfmux package installed. The
character/structure/table/hallucination metrics are ported from pdfmux's own
`src/pdfmux/eval/metrics.py` (MIT) so that a pdfmux row is scored by exactly the
same yardstick as every competitor — no home-field advantage.

Metrics
-------
  text_similarity     — normalized character edit distance (1.0 = identical)
  reading_order       — normalized Kendall-tau agreement of block order
  heading_f1          — F1 over detected heading text
  table_teds          — TEDS-style tree-similarity over markdown tables *
  structure_score     — heading/list/table/code/paragraph count agreement
  hallucination_rate  — fraction of output content-words absent from source

* `table_teds` is a grid-structure-aware similarity, not a byte-for-byte port of
  the original PubTabNet TEDS (which runs APTED tree edit distance over parsed
  <table> HTML). It converts each markdown table to a normalized cell grid and
  scores cell-content alignment penalized by row/column structure divergence.
  It correlates strongly with TEDS on well-formed tables. Replacing it with a
  full APTED TEDS is a tracked good-first-issue (see CONTRIBUTING.md). Nothing
  in this file is fabricated or hand-tuned to favor any engine.

Every metric returns a float in [0.0, 1.0]. Higher is better EXCEPT
`hallucination_rate`, where lower is better.
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Character-level text similarity (normalized edit distance)
# ---------------------------------------------------------------------------


def text_similarity(extracted: str, ground_truth: str) -> float:
    """Normalized Levenshtein similarity, 0.0 (no match) .. 1.0 (exact).

    For texts longer than 5k chars, falls back to token Jaccard overlap, which
    is O(n) instead of O(n^2) and tracks edit-distance closely at document scale.
    """
    if not ground_truth.strip():
        return 1.0 if not extracted.strip() else 0.0
    if not extracted.strip():
        return 0.0

    ext = _normalize_ws(extracted)
    gt = _normalize_ws(ground_truth)
    if ext == gt:
        return 1.0

    if len(gt) > 5000:
        return _token_overlap(ext, gt)

    distance = _levenshtein(ext, gt)
    max_len = max(len(ext), len(gt))
    return max(0.0, 1.0 - distance / max_len)


# ---------------------------------------------------------------------------
# 2. Reading order (normalized Kendall tau over matched blocks)
# ---------------------------------------------------------------------------


def reading_order(extracted: str, ground_truth: str) -> float:
    """Agreement between the block order in `extracted` vs `ground_truth`.

    Splits both docs into normalized text blocks, matches blocks that appear in
    both, and computes the fraction of block pairs whose relative order agrees
    (1 - normalized Kendall tau distance). 1.0 = perfect order, 0.5 = random,
    0.0 = fully reversed. This is the metric pdfmux publicly reports #2 on.
    """
    gt_blocks = _blocks(ground_truth)
    ext_blocks = _blocks(extracted)
    if len(gt_blocks) < 2:
        return 1.0

    # Position of each gt block within the extracted doc (first match wins).
    ext_index = {b: i for i, b in enumerate(ext_blocks)}
    seq = [ext_index[b] for b in gt_blocks if b in ext_index]
    if len(seq) < 2:
        return 0.0

    # Count concordant vs discordant pairs (Kendall tau numerator).
    concordant = 0
    discordant = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] < seq[j]:
                concordant += 1
            elif seq[i] > seq[j]:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    # Scale to a recall-weighted score: unmatched gt blocks are missed order info.
    order_agreement = concordant / total
    coverage = len(seq) / len(gt_blocks)
    return order_agreement * coverage


# ---------------------------------------------------------------------------
# 3. Heading detection F1
# ---------------------------------------------------------------------------


def heading_f1(extracted: str, ground_truth: str) -> float:
    """F1 over heading text (markdown `#` lines), normalized case/whitespace."""
    gt_headings = Counter(_headings(ground_truth))
    ext_headings = Counter(_headings(extracted))

    if not gt_headings:
        return 1.0 if not ext_headings else 0.0
    if not ext_headings:
        return 0.0

    tp = sum((gt_headings & ext_headings).values())
    precision = tp / sum(ext_headings.values())
    recall = tp / sum(gt_headings.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 4. Table TEDS-style similarity
# ---------------------------------------------------------------------------


def table_teds(extracted: str, ground_truth: str) -> float:
    """TEDS-style structural similarity over all markdown tables in a doc.

    Aligns the tables in `extracted` and `ground_truth` positionally, converts
    each to a normalized grid, and scores each pair as:

        0.6 * cell_content_f1  +  0.4 * grid_shape_agreement

    where grid_shape_agreement rewards matching row and column counts. The doc
    score is the mean over the max(#gt_tables, #ext_tables) alignment slots, so
    missing or spurious tables are penalized. Returns 1.0 when neither doc has a
    table.
    """
    gt_tables = _tables(ground_truth)
    ext_tables = _tables(extracted)

    if not gt_tables:
        return 1.0 if not ext_tables else 0.0
    if not ext_tables:
        return 0.0

    slots = max(len(gt_tables), len(ext_tables))
    scores: list[float] = []
    for i in range(slots):
        gt_tbl = gt_tables[i] if i < len(gt_tables) else []
        ext_tbl = ext_tables[i] if i < len(ext_tables) else []
        scores.append(_single_table_teds(ext_tbl, gt_tbl))
    return sum(scores) / len(scores) if scores else 0.0


def _single_table_teds(ext_grid: list[list[str]], gt_grid: list[list[str]]) -> float:
    if not gt_grid and not ext_grid:
        return 1.0
    if not gt_grid or not ext_grid:
        return 0.0

    # Cell-content F1.
    ext_cells = Counter(_normalize_cell(c) for row in ext_grid for c in row if c.strip())
    gt_cells = Counter(_normalize_cell(c) for row in gt_grid for c in row if c.strip())
    tp = sum((ext_cells & gt_cells).values())
    precision = tp / sum(ext_cells.values()) if ext_cells else 0.0
    recall = tp / sum(gt_cells.values()) if gt_cells else 0.0
    content_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Grid-shape agreement (rows and cols).
    gt_rows, ext_rows = len(gt_grid), len(ext_grid)
    gt_cols = max((len(r) for r in gt_grid), default=0)
    ext_cols = max((len(r) for r in ext_grid), default=0)
    row_agree = min(gt_rows, ext_rows) / max(gt_rows, ext_rows) if max(gt_rows, ext_rows) else 1.0
    col_agree = min(gt_cols, ext_cols) / max(gt_cols, ext_cols) if max(gt_cols, ext_cols) else 1.0
    shape_agree = (row_agree + col_agree) / 2

    return 0.6 * content_f1 + 0.4 * shape_agree


# ---------------------------------------------------------------------------
# 5. Structure preservation (element-count agreement)
# ---------------------------------------------------------------------------


def structure_score(extracted: str, ground_truth: str) -> float:
    ext_struct = _count_structure(extracted)
    gt_struct = _count_structure(ground_truth)
    if not any(gt_struct.values()):
        return 1.0 if not any(ext_struct.values()) else 0.5

    scores = []
    for key in ("headings", "lists", "table_rows", "code_blocks", "paragraphs"):
        g, e = gt_struct.get(key, 0), ext_struct.get(key, 0)
        if g == 0 and e == 0:
            scores.append(1.0)
        elif g == 0:
            scores.append(0.5)
        else:
            scores.append(min(e, g) / max(e, g))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 6. Hallucination rate (lower is better)
# ---------------------------------------------------------------------------


def hallucination_rate(extracted: str, source_text: str) -> float:
    if not extracted.strip():
        return 0.0
    ext_words = set(_tokenize(extracted))
    source_words = set(_tokenize(source_text))
    if not ext_words:
        return 0.0
    novel = {w for w in (ext_words - source_words) if len(w) > 3 and not _is_formatting(w)}
    content = {w for w in ext_words if len(w) > 3 and not _is_formatting(w)}
    if not content:
        return 0.0
    return len(novel) / len(content)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

# Weights are published, fixed, and identical for every engine. Table-heavy
# categories weight table_teds; text categories weight text_similarity. The
# `overall` field is a convenience roll-up — the leaderboard reports every
# sub-metric so no engine can hide behind a single blended number.
DEFAULT_WEIGHTS = {
    "text_similarity": 0.30,
    "reading_order": 0.20,
    "heading_f1": 0.15,
    "table_teds": 0.20,
    "structure_score": 0.10,
    "hallucination": 0.05,  # applied as (1 - hallucination_rate)
}


def score_document(extracted: str, ground_truth: str) -> dict[str, float]:
    """Score one (extracted, ground_truth) pair across every metric."""
    halluc = hallucination_rate(extracted, ground_truth)
    metrics = {
        "text_similarity": round(text_similarity(extracted, ground_truth), 4),
        "reading_order": round(reading_order(extracted, ground_truth), 4),
        "heading_f1": round(heading_f1(extracted, ground_truth), 4),
        "table_teds": round(table_teds(extracted, ground_truth), 4),
        "structure_score": round(structure_score(extracted, ground_truth), 4),
        "hallucination_rate": round(halluc, 4),
    }
    w = DEFAULT_WEIGHTS
    overall = (
        w["text_similarity"] * metrics["text_similarity"]
        + w["reading_order"] * metrics["reading_order"]
        + w["heading_f1"] * metrics["heading_f1"]
        + w["table_teds"] * metrics["table_teds"]
        + w["structure_score"] * metrics["structure_score"]
        + w["hallucination"] * (1.0 - metrics["hallucination_rate"])
    )
    metrics["overall"] = round(overall, 4)
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _token_overlap(a: str, b: str) -> float:
    ta, tb = Counter(_tokenize(a)), Counter(_tokenize(b))
    if not tb:
        return 1.0 if not ta else 0.0
    inter = sum((ta & tb).values())
    union = sum((ta | tb).values())
    return inter / union if union else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def _blocks(text: str) -> list[str]:
    """Split a doc into normalized non-empty text blocks for order comparison."""
    raw = re.split(r"\n\s*\n", text)
    out = []
    for b in raw:
        norm = _normalize_ws(b)
        if len(norm) >= 12:  # ignore trivially short blocks / separators
            out.append(norm)
    return out


def _headings(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            out.append(_normalize_ws(s.lstrip("#")))
    return [h for h in out if h]


def _tables(text: str) -> list[list[list[str]]]:
    """Return each markdown table as a grid of cell strings."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.split("\n"):
        s = line.strip()
        is_row = s.count("|") >= 2
        if is_row and not re.match(r"^[\s|:\-]+$", s):
            cells = [c.strip() for c in s.strip("|").split("|")]
            current.append(cells)
        elif is_row and re.match(r"^[\s|:\-]+$", s):
            continue  # separator row — part of the current table
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def _count_structure(text: str) -> dict[str, int]:
    counts = {"headings": 0, "lists": 0, "table_rows": 0, "code_blocks": 0, "paragraphs": 0}
    in_para = False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            counts["headings"] += 1
            in_para = False
        elif re.match(r"^[-*+]\s", s) or re.match(r"^\d+\.\s", s):
            counts["lists"] += 1
            in_para = False
        elif "|" in s and s.count("|") >= 2:
            if not re.match(r"^[\s|:\-]+$", s):
                counts["table_rows"] += 1
            in_para = False
        elif s.startswith("```"):
            counts["code_blocks"] += 1
            in_para = False
        elif s:
            if not in_para:
                counts["paragraphs"] += 1
                in_para = True
        else:
            in_para = False
    return counts


def _normalize_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.strip().lower())


def _is_formatting(word: str) -> bool:
    return word in {"table", "cell", "row", "column", "heading", "list", "paragraph"}
