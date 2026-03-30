"""Extraction quality metrics for benchmarking.

Four metrics that measure different aspects of extraction quality:
  1. text_accuracy — character-level edit distance ratio
  2. structure_preservation — heading/list/table structure match
  3. table_f1 — cell-level precision/recall for tables
  4. hallucination_rate — words in output that don't appear in source
"""

from __future__ import annotations

import re
from collections import Counter


def text_accuracy(extracted: str, ground_truth: str) -> float:
    """Character-level accuracy using normalized Levenshtein distance.

    Returns 0.0 (no match) to 1.0 (exact match).
    Uses a fast approximation for long texts.
    """
    if not ground_truth.strip():
        return 1.0 if not extracted.strip() else 0.0
    if not extracted.strip():
        return 0.0

    # Normalize whitespace for fair comparison
    ext = _normalize(extracted)
    gt = _normalize(ground_truth)

    if ext == gt:
        return 1.0

    # For long texts, use token overlap (much faster than full edit distance)
    if len(gt) > 5000:
        return _token_overlap(ext, gt)

    # Full edit distance for shorter texts
    distance = _levenshtein(ext, gt)
    max_len = max(len(ext), len(gt))
    return max(0.0, 1.0 - distance / max_len)


def structure_preservation(extracted: str, ground_truth: str) -> float:
    """Measure how well document structure is preserved.

    Compares counts of structural elements:
    - Headings (# lines)
    - List items (- or * or numbered)
    - Table rows (| delimited)
    - Code blocks (``` fenced)
    - Paragraphs (text blocks separated by blank lines)

    Returns 0.0 to 1.0.
    """
    ext_struct = _count_structure(extracted)
    gt_struct = _count_structure(ground_truth)

    if not gt_struct:
        return 1.0 if not ext_struct else 0.5

    # Compare each structural element type
    scores = []
    for element_type in ("headings", "lists", "table_rows", "code_blocks", "paragraphs"):
        gt_count = gt_struct.get(element_type, 0)
        ext_count = ext_struct.get(element_type, 0)

        if gt_count == 0 and ext_count == 0:
            scores.append(1.0)
        elif gt_count == 0:
            scores.append(0.5)  # extra structure isn't terrible
        else:
            ratio = min(ext_count, gt_count) / max(ext_count, gt_count)
            scores.append(ratio)

    return sum(scores) / len(scores) if scores else 0.0


def table_f1(extracted: str, ground_truth: str) -> float:
    """Cell-level F1 score for table extraction.

    Extracts all table cells from both texts (markdown table format),
    then computes precision, recall, and F1 on the cell contents.

    Returns 0.0 to 1.0.
    """
    ext_cells = _extract_table_cells(extracted)
    gt_cells = _extract_table_cells(ground_truth)

    if not gt_cells:
        return 1.0 if not ext_cells else 0.0
    if not ext_cells:
        return 0.0

    # Normalize cells for comparison
    ext_normalized = Counter(_normalize_cell(c) for c in ext_cells if _normalize_cell(c))
    gt_normalized = Counter(_normalize_cell(c) for c in gt_cells if _normalize_cell(c))

    # True positives: cells in both
    true_positives = sum((ext_normalized & gt_normalized).values())

    precision = true_positives / sum(ext_normalized.values()) if ext_normalized else 0.0
    recall = true_positives / sum(gt_normalized.values()) if gt_normalized else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def hallucination_rate(extracted: str, source_text: str) -> float:
    """Rate of hallucinated content (words not in source).

    Returns 0.0 (no hallucinations) to 1.0 (all hallucinated).
    Lower is better.
    """
    if not extracted.strip():
        return 0.0

    ext_words = set(_tokenize(extracted))
    source_words = set(_tokenize(source_text))

    if not ext_words:
        return 0.0

    # Words in extraction that don't appear in source
    novel_words = ext_words - source_words

    # Filter out common markdown formatting tokens and short words
    novel_words = {w for w in novel_words if len(w) > 3 and not _is_formatting(w)}
    ext_content_words = {w for w in ext_words if len(w) > 3 and not _is_formatting(w)}

    if not ext_content_words:
        return 0.0

    return len(novel_words) / len(ext_content_words)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalize whitespace and case for comparison."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def _tokenize(text: str) -> list[str]:
    """Split text into words."""
    return re.findall(r"\b\w+\b", text.lower())


def _token_overlap(text_a: str, text_b: str) -> float:
    """Fast token-level overlap metric for long texts."""
    tokens_a = Counter(_tokenize(text_a))
    tokens_b = Counter(_tokenize(text_b))

    if not tokens_b:
        return 1.0 if not tokens_a else 0.0

    intersection = sum((tokens_a & tokens_b).values())
    union = sum((tokens_a | tokens_b).values())

    return intersection / union if union else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance. O(min(m,n)) space."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _count_structure(text: str) -> dict[str, int]:
    """Count structural markdown elements."""
    lines = text.split("\n")
    counts = {
        "headings": 0,
        "lists": 0,
        "table_rows": 0,
        "code_blocks": 0,
        "paragraphs": 0,
    }

    in_paragraph = False
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#"):
            counts["headings"] += 1
            in_paragraph = False
        elif re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            counts["lists"] += 1
            in_paragraph = False
        elif "|" in stripped and stripped.count("|") >= 2:
            # Skip separator rows (---|----|---)
            if not re.match(r"^[\s|:-]+$", stripped):
                counts["table_rows"] += 1
            in_paragraph = False
        elif stripped.startswith("```"):
            counts["code_blocks"] += 1
            in_paragraph = False
        elif stripped:
            if not in_paragraph:
                counts["paragraphs"] += 1
                in_paragraph = True
        else:
            in_paragraph = False

    return counts


def _extract_table_cells(text: str) -> list[str]:
    """Extract all cell contents from markdown tables."""
    cells = []
    for line in text.split("\n"):
        stripped = line.strip()
        if "|" not in stripped or stripped.count("|") < 2:
            continue
        # Skip separator rows
        if re.match(r"^[\s|:-]+$", stripped):
            continue
        # Extract cells between pipes
        parts = stripped.split("|")
        for part in parts:
            cell = part.strip()
            if cell:
                cells.append(cell)
    return cells


def _normalize_cell(cell: str) -> str:
    """Normalize a table cell for comparison."""
    return re.sub(r"\s+", " ", cell.strip().lower())


def _is_formatting(word: str) -> bool:
    """Check if a word is markdown formatting."""
    formatting = {"table", "cell", "row", "column", "heading", "list", "paragraph"}
    return word in formatting
