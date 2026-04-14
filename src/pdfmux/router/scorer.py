"""Dynamic quality scoring for LLM extraction output.

Replaces the hardcoded confidence=0.90 with a multi-signal scorer
that evaluates actual extraction quality.

Four signals:
  1. Text coherence — word structure, encoding quality
  2. Structure preservation — markdown elements present
  3. Completeness — text density relative to page area
  4. Consistency — overlap with fast-path (PyMuPDF) output
"""

from __future__ import annotations

import re
from collections import Counter


def score_llm_output(
    llm_text: str,
    fast_text: str | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
) -> float:
    """Score LLM extraction quality dynamically.

    Args:
        llm_text: Text extracted by the LLM.
        fast_text: Text extracted by PyMuPDF fast path (for consistency check).
                   None if not available (e.g., scanned page with no fast text).
        page_width: Page width in points (72 DPI).
        page_height: Page height in points.

    Returns:
        Confidence score 0.0 to 1.0.
    """
    if not llm_text or len(llm_text.strip()) < 5:
        return 0.0

    text = llm_text.strip()

    # Compute individual signals
    coherence = _text_coherence(text)
    structure = _structure_signal(text)
    completeness = _completeness_signal(text, page_width, page_height)

    if fast_text and len(fast_text.strip()) > 20:
        consistency = _consistency_signal(text, fast_text)
        # With fast-path reference: all 4 signals
        score = 0.30 * coherence + 0.20 * structure + 0.20 * completeness + 0.30 * consistency
    else:
        # No fast-path reference (pure scan): intrinsic signals only
        score = 0.40 * coherence + 0.30 * structure + 0.30 * completeness

    return round(max(0.0, min(1.0, score)), 4)


def _text_coherence(text: str) -> float:
    """Evaluate text plausibility — word structure and encoding quality.

    Returns 0.0 (garbled) to 1.0 (clean natural text).
    """
    score = 1.0

    # 1. Alphabetic ratio — clean text is mostly letters
    alpha_count = sum(1 for c in text if c.isalpha())
    total_chars = len(text)
    if total_chars > 0:
        alpha_ratio = alpha_count / total_chars
        if alpha_ratio < 0.30:
            score -= 0.30
        elif alpha_ratio < 0.50:
            score -= 0.15

    # 2. Average word length — natural language is 3-10 chars
    words = text.split()
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < 2 or avg_len > 20:
            score -= 0.20
        elif avg_len < 3 or avg_len > 15:
            score -= 0.10

    # 3. Encoding quality — detect mojibake / garbled text
    mojibake_patterns = [
        r"[\ufffd\ufffe\ufeff]",  # replacement chars
        # unusual unicode runs
        r"[^\x00-\x7f\u00c0-\u024f\u0400-\u04ff\u4e00-\u9fff\u3000-\u303f]{3,}",
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",  # control chars
    ]
    mojibake_count = sum(len(re.findall(p, text)) for p in mojibake_patterns)
    if mojibake_count > 10:
        score -= 0.25
    elif mojibake_count > 3:
        score -= 0.10

    # 4. Repetition detection — LLM hallucination often repeats
    if len(words) > 20:
        word_freq = Counter(w.lower() for w in words if len(w) > 3)
        if word_freq:
            max_freq = word_freq.most_common(1)[0][1]
            repeat_ratio = max_freq / len(words)
            if repeat_ratio > 0.20:
                score -= 0.20
            elif repeat_ratio > 0.10:
                score -= 0.05

    return max(0.0, score)


def _structure_signal(text: str) -> float:
    """Detect markdown structure — higher score if structured.

    Returns 0.0 (no structure) to 1.0 (well-structured).
    """
    lines = text.split("\n")
    if not lines:
        return 0.3  # neutral for empty

    signals = 0
    total_checks = 5

    # Headings
    heading_count = sum(1 for ln in lines if ln.strip().startswith("#"))
    if heading_count > 0:
        signals += 1

    # Lists
    list_count = sum(
        1
        for ln in lines
        if re.match(r"^\s*[-*+]\s", ln.strip()) or re.match(r"^\s*\d+\.\s", ln.strip())
    )
    if list_count > 0:
        signals += 1

    # Tables
    table_count = sum(1 for ln in lines if "|" in ln and ln.count("|") >= 2)
    if table_count > 0:
        signals += 1

    # Paragraphs (multi-line blocks)
    paragraph_breaks = sum(1 for ln in lines if ln.strip() == "")
    if paragraph_breaks >= 2:
        signals += 1

    # Bold/italic (actual formatting)
    formatting_count = len(re.findall(r"\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_", text))
    if formatting_count > 0:
        signals += 1

    # Base score: even without structure, text may be valid
    base = 0.40
    structure_bonus = 0.60 * (signals / total_checks)
    return base + structure_bonus


def _completeness_signal(text: str, page_width: float = 612.0, page_height: float = 792.0) -> float:
    """Estimate if the extraction is complete based on text density.

    Returns 0.0 (suspiciously empty) to 1.0 (reasonable amount of text).
    """
    # Estimate expected characters for a full page at ~250 words/page
    # Average: 250 words * 5 chars = 1250 chars for a standard letter page
    page_area = page_width * page_height
    standard_area = 612.0 * 792.0  # US Letter
    area_ratio = page_area / standard_area if standard_area > 0 else 1.0

    expected_chars = 1250 * area_ratio
    actual_chars = len(text.strip())

    if actual_chars == 0:
        return 0.0

    # Ratio of actual to expected
    fill_ratio = actual_chars / expected_chars

    if fill_ratio >= 0.80:
        return 1.0
    elif fill_ratio >= 0.50:
        return 0.85
    elif fill_ratio >= 0.20:
        return 0.65
    elif fill_ratio >= 0.05:
        return 0.40
    else:
        return 0.20


def _consistency_signal(llm_text: str, fast_text: str) -> float:
    """Measure overlap between LLM output and fast-path extraction.

    High overlap = LLM is consistent with rule-based extraction.
    Low overlap could mean LLM hallucinated OR LLM recovered text
    that rule-based missed (scanned pages).

    Returns 0.0 (no overlap) to 1.0 (high overlap).
    """
    llm_words = set(_tokenize(llm_text))
    fast_words = set(_tokenize(fast_text))

    if not fast_words or not llm_words:
        return 0.5  # neutral when comparison isn't possible

    # Jaccard similarity
    intersection = llm_words & fast_words
    union = llm_words | fast_words

    jaccard = len(intersection) / len(union) if union else 0.0

    # Adjust: very low Jaccard on a digital page = suspicious
    # But low Jaccard on a scanned page = expected (fast path had no text)
    return jaccard


def _tokenize(text: str) -> list[str]:
    """Split into lowercase words, filter short ones."""
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 2]
