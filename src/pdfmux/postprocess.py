"""Post-processing — clean extracted text.

Text cleanup only. Confidence scoring has moved to audit.py
(per-page scoring with 5 concrete checks + content-weighted averaging).

The clean_text() function handles:
    - Control character removal
    - Whitespace normalization
    - Broken hyphenation repair
    - Spaced-out text detection and repair
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Legacy ProcessedResult — kept for backward compat
# ---------------------------------------------------------------------------


@dataclass
class ProcessedResult:
    """Result of post-processing extracted text (legacy)."""

    text: str
    confidence: float
    page_count: int
    warnings: list[str]


def clean_and_score(
    raw_text: str,
    page_count: int,
    *,
    extraction_limited: bool = False,
    graphical_page_count: int = 0,
    ocr_page_count: int = 0,
) -> ProcessedResult:
    """Legacy entry point — clean text and compute confidence.

    New code should use clean_text() + audit.compute_document_confidence().
    """
    from pdfmux.audit import compute_document_confidence
    from pdfmux.types import PageQuality, PageResult

    text = clean_text(raw_text)

    # Build synthetic pages for legacy confidence scoring
    pages = [
        PageResult(
            page_num=0,
            text=text,
            confidence=1.0,
            quality=PageQuality.GOOD,
            extractor="legacy",
        )
    ]

    unrecovered = graphical_page_count if extraction_limited else 0
    confidence, warnings = compute_document_confidence(
        pages,
        ocr_page_count=ocr_page_count,
        unrecovered_count=unrecovered,
    )

    return ProcessedResult(
        text=text,
        confidence=confidence,
        page_count=page_count,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Clean text — the primary export
# ---------------------------------------------------------------------------


def clean_text(raw_text: str) -> str:
    """Clean extracted text — remove artifacts, normalize whitespace.

    Steps:
        1. Remove control characters (except newlines/tabs)
        2. Collapse 3+ consecutive blank lines into 2
        3. Fix broken words (hyphenation at line breaks)
        4. Fix spaced-out text artifacts
        5. Remove trailing whitespace from lines
        6. Strip leading/trailing whitespace from document

    Args:
        raw_text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    text = raw_text

    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Fix broken words (hyphenation at line breaks)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Fix spaced-out text
    text = _fix_spaced_text(text)

    # Remove trailing whitespace from lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Strip document
    text = text.strip()

    return text


def _fix_spaced_text(text: str) -> str:
    """Fix spaced-out text — a common PDF extraction artifact.

    Some PDFs render text with individual character placement:
    "W i t h  o v e r  1 7  y e a r s" → "With over 17 years"

    Detection: a line where >50% of "words" are single characters.
    """
    lines = text.split("\n")
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue

        words = stripped.split()
        if len(words) < 5:
            fixed_lines.append(line)
            continue

        single_char_count = sum(1 for w in words if len(w) == 1)
        single_char_ratio = single_char_count / len(words)

        if single_char_ratio > 0.5:
            groups = re.split(r"  +", stripped)
            fixed_groups = []
            for group in groups:
                parts = group.split(" ")
                if all(len(p) <= 1 for p in parts) and len(parts) >= 2:
                    fixed_groups.append("".join(parts))
                else:
                    fixed_groups.append(group)
            fixed_line = " ".join(fixed_groups)
            leading = len(line) - len(line.lstrip())
            fixed_lines.append(" " * leading + fixed_line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)
