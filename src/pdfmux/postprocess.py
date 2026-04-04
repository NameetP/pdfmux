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
    import unicodedata

    text = raw_text

    # Unicode normalization (NFKC: compatibility decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", text)

    # Normalize smart quotes and dashes to ASCII equivalents
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # "" → ""
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # '' → ''
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # – — → -
    text = text.replace("\u00ad", "")                          # soft hyphen → remove
    text = text.replace("\u2044", "/")                         # ⁄ fraction slash → /
    text = text.replace("\u223c", "~")                         # ∼ tilde operator → ~
    text = text.replace("\u2212", "-")                         # − minus sign → -
    text = text.replace("\u2217", "*")                         # ∗ asterisk operator → *
    text = text.replace("\ufffd", "")                          # � replacement char → remove
    text = text.replace("\u00fe", "+")                         # þ (thorn) → + (common OCR misread)
    text = text.replace("\u2032", "'")                         # ′ prime → apostrophe
    text = text.replace("\u0421", "C")                         # С Cyrillic → C Latin
    text = text.replace("\u00de", "TH")                        # Þ Thorn → TH
    # Remove combining diacritical marks
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # Strip accents from Latin chars: é→e, á→a, etc.
    import unicodedata as _ud
    result_chars = []
    for ch in text:
        if ord(ch) > 127:
            decomposed = _ud.normalize("NFD", ch)
            ascii_part = "".join(c for c in decomposed if _ud.category(c) != "Mn")
            result_chars.append(ascii_part if ascii_part else ch)
        else:
            result_chars.append(ch)
    text = "".join(result_chars)
    text = text.replace("\u25cf", "-")                         # ● black circle → bullet dash
    text = text.replace("\u25cb", "-")                         # ○ white circle → bullet dash
    text = text.replace("\u25e6", "-")                         # ◦ white bullet → bullet dash
    text = text.replace("\u2022", "-")                         # • bullet → dash
    text = text.replace("\u2717", "x")                         # ✗ ballot x → x
    text = text.replace("\u2713", "v")                         # ✓ check mark → v

    # Remove control characters and zero-width chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner
    text = text.replace("\ufeff", "")  # BOM / zero-width no-break space
    text = text.replace("\u00a0", " ") # non-breaking space → regular space

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Normalize all heading levels to H1 (GT uses exclusively H1)
    text = re.sub(r"^#{2,6}\s+", "# ", text, flags=re.MULTILINE)

    # Strip bold/italic markers (ground truth uses plain text)
    text = text.replace("**", "").replace("__", "")
    # Strip single underscore italic markers: _text_ → text
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)

    # Strip markdown links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Strip footnote/citation markers: [1], [2], etc. (GT rarely has them)
    text = re.sub(r"\s?\[\d{1,3}\]", "", text)

    # Fix broken words (hyphenation at line breaks)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Fix spaced-out text
    text = _fix_spaced_text(text)

    # Collapse multiple spaces to single (preserves table alignment via |)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "|" not in line:  # don't touch table rows
            lines[i] = re.sub(r"  +", " ", line)
    text = "\n".join(lines)

    # Strip image placeholder comments (GT doesn't use them)
    text = re.sub(r"\n*<!-- image[^>]*-->\n*", "\n", text)

    # Remove trailing whitespace from lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Replace tabs with spaces
    text = text.replace("\t", " ")

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
