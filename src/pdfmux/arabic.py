"""Arabic text handling — detection, BiDi reordering, normalization.

PyMuPDF extracts Arabic glyphs in storage order (LTR), which reverses the
visual reading order. This module applies the Unicode Bidirectional Algorithm
to restore correct RTL order, plus offers normalization helpers commonly
used in Arabic NLP pipelines (Tatweel removal, Alef/Yeh unification, Tashkeel
stripping).

Designed for the GCC logistics pipeline: bills of lading, customs forms,
SABER certificates, and other documents mixing Arabic with Latin script
(invoice numbers, dates, English part names).

Public API
----------
- ``is_arabic_text(text)`` — any Arabic codepoints present?
- ``is_rtl_dominant(text)`` — >50% of letter chars are RTL?
- ``fix_bidi_order(text)`` — apply Unicode BiDi algorithm line-by-line
- ``normalize_arabic(text)`` — canonicalize Arabic text for indexing/search

The BiDi implementation defers to the optional ``python-bidi`` package and
falls back gracefully (returning the input unchanged) when unavailable.
Install with: ``pip install pdfmux[arabic]``
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode ranges — Arabic and related blocks
# ---------------------------------------------------------------------------

# Core Arabic block (the spec calls out U+0600 to U+06FF specifically).
# We also recognize supplements and presentation forms so detection works
# on PDFs that store glyphs as presentation-form codepoints.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Hebrew is also RTL — useful for ``is_rtl_dominant`` even though the
# top-level API targets Arabic specifically.
_HEBREW_RANGE: tuple[int, int] = (0x0590, 0x05FF)

# Tatweel (kashida) — visual elongation, semantically meaningless.
_TATWEEL = "ـ"

# Alef variants → bare Alef (U+0627).
_ALEF_VARIANTS: dict[str, str] = {
    "آ": "ا",  # Alef with Madda above
    "أ": "ا",  # Alef with Hamza above
    "إ": "ا",  # Alef with Hamza below
    "ٱ": "ا",  # Alef Wasla
    "ٲ": "ا",  # Alef with Wavy Hamza above
    "ٳ": "ا",  # Alef with Wavy Hamza below
}

# Yeh variants → standard Yeh (U+064A). Alef Maksura (U+0649) is also
# normalized to Yeh because OCR engines frequently confuse the two.
_YEH_VARIANTS: dict[str, str] = {
    "ى": "ي",  # Alef Maksura
    "ی": "ي",  # Farsi Yeh
    "ے": "ي",  # Yeh Barree
}

# Teh Marbuta (U+0629) is intentionally NOT normalized to Heh — that
# changes meaning. We keep it as-is.

# Tashkeel (Arabic diacritics) — removed by ``normalize_arabic``.
_TASHKEEL_RANGE: tuple[int, int] = (0x064B, 0x065F)
# Plus the standalone Sukun and Shadda which fall in that range, plus
# Quranic annotation signs U+0670 (superscript Alef) and U+06D6–U+06ED.
_EXTRA_TASHKEEL: tuple[int, ...] = (
    0x0670,  # Arabic Letter Superscript Alef
    *range(0x06D6, 0x06ED + 1),  # Quranic annotation signs
    *range(0x08D3, 0x08E1 + 1),  # Extended Arabic marks (excluding 0x08E2)
    *range(0x08E3, 0x08FF + 1),  # Continued extended marks
)

# Pre-built translation table for fast normalization.
_NORMALIZATION_TABLE: dict[int, str | None] = {}
for src, dst in {**_ALEF_VARIANTS, **_YEH_VARIANTS}.items():
    _NORMALIZATION_TABLE[ord(src)] = dst
_NORMALIZATION_TABLE[ord(_TATWEEL)] = None
for cp in range(_TASHKEEL_RANGE[0], _TASHKEEL_RANGE[1] + 1):
    _NORMALIZATION_TABLE[cp] = None
for cp in _EXTRA_TASHKEEL:
    _NORMALIZATION_TABLE[cp] = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _is_arabic_codepoint(cp: int) -> bool:
    """True if codepoint sits in any Arabic Unicode block."""
    for lo, hi in _ARABIC_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _is_rtl_codepoint(cp: int) -> bool:
    """True for Arabic or Hebrew codepoints (any RTL script)."""
    if _is_arabic_codepoint(cp):
        return True
    return _HEBREW_RANGE[0] <= cp <= _HEBREW_RANGE[1]


def is_arabic_text(text: str) -> bool:
    """Return True if ``text`` contains any Arabic character.

    Checks the core Arabic block plus supplements and presentation forms.
    Empty strings return False.
    """
    if not text:
        return False
    for ch in text:
        if _is_arabic_codepoint(ord(ch)):
            return True
    return False


def is_rtl_dominant(text: str) -> bool:
    """Return True if more than 50% of letter characters are RTL.

    Whitespace, digits, and punctuation are ignored — only "letter"
    Unicode categories (Lo, Lu, Ll, Lt, Lm) participate in the ratio.
    Empty or letter-free strings return False.
    """
    if not text:
        return False
    rtl = 0
    letters = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if not cat.startswith("L"):
            continue
        letters += 1
        if _is_rtl_codepoint(ord(ch)):
            rtl += 1
    if letters == 0:
        return False
    return (rtl / letters) > 0.5


def arabic_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Arabic.

    Used by ``detect.classify`` to flag Arabic-heavy documents. Returns
    0.0 for empty or whitespace-only input.
    """
    if not text:
        return 0.0
    arabic = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if _is_arabic_codepoint(ord(ch)):
            arabic += 1
    if total == 0:
        return 0.0
    return arabic / total


# ---------------------------------------------------------------------------
# BiDi reordering
# ---------------------------------------------------------------------------


def fix_bidi_order(text: str) -> str:
    """Apply the Unicode Bidirectional Algorithm to fix mixed RTL/LTR text.

    PyMuPDF returns Arabic glyphs in storage order, which reverses
    the visual reading order. This function processes line-by-line —
    only lines containing RTL characters are reordered, so English-only
    content is untouched.

    Markdown-aware: preserves heading prefixes (``#``) and pipe-table
    structure; BiDi is applied within each cell.

    Falls back to returning ``text`` unchanged if ``python-bidi`` is
    not installed. Install with ``pip install pdfmux[arabic]``.
    """
    if not text:
        return text

    try:
        from bidi.algorithm import get_display  # type: ignore[import-not-found]
    except ImportError:
        return text

    fixed_lines: list[str] = []
    for line in text.split("\n"):
        if not _line_has_rtl(line):
            fixed_lines.append(line)
            continue

        stripped = line.strip()

        # Markdown table row — BiDi each cell separately so the pipe
        # structure (and any cell that's pure English) survives intact.
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = stripped.split("|")
            fixed_cells: list[str] = []
            for cell in cells:
                if _line_has_rtl(cell):
                    fixed_cells.append(get_display(cell.strip()))
                else:
                    fixed_cells.append(cell)
            fixed_lines.append("|".join(fixed_cells))
            continue

        # Markdown heading — keep the ``#`` prefix on the left.
        if stripped.startswith("#"):
            match = re.match(r"^(\s*#+\s*)", line)
            if match:
                prefix = match.group(1)
                rest = line[len(prefix) :]
                fixed_lines.append(prefix + get_display(rest))
                continue

        fixed_lines.append(get_display(line))

    return "\n".join(fixed_lines)


def _line_has_rtl(line: str) -> bool:
    """Fast scan: any RTL codepoint on this line?"""
    for ch in line:
        if _is_rtl_codepoint(ord(ch)):
            return True
    return False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_arabic(text: str) -> str:
    """Canonicalize Arabic text for search and indexing.

    Applies five normalizations in order:
      1. Strip Tatweel (U+0640) — visual elongation only
      2. Unify Alef variants → bare Alef (U+0627)
      3. Unify Yeh variants → standard Yeh (U+064A)
         (includes Alef Maksura, OCR-confusion-prone)
      4. Strip Tashkeel (U+064B–U+065F + supplementary marks)
      5. Collapse runs of whitespace

    Non-Arabic characters pass through untouched, so this is safe to
    apply to mixed Arabic/English text.
    """
    if not text:
        return text

    # Steps 1-4: single translate() pass over the prebuilt table.
    text = text.translate(_NORMALIZATION_TABLE)

    # Step 5: collapse whitespace (Tashkeel removal can leave runs).
    text = re.sub(r"[ \t ]+", " ", text)

    return text
