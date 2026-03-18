"""Heading detection via font-size analysis.

Analyzes PyMuPDF font metadata to identify heading spans and inject
markdown ``#`` markers into extracted text. Pure heuristic — no ML,
no dependencies beyond PyMuPDF.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import fitz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_headings(text: str, page: fitz.Page) -> str:
    """Insert ``#`` heading markers based on font size analysis.

    If the text already contains 2+ ATX headings (pymupdf4llm detected
    them), the text is returned unchanged to avoid double-marking.
    """
    if not text or not text.strip():
        return text

    # Early exit: pymupdf4llm already detected headings
    existing = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
    if existing >= 2:
        return text

    body_size, candidates = _build_font_census(page)
    if body_size <= 0 or not candidates:
        return _promote_bold_lines(text)

    heading_map = _assign_levels(candidates, body_size)
    if not heading_map:
        return _promote_bold_lines(text)

    text = _inject_markers(text, heading_map)
    text = _promote_bold_lines(text)
    return text


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _HeadingCandidate:
    text: str
    size: float
    is_bold: bool
    y_position: float


# ---------------------------------------------------------------------------
# Font census
# ---------------------------------------------------------------------------

def _build_font_census(
    page: fitz.Page,
) -> tuple[float, list[_HeadingCandidate]]:
    """Return ``(body_size, heading_candidates)`` from page font data."""
    try:
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    except Exception:
        return 0.0, []

    blocks = page_dict.get("blocks", [])

    # Weighted char count per font size
    size_counts: Counter[float] = Counter()
    candidates: list[_HeadingCandidate] = []

    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            # Aggregate line text and dominant size/flags
            line_text_parts: list[str] = []
            line_sizes: list[tuple[float, int]] = []  # (size, char_count)
            line_bold = False

            for span in spans:
                span_text = span.get("text", "").strip()
                span_size = round(span.get("size", 0), 1)
                span_flags = span.get("flags", 0)

                if not span_text:
                    continue

                char_count = len(span_text)
                size_counts[span_size] += char_count
                line_text_parts.append(span_text)
                line_sizes.append((span_size, char_count))

                if span_flags & (1 << 4):  # bold flag
                    line_bold = True

            line_text = " ".join(line_text_parts).strip()
            if not line_text or len(line_text) < 2:
                continue

            # Dominant font size for this line
            if line_sizes:
                dominant_size = max(line_sizes, key=lambda x: x[1])[0]
            else:
                continue

            # y-position from first span bbox
            y_pos = 0.0
            if spans and "bbox" in spans[0]:
                y_pos = spans[0]["bbox"][1]

            candidates.append(
                _HeadingCandidate(
                    text=line_text,
                    size=dominant_size,
                    is_bold=line_bold,
                    y_position=y_pos,
                )
            )

    # Body size = font size with most characters
    if not size_counts:
        return 0.0, []
    body_size = size_counts.most_common(1)[0][0]

    return body_size, candidates


# ---------------------------------------------------------------------------
# Level assignment
# ---------------------------------------------------------------------------

_SIZE_RATIO = 1.2   # 20% larger than body → heading
_BOLD_RATIO = 1.05  # 5% larger + bold → heading
_MAX_HEADING_CHARS = 120  # headings are short


def _assign_levels(
    candidates: list[_HeadingCandidate],
    body_size: float,
) -> dict[str, int]:
    """Map candidate texts to heading levels (1–3)."""
    heading_candidates: list[_HeadingCandidate] = []

    for c in candidates:
        text_len = len(c.text)
        if text_len > _MAX_HEADING_CHARS:
            continue  # too long to be a heading

        is_large = c.size >= body_size * _SIZE_RATIO
        is_bold_large = (
            c.size >= body_size * _BOLD_RATIO
            and c.is_bold
            and text_len < 80
        )
        # Bold at same size as body, short line — likely a heading
        is_bold_same_size = (
            c.is_bold
            and abs(c.size - body_size) < 0.5
            and text_len < 80
        )

        if is_large or is_bold_large or is_bold_same_size:
            heading_candidates.append(c)

    if not heading_candidates:
        return {}

    # Distinct sizes → heading levels
    distinct_sizes = sorted(
        {c.size for c in heading_candidates}, reverse=True
    )

    size_to_level: dict[float, int] = {}
    for idx, size in enumerate(distinct_sizes[:3]):  # cap at h3
        size_to_level[size] = idx + 1

    heading_map: dict[str, int] = {}
    for c in heading_candidates:
        level = size_to_level.get(c.size, 3)
        # Don't overwrite if we already assigned a higher level (lower number)
        if c.text not in heading_map or level < heading_map[c.text]:
            heading_map[c.text] = level

    return heading_map


# ---------------------------------------------------------------------------
# Markdown injection
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return _WS_RE.sub(" ", s).strip()


def _inject_markers(text: str, heading_map: dict[str, int]) -> str:
    """Find heading text in markdown and prepend ``#`` markers."""
    lines = text.split("\n")
    result: list[str] = []

    # Build normalized lookup
    norm_map: dict[str, tuple[str, int]] = {}
    for raw_text, level in heading_map.items():
        norm = _normalize(raw_text)
        if norm and len(norm) >= 2:
            norm_map[norm] = (raw_text, level)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue

        # Skip lines that already have heading markers
        if re.match(r"^#{1,6}\s", stripped):
            result.append(line)
            continue

        # Strip existing bold markers for matching
        clean = stripped.replace("**", "").replace("__", "")
        norm_line = _normalize(clean)

        matched = False
        for norm_key, (_, level) in norm_map.items():
            # Exact or near-exact match (heading text is the whole line
            # or the line starts with the heading text)
            if norm_line == norm_key or (
                norm_line.startswith(norm_key)
                and len(norm_key) / max(len(norm_line), 1) > 0.8
            ):
                prefix = "#" * level + " "
                # Remove bold markers from heading lines
                clean_line = stripped.replace("**", "").replace("__", "")
                result.append(prefix + clean_line)
                matched = True
                break

        if not matched:
            result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Bold-line promotion (fallback)
# ---------------------------------------------------------------------------

_BOLD_LINE_RE = re.compile(
    r"^\*\*(.{3,60})\*\*$"  # **short text** as entire line
)


def _promote_bold_lines(text: str) -> str:
    """Convert short bold-only lines at paragraph starts to ``###``."""
    lines = text.split("\n")
    result: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _BOLD_LINE_RE.match(stripped)

        if m:
            # Only promote if preceded by blank line or start of text
            prev_blank = i == 0 or not lines[i - 1].strip()
            if prev_blank:
                result.append("### " + m.group(1))
                continue

        result.append(line)

    return "\n".join(result)
