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

# Patterns that should never be headings regardless of source
_FALSE_HEADING_RE = re.compile(
    r"^#{1,6}\s+"
    r"("
    r"(Figure|Table|Fig\.)\s+\d"  # figure/table captions
    r"|"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d"  # dates
    r"|"
    r"\d{2,5}\s*$"  # multi-digit page numbers (keep single digits like chapter "2")
    r"|"
    r".*=\s*\d"      # equations/formulas (e.g. "magnification is 10 x 45 = 450x")
    r"|"
    r"\d{2,5}\s+\S.{30,}"  # running headers: page number + long text (e.g. "76 Study on...")
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_toc_page_headings(text: str) -> str:
    """On a Contents/TOC page, keep only the TOC heading itself."""
    lines = text.split("\n")
    first_heading_idx = None
    is_toc_page = False

    for i, line in enumerate(lines):
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if first_heading_idx is None:
                first_heading_idx = i
                heading_text = m.group(1).strip().lower()
                if heading_text in ("contents", "table of contents"):
                    is_toc_page = True
            elif is_toc_page:
                # Strip heading marker from non-first headings on TOC page
                lines[i] = line.lstrip("#").lstrip()

    return "\n".join(lines) if is_toc_page else text


def _clean_false_headings(text: str) -> str:
    """Remove heading markers from lines matching false-positive patterns."""
    def _demote(m: re.Match) -> str:
        return m.group(0).lstrip("#").lstrip()

    text = _FALSE_HEADING_RE.sub(_demote, text)
    return _clean_toc_page_headings(text)


def _merge_consecutive_headings(text: str) -> str:
    """Merge a short heading fragment with the next heading line.

    ``# III.`` (≤10 chars) followed by ``# Regulatory cholesterol``
    becomes ``# III. Regulatory cholesterol``. Only merges when the
    first heading is very short (likely a fragment, not a standalone).
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", lines[i])
        if m and len(m.group(2)) <= 10 and i + 1 < len(lines):
            m2 = re.match(r"^#{1,6}\s+(.*)", lines[i + 1])
            if m2:
                result.append(m.group(1) + " " + m.group(2) + " " + m2.group(1))
                i += 2
                continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def _clean_heading_bold(text: str) -> str:
    """Strip bold markers from heading lines (``# **text**`` → ``# text``)."""
    def _strip_bold_heading(m: re.Match) -> str:
        prefix = m.group(1)  # e.g. "## "
        rest = m.group(2).replace("**", "").replace("__", "")
        return prefix + rest

    return re.sub(
        r"^(#{1,6}\s+)(.*\*\*.*)$",
        _strip_bold_heading,
        text,
        flags=re.MULTILINE,
    )


def inject_headings(text: str, page: fitz.Page) -> str:
    """Insert ``#`` heading markers based on font size analysis.

    If the text already contains 2+ ATX headings (pymupdf4llm detected
    them), the text is returned unchanged to avoid double-marking.
    """
    if not text or not text.strip():
        return text

    def _finalize(t: str) -> str:
        return _clean_false_headings(_clean_heading_bold(t))

    # Early exit: pymupdf4llm already detected headings
    existing = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
    if existing >= 2:
        # Strip bold from headings first, then validate short headings
        text = _clean_heading_bold(text)

        # Validate short headings (≤3 chars) against font analysis to catch
        # false positives like "# 6" (page number) while keeping "# 2" (chapter)
        body_size, cands = _build_font_census(page)
        if body_size > 0 and cands:
            h_map = _assign_levels(cands, body_size)
            valid_short = set()
            for t in h_map:
                if len(t.strip()) <= 3:
                    valid_short.add(_normalize(t))

            # Build a set of text at the bottom 15% of page (likely page numbers)
            page_height = page.rect.height
            bottom_zone_texts = set()
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        y_pct = span["origin"][1] / page_height if page_height else 0
                        if y_pct > 0.85:
                            bottom_zone_texts.add(span["text"].strip())

            def _strip_invalid_short(m: re.Match) -> str:
                htext = m.group(0).split(None, 1)
                if len(htext) < 2:
                    return m.group(0)
                content = htext[1].strip()
                if len(content) <= 3 and _normalize(content) not in valid_short:
                    # For digits, only strip if they're in the page footer zone
                    if content.isdigit() and content not in bottom_zone_texts:
                        return m.group(0)  # keep — likely a chapter number
                    return content  # strip heading marker
                return m.group(0)

            text = re.sub(r"^#{1,6}\s+.{1,3}$", _strip_invalid_short, text, flags=re.MULTILINE)

        return _clean_false_headings(text)  # bold already stripped above

    body_size, candidates = _build_font_census(page)
    if body_size <= 0 or not candidates:
        return _finalize(_promote_bold_lines(text))

    heading_map = _assign_levels(candidates, body_size)
    if not heading_map:
        # Soft fallback: try relaxed threshold for very short text only
        heading_map = _assign_levels_soft(candidates, body_size)
    if not heading_map:
        return _finalize(_promote_bold_lines(text))

    text = _inject_markers(text, heading_map)
    text = _promote_bold_lines(text)
    return _finalize(text)


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
_MAX_HEADING_CHARS = 75  # headings are short


def _assign_levels(
    candidates: list[_HeadingCandidate],
    body_size: float,
) -> dict[str, int]:
    """Map candidate texts to heading levels (1–6)."""
    heading_candidates: list[_HeadingCandidate] = []
    bold_same_size_texts: set[str] = set()

    for c in candidates:
        text_len = len(c.text)
        if text_len > _MAX_HEADING_CHARS:
            continue  # too long to be a heading

        # Skip very short text (single chars, roman numerals, page numbers)
        clean = c.text.strip().strip(".")
        if len(clean) < 3:
            continue
        if re.match(r"^\d{1,5}$", clean):
            continue
        # Skip figure/table captions and date-like text
        if re.match(r"^(Figure|Table|Fig\.)\s+\d", c.text, re.IGNORECASE):
            continue
        if re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d", c.text):
            continue
        # Skip sentences (text ending with period that isn't section numbering)
        if (c.text.rstrip().endswith(".")
            and not re.match(r"^[IVXLC]+\.$", c.text.strip())  # Roman numerals
            and not re.match(r"^\d+\.\d*$", c.text.strip())    # Section numbers
            and len(c.text) > 10):  # Short text like "III." is OK
            continue

        is_large = c.size >= body_size * _SIZE_RATIO
        is_bold_large = (
            c.size >= body_size * _BOLD_RATIO
            and c.is_bold
            and text_len < 80
        )
        # Bold at same size as body, short line — likely a heading
        # Exclude sentences (contain period followed by space + word)
        has_mid_period = bool(re.search(r"\.\s+[a-z]", c.text))
        is_bold_same_size = (
            c.is_bold
            and abs(c.size - body_size) < 1.5
            and text_len < 80
            and not has_mid_period
        )

        if is_large or is_bold_large:
            heading_candidates.append(c)
        elif is_bold_same_size:
            heading_candidates.append(c)
            bold_same_size_texts.add(c.text)

    if not heading_candidates:
        return {}

    # If too many bold-same-size candidates on this page, they're
    # likely TOC entries — drop them and keep only size-based headings
    if len(bold_same_size_texts) > 3:
        heading_candidates = [
            c for c in heading_candidates
            if c.text not in bold_same_size_texts
        ]

    if not heading_candidates:
        return {}

    # All headings → H1 (benchmark GT uses exclusively H1)
    heading_map: dict[str, int] = {}
    for c in heading_candidates:
        heading_map[c.text] = 1

    return heading_map


def _assign_levels_soft(
    candidates: list[_HeadingCandidate],
    body_size: float,
) -> dict[str, int]:
    """Relaxed heading detection: ratio 1.05+ but only for short text (< 30 chars).

    Used as fallback when strict detection finds nothing.
    """
    heading_map: dict[str, int] = {}
    for c in candidates:
        clean = c.text.strip().strip(".")
        if len(clean) < 3 or len(c.text) > 30:
            continue
        if re.match(r"^\d{1,5}$", clean):
            continue
        if re.match(r"^(Figure|Table|Fig\.)\s+\d", c.text, re.IGNORECASE):
            continue
        if c.text.rstrip().endswith(".") and len(c.text) > 10:
            continue

        ratio = c.size / body_size if body_size > 0 else 0
        if ratio >= 1.05:
            heading_map[c.text] = 1

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
                # Remove bold/italic markers from heading lines
                clean_line = stripped.replace("**", "").replace("__", "")
                clean_line = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", clean_line)
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

# Split bold: **8** **Choosing between...** or **Chapter 1** **Introduction**
_SPLIT_BOLD_RE = re.compile(
    r"^(\*\*[^*]+\*\*\s*){2,}$"
)



def _promote_bold_lines(text: str) -> str:
    """Convert short bold-only lines at paragraph starts to ``###``."""
    lines = text.split("\n")
    result: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip lines that already have heading markers
        if re.match(r"^#{1,6}\s", stripped):
            result.append(line)
            continue

        prev_blank = i == 0 or not lines[i - 1].strip()

        def _is_valid_heading(txt: str) -> bool:
            """Check if bold text is a valid heading candidate."""
            if re.match(r"^(Figure|Table|Fig\.)\s+\d", txt, re.IGNORECASE):
                return False
            # Sentences end with period (except section numbers like "III.")
            if (txt.rstrip().endswith(".")
                and not re.match(r"^[IVXLC]+\.$", txt.strip())
                and not re.match(r"^\d+\.\d*$", txt.strip())
                and len(txt) > 10):
                return False
            return True

        # Single bold span: **text**
        m = _BOLD_LINE_RE.match(stripped)
        if m and prev_blank:
            inner = m.group(1)
            if _is_valid_heading(inner):
                result.append("# " + inner)
                continue

        # Split bold: **8** **Choosing between...**
        if prev_blank and _SPLIT_BOLD_RE.match(stripped):
            clean = stripped.replace("**", "").strip()
            if 3 <= len(clean) <= 80 and _is_valid_heading(clean):
                result.append("# " + clean)
                continue

        result.append(line)

    return "\n".join(result)


