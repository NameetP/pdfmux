"""Section-aware chunking + token estimation for LLM pipelines.

Splits extracted Markdown text into chunks at heading boundaries,
with per-chunk page tracking and token estimates.

Used by:
    - load_llm_context() public API
    - pdfmux convert --format llm
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Page separator used throughout pdfmux (pipeline.py, fast.py, json_fmt.py)
PAGE_SEPARATOR = "\n\n---\n\n"

# Heading pattern: ATX-style headings at start of line
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """A single section-aware chunk of a document."""

    title: str  # heading text, or "Page N" if no heading
    text: str  # the content under this heading
    page_start: int  # 1-indexed first page
    page_end: int  # 1-indexed last page
    tokens: int  # estimated token count
    confidence: float  # inherited from document confidence


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 approximation.

    This is the standard GPT-family approximation. No external
    tokenizer dependency needed. Good enough for context window
    planning and RAG chunk sizing.
    """
    return max(1, len(text.strip()) // 4)


def chunk_by_sections(
    text: str,
    confidence: float = 1.0,
) -> list[Chunk]:
    """Split text into section-aware chunks at heading boundaries.

    Strategy:
    1. Split on page separators to build a page offset map
    2. Find all ATX headings to identify section boundaries
    3. Map each section to page_start/page_end via character offsets
    4. No headings → fall back to one chunk per page

    Args:
        text: Post-processed Markdown text (with page separators).
        confidence: Document-level confidence score to inherit.

    Returns:
        List of Chunk objects in document order.
    """
    if not text or not text.strip():
        return []

    # Build page offset map: list of (start_offset, end_offset) per page
    page_offsets = _build_page_offsets(text)

    # Find heading-based sections
    sections = _find_sections(text)

    if sections:
        return _chunks_from_sections(text, sections, page_offsets, confidence)
    else:
        return _chunks_from_pages(text, page_offsets, confidence)


def _build_page_offsets(text: str) -> list[tuple[int, int]]:
    """Build a list of (start, end) character offsets for each page.

    Pages are delimited by PAGE_SEPARATOR in the text.
    """
    pages = text.split(PAGE_SEPARATOR)
    offsets = []
    pos = 0
    for page_text in pages:
        start = pos
        end = pos + len(page_text)
        offsets.append((start, end))
        pos = end + len(PAGE_SEPARATOR)  # skip the separator
    return offsets


def _offset_to_page(offset: int, page_offsets: list[tuple[int, int]]) -> int:
    """Convert a character offset to a 1-indexed page number."""
    for i, (start, end) in enumerate(page_offsets):
        if start <= offset <= end:
            return i + 1
    # Past the end → last page
    return len(page_offsets)


def _find_sections(text: str) -> list[tuple[str, int, int]]:
    """Find heading-based sections in the text.

    Returns list of (title, start_offset, end_offset) tuples.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        # Section ends at the next heading, or end of text
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, start, end))

    return sections


def _chunks_from_sections(
    text: str,
    sections: list[tuple[str, int, int]],
    page_offsets: list[tuple[int, int]],
    confidence: float,
) -> list[Chunk]:
    """Create chunks from heading-based sections."""
    chunks = []
    for title, start, end in sections:
        # Slice the full section text from the original document
        section_text = text[start:end].strip()
        if not section_text:
            continue

        page_start = _offset_to_page(start, page_offsets)
        page_end = _offset_to_page(max(start, end - 1), page_offsets)

        chunks.append(
            Chunk(
                title=title,
                text=section_text,
                page_start=page_start,
                page_end=page_end,
                tokens=estimate_tokens(section_text),
                confidence=confidence,
            )
        )
    return chunks


def _chunks_from_pages(
    text: str,
    page_offsets: list[tuple[int, int]],
    confidence: float,
) -> list[Chunk]:
    """Fallback: create one chunk per page when no headings exist."""
    pages = text.split(PAGE_SEPARATOR)
    chunks = []
    for i, page_text in enumerate(pages):
        page_text = page_text.strip()
        if not page_text:
            continue
        page_num = i + 1
        chunks.append(
            Chunk(
                title=f"Page {page_num}",
                text=page_text,
                page_start=page_num,
                page_end=page_num,
                tokens=estimate_tokens(page_text),
                confidence=confidence,
            )
        )
    return chunks
