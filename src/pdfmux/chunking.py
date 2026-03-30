"""Section-aware chunking + token estimation for LLM pipelines.

Splits extracted Markdown text into chunks at heading boundaries,
with per-chunk page tracking and token estimates.

Used by:
    - load_llm_context() public API
    - pdfmux convert --format llm
"""

from __future__ import annotations

import re

from pdfmux.types import Chunk

# Page separator used throughout pdfmux
PAGE_SEPARATOR = "\n\n---\n\n"

# Heading pattern: ATX-style headings at start of line
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 approximation.

    Standard GPT-family approximation. No external tokenizer needed.
    """
    return max(1, len(text.strip()) // 4)


def chunk_by_sections(
    text: str,
    confidence: float = 1.0,
    *,
    extractor: str = "",
    ocr_applied: bool = False,
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
        extractor: Name of the extractor that produced the text.
        ocr_applied: Whether OCR was used on any page.

    Returns:
        List of Chunk objects in document order.
    """
    if not text or not text.strip():
        return []

    page_offsets = _build_page_offsets(text)
    sections = _find_sections(text)

    if sections:
        return _chunks_from_sections(
            text, sections, page_offsets, confidence, extractor, ocr_applied
        )
    else:
        return _chunks_from_pages(text, page_offsets, confidence, extractor, ocr_applied)


def _build_page_offsets(text: str) -> list[tuple[int, int]]:
    """Build (start, end) character offsets for each page."""
    pages = text.split(PAGE_SEPARATOR)
    offsets = []
    pos = 0
    for page_text in pages:
        start = pos
        end = pos + len(page_text)
        offsets.append((start, end))
        pos = end + len(PAGE_SEPARATOR)
    return offsets


def _offset_to_page(offset: int, page_offsets: list[tuple[int, int]]) -> int:
    """Convert a character offset to a 1-indexed page number."""
    for i, (start, end) in enumerate(page_offsets):
        if start <= offset <= end:
            return i + 1
    return len(page_offsets)


def _find_sections(text: str) -> list[tuple[str, int, int]]:
    """Find heading-based sections. Returns (title, start, end) tuples."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, start, end))

    return sections


def _chunks_from_sections(
    text: str,
    sections: list[tuple[str, int, int]],
    page_offsets: list[tuple[int, int]],
    confidence: float,
    extractor: str = "",
    ocr_applied: bool = False,
) -> list[Chunk]:
    """Create chunks from heading-based sections."""
    chunks = []
    for title, start, end in sections:
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
                extractor=extractor,
                ocr_applied=ocr_applied,
            )
        )
    return chunks


def chunk_for_rag(
    text: str,
    confidence: float = 1.0,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    extractor: str = "",
    ocr_applied: bool = False,
) -> list[Chunk]:
    """RAG-optimized chunking with token bounds and overlap.

    Strategy:
    1. Split at heading boundaries (layout-aware)
    2. If a chunk exceeds max_tokens, sub-split at paragraph boundaries
    3. Apply overlap: repeat last N tokens at start of next chunk
    4. Attach metadata: page range, confidence, extractor

    Args:
        text: Post-processed Markdown text (with page separators).
        confidence: Document-level confidence score.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Tokens to repeat between adjacent chunks.
        extractor: Extractor name for metadata.
        ocr_applied: Whether OCR was used.

    Returns:
        List of Chunk objects, token-bounded with overlap.
    """
    if not text or not text.strip():
        return []

    # Step 1: Get section-based chunks (existing logic)
    raw_chunks = chunk_by_sections(
        text, confidence, extractor=extractor, ocr_applied=ocr_applied
    )

    if not raw_chunks:
        return []

    # Step 2: Split oversized chunks at paragraph boundaries
    bounded_chunks: list[Chunk] = []
    for chunk in raw_chunks:
        if chunk.tokens <= max_tokens:
            bounded_chunks.append(chunk)
        else:
            sub_chunks = _split_chunk(chunk, max_tokens)
            bounded_chunks.extend(sub_chunks)

    # Step 3: Apply overlap
    if overlap_tokens > 0 and len(bounded_chunks) > 1:
        bounded_chunks = _apply_overlap(bounded_chunks, overlap_tokens)

    # Step 4: Build result
    result = []
    for i, chunk in enumerate(bounded_chunks):
        result.append(
            Chunk(
                title=chunk.title,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                tokens=estimate_tokens(chunk.text),
                confidence=chunk.confidence,
                extractor=chunk.extractor,
                ocr_applied=chunk.ocr_applied,
            )
        )

    return result


def _split_chunk(chunk: Chunk, max_tokens: int) -> list[Chunk]:
    """Split an oversized chunk at paragraph or sentence boundaries."""
    paragraphs = re.split(r"\n\n+", chunk.text)

    # Further split any paragraph that still exceeds max_tokens at sentence boundaries
    expanded = []
    for para in paragraphs:
        if estimate_tokens(para) > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            if len(sentences) > 1:
                expanded.extend(sentences)
            else:
                expanded.append(para)
        else:
            expanded.append(para)
    paragraphs = expanded

    sub_chunks = []
    current_text = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_text:
            # Emit current chunk
            sub_chunks.append(
                Chunk(
                    title=chunk.title,
                    text=current_text.strip(),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    tokens=estimate_tokens(current_text),
                    confidence=chunk.confidence,
                    extractor=chunk.extractor,
                    ocr_applied=chunk.ocr_applied,
                )
            )
            current_text = para + "\n\n"
            current_tokens = para_tokens
        else:
            current_text += para + "\n\n"
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_text.strip():
        sub_chunks.append(
            Chunk(
                title=chunk.title,
                text=current_text.strip(),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                tokens=estimate_tokens(current_text),
                confidence=chunk.confidence,
                extractor=chunk.extractor,
                ocr_applied=chunk.ocr_applied,
            )
        )

    return sub_chunks if sub_chunks else [chunk]


def _apply_overlap(chunks: list[Chunk], overlap_tokens: int) -> list[Chunk]:
    """Add overlap text from previous chunk to start of next chunk."""
    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1].text
        prev_words = prev_text.split()

        # Take last N words as overlap (tokens ≈ words for English)
        overlap_words = prev_words[-overlap_tokens:] if len(prev_words) > overlap_tokens else []
        overlap_text = " ".join(overlap_words)

        if overlap_text:
            new_text = f"...{overlap_text}\n\n{chunks[i].text}"
        else:
            new_text = chunks[i].text

        result.append(
            Chunk(
                title=chunks[i].title,
                text=new_text,
                page_start=chunks[i].page_start,
                page_end=chunks[i].page_end,
                tokens=estimate_tokens(new_text),
                confidence=chunks[i].confidence,
                extractor=chunks[i].extractor,
                ocr_applied=chunks[i].ocr_applied,
            )
        )

    return result


def _chunks_from_pages(
    text: str,
    page_offsets: list[tuple[int, int]],
    confidence: float,
    extractor: str = "",
    ocr_applied: bool = False,
) -> list[Chunk]:
    """Fallback: one chunk per page when no headings exist."""
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
                extractor=extractor,
                ocr_applied=ocr_applied,
            )
        )
    return chunks
