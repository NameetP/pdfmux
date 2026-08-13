"""JSON formatter — structured output with metadata.

Useful for programmatic consumption, RAG pipelines that want
per-page chunks, or when you need extraction metadata alongside text.
"""

from __future__ import annotations

import json
import re

# Strip control characters except \n, \r, \t — these break JSON parsers
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def format_json(
    text: str,
    source: str = "",
    page_count: int = 0,
    confidence: float = 0.0,
    extractor: str = "",
    warnings: list[str] | None = None,
    ocr_pages: list[int] | None = None,
    *,
    error_code: str | None = None,
    tables: list | None = None,
    key_values: list | None = None,
    structured: dict | None = None,
    page_entries: list[dict] | None = None,
) -> str:
    """Format extracted text as structured JSON with locked schema.

    Args:
        text: Post-processed extracted text.
        source: Source file path.
        page_count: Number of pages in source PDF.
        confidence: Confidence score (0-1).
        extractor: Name of the extractor used.
        warnings: List of warning messages.
        ocr_pages: List of 0-indexed page numbers re-extracted with OCR.
        error_code: Structured error code (null on success).
        tables: Structured table data (list of dicts with headers/rows).
        key_values: Key-value pairs extracted from non-table regions.
        structured: Schema-mapped structured output (if schema was provided).
        page_entries: True per-page structures from the pipeline, each a dict
            with ``page`` (1-indexed), ``text``, and ``ocr``. When supplied,
            this is authoritative — the per-page split is NOT reconstructed by
            slicing ``content``. Callers that don't have per-page data omit it
            and fall back to separator-splitting (single blob when none found).

    Returns:
        JSON string with text and metadata.
    """
    # Sanitize control characters that break JSON parsers
    text = _CONTROL_CHAR_RE.sub("", text)

    ocr_set = set(ocr_pages or [])
    if page_entries is not None:
        # Authoritative per-page data from the pipeline. Do NOT re-split content:
        # pages are joined with "\n\n" (not the "---" separator below), so slicing
        # would collapse the whole document into one page and silently drop the
        # per-page confidence/ocr metadata that is the point of this format.
        pages_out = [
            {
                "page": int(p.get("page", i + 1)),
                "text": _CONTROL_CHAR_RE.sub("", p.get("text", "")),
                "ocr": bool(p.get("ocr", (int(p.get("page", i + 1)) - 1) in ocr_set)),
            }
            for i, p in enumerate(page_entries)
        ]
    else:
        # Fallback for standalone callers with only a joined string: split on the
        # page separator if present, else treat the whole text as one page.
        page_separator = "\n\n---\n\n"
        parts = text.split(page_separator) if page_separator in text else [text]
        pages_out = [
            {"page": i + 1, "text": p.strip(), "ocr": i in ocr_set} for i, p in enumerate(parts)
        ]

    output: dict = {
        "schema_version": "1.1.0",
        "source": source,
        "converter": "pdfmux",
        "extractor": extractor,
        "page_count": page_count,
        "confidence": round(confidence, 3),
        "error_code": error_code,
        "warnings": warnings or [],
        "ocr_pages": ocr_pages or [],
        "content": text,
        "pages": pages_out,
    }

    # Include structured data when available
    if tables:
        output["tables"] = tables
    if key_values:
        output["key_values"] = key_values
    if structured:
        output["structured"] = structured

    return json.dumps(output, indent=2, ensure_ascii=False)


def format_llm(
    text: str,
    source: str = "",
    confidence: float = 0.0,
    *,
    extractor: str = "",
    ocr_applied: bool = False,
) -> str:
    """Format extracted text as LLM-ready chunked JSON.

    Uses section-aware chunking to split the document at heading
    boundaries, with per-chunk token estimates and provenance.

    Args:
        text: Post-processed extracted text.
        source: Source file path.
        confidence: Document-level confidence score.
        extractor: Name of the extractor used.
        ocr_applied: Whether OCR was used on any page.

    Returns:
        JSON string with chunked structure.
    """
    from pdfmux.chunking import chunk_by_sections

    chunks = chunk_by_sections(
        text,
        confidence=confidence,
        extractor=extractor,
        ocr_applied=ocr_applied,
    )

    output = {
        "document": source,
        "chunks": [
            {
                "title": c.title,
                "text": c.text,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "tokens": c.tokens,
                "confidence": round(c.confidence, 3),
                "extractor": c.extractor,
                "ocr_applied": c.ocr_applied,
            }
            for c in chunks
        ],
    }

    return json.dumps(output, indent=2, ensure_ascii=False)
