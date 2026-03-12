"""JSON formatter — structured output with metadata.

Useful for programmatic consumption, RAG pipelines that want
per-page chunks, or when you need extraction metadata alongside text.
"""

from __future__ import annotations

import json


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

    Returns:
        JSON string with text and metadata.
    """
    # Split into pages if page separators exist
    ocr_set = set(ocr_pages or [])
    page_separator = "\n\n---\n\n"
    if page_separator in text:
        pages = [p.strip() for p in text.split(page_separator)]
    else:
        pages = [text]

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
        "pages": [
            {"page": i + 1, "text": page, "ocr": i in ocr_set} for i, page in enumerate(pages)
        ],
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
