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
) -> str:
    """Format extracted text as structured JSON.

    Args:
        text: Post-processed extracted text.
        source: Source file path.
        page_count: Number of pages in source PDF.
        confidence: Confidence score (0-1).
        extractor: Name of the extractor used.
        warnings: List of warning messages.

    Returns:
        JSON string with text and metadata.
    """
    # Split into pages if page separators exist
    page_separator = "\n\n---\n\n"
    if page_separator in text:
        pages = [p.strip() for p in text.split(page_separator)]
    else:
        pages = [text]

    output = {
        "source": source,
        "converter": "readable",
        "extractor": extractor,
        "page_count": page_count,
        "confidence": round(confidence, 3),
        "warnings": warnings or [],
        "content": text,
        "pages": [{"page": i + 1, "text": page} for i, page in enumerate(pages)],
    }

    return json.dumps(output, indent=2, ensure_ascii=False)
