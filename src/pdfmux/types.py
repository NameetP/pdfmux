"""Core types — the data model for pdfmux.

Six frozen types that flow through the entire pipeline:

    Quality           → extraction preset (fast / standard / high)
    OutputFormat      → output format (markdown / json / csv / llm)
    PageQuality       → per-page audit result (good / bad / empty)
    PageResult        → one page's extraction output + metadata
    DocumentResult    → full document output + metadata
    Chunk             → section-aware piece for LLM consumption
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Quality(Enum):
    """Extraction quality preset."""

    FAST = "fast"  # PyMuPDF only, skip audit
    STANDARD = "standard"  # multi-pass: fast → audit → selective OCR → merge
    HIGH = "high"  # LLM vision on every page


class OutputFormat(Enum):
    """Output format."""

    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    LLM = "llm"  # section-aware chunked JSON with token estimates


class PageQuality(Enum):
    """Per-page extraction quality classification."""

    GOOD = "good"  # text extraction succeeded
    BAD = "bad"  # has images but insufficient text — needs OCR
    EMPTY = "empty"  # near-zero text — needs OCR


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageResult:
    """Extraction result for a single page.

    Produced by extractors (one per page), consumed by the pipeline
    for auditing, re-extraction, and merging.
    """

    page_num: int  # 0-indexed
    text: str
    confidence: float  # 0.0–1.0, per-page
    quality: PageQuality
    extractor: str  # name of the extractor that produced this
    image_count: int = 0
    ocr_applied: bool = False

    @property
    def char_count(self) -> int:
        """Character count of stripped text."""
        return len(self.text.strip())


@dataclass(frozen=True)
class DocumentResult:
    """Complete result of processing a PDF through the pipeline.

    This is what process() returns. Contains the formatted output
    text plus all metadata needed for downstream consumption.
    """

    pages: tuple[PageResult, ...]
    source: str  # file path
    confidence: float  # content-weighted average across pages
    extractor_used: str  # summary string e.g. "pymupdf4llm + rapidocr (3 pages)"
    format: str  # output format name
    text: str  # the formatted output (markdown, JSON string, CSV, etc.)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    ocr_pages: tuple[int, ...] = field(default_factory=tuple)  # 0-indexed

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(frozen=True)
class Chunk:
    """Section-aware chunk for LLM consumption.

    Produced by chunk_by_sections(), used by load_llm_context()
    and the --format llm CLI output.
    """

    title: str  # heading text, or "Page N" if no heading
    text: str  # content under this heading
    page_start: int  # 1-indexed
    page_end: int  # 1-indexed
    tokens: int  # estimated token count (chars // 4)
    confidence: float  # inherited from document confidence
