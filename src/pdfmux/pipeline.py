"""Pipeline — the core of pdfmux.

Routes PDFs through the best extraction strategy:

    Standard mode → multi-pass: fast extract → audit → OCR bad pages → merge
    Has tables    → Docling (free, 0.3-3s/page)
    Fast mode     → PyMuPDF only (free, 0.01s/page)
    High mode     → Gemini Flash ($0.01-0.05)

Returns DocumentResult with streaming PageResults internally.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from pdfmux.audit import (
    audit_document,
    compute_document_confidence,
    score_page,
)
from pdfmux.detect import PDFClassification, classify
from pdfmux.errors import FileError, FormatError, OCRTimeoutError
from pdfmux.types import (
    OutputFormat,
    PageQuality,
    PageResult,
    Quality,
)

logger = logging.getLogger(__name__)

# OCR budget: cap at 30% of document pages in standard mode
OCR_BUDGET_RATIO = float(os.environ.get("PDFMUX_OCR_BUDGET", "0.30"))

# Dynamic OCR budget threshold: >50% graphical pages = OCR everything
IMAGE_HEAVY_THRESHOLD = 0.50

# --- Security limits ---
MAX_FILE_SIZE_MB = int(os.environ.get("PDFMUX_MAX_FILE_SIZE_MB", "500"))
MAX_PAGE_COUNT = int(os.environ.get("PDFMUX_MAX_PAGES", "10000"))
EXTRACTION_TIMEOUT_S = int(os.environ.get("PDFMUX_TIMEOUT", "300"))


# ---------------------------------------------------------------------------
# Legacy ConversionResult — kept for backward compat during transition
# ---------------------------------------------------------------------------


@dataclass
class ConversionResult:
    """Full result of converting a PDF (legacy wrapper)."""

    text: str
    format: str
    confidence: float
    extractor_used: str
    page_count: int
    warnings: list[str]
    classification: PDFClassification
    ocr_pages: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process(
    file_path: str | Path,
    output_format: str = "markdown",
    quality: str = "standard",
    show_confidence: bool = False,
    schema: str | None = None,
) -> ConversionResult:
    """Process a PDF through the tiered extraction pipeline.

    This is the main entry point. Returns a ConversionResult with
    extracted text formatted per output_format.

    Args:
        file_path: Path to the PDF file.
        output_format: "markdown" | "json" | "csv" | "llm".
        quality: "fast" | "standard" | "high".
        show_confidence: Whether to include confidence in markdown output.
        schema: Optional JSON schema file path or preset name for
            structured extraction. When provided, output includes
            tables, key-values, and schema-mapped structured data.

    Returns:
        ConversionResult with extracted text and metadata.

    Raises:
        FileError: If file doesn't exist or isn't a PDF.
        FormatError: If output_format is invalid.
    """
    file_path = Path(file_path)

    # Security: file size check
    if file_path.exists():
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise FileError(
                f"File too large: {file_size_mb:.0f}MB exceeds "
                f"{MAX_FILE_SIZE_MB}MB limit. Set PDFMUX_MAX_FILE_SIZE_MB to override.",
                code="PDF_TOO_LARGE",
            )

    # Validate format
    try:
        fmt = OutputFormat(output_format)
    except ValueError:
        raise FormatError(
            f"Unknown output format: {output_format}. "
            f"Valid formats: {', '.join(f.value for f in OutputFormat)}"
        )

    # Validate quality
    try:
        qual = Quality(quality)
    except ValueError:
        valid = ", ".join(q.value for q in Quality)
        raise FormatError(
            f"Unknown quality preset: {quality}. Valid presets: {valid}"
        )

    # Step 1: Classify the PDF
    classification = classify(file_path)

    # Security: page count check
    if classification.page_count > MAX_PAGE_COUNT:
        raise FileError(
            f"Too many pages: {classification.page_count} exceeds "
            f"{MAX_PAGE_COUNT} page limit. Set PDFMUX_MAX_PAGES to override.",
            code="PDF_TOO_LARGE",
        )

    # Step 2: Route to the best extractor, get page results (with timeout)
    if EXTRACTION_TIMEOUT_S > 0:
        from concurrent.futures import TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as _timeout_pool:
            _fut = _timeout_pool.submit(_route_and_extract, file_path, classification, qual)
            try:
                pages, extractor_name, ocr_pages = _fut.result(timeout=EXTRACTION_TIMEOUT_S)
            except FuturesTimeout:
                _fut.cancel()
                raise OCRTimeoutError(
                    f"Extraction timed out after {EXTRACTION_TIMEOUT_S}s. "
                    "Set PDFMUX_TIMEOUT to increase the limit."
                )
    else:
        pages, extractor_name, ocr_pages = _route_and_extract(file_path, classification, qual)

    # Step 2.5: Agentic multi-pass improvement (standard + high modes only)
    if qual != Quality.FAST:
        try:
            from pdfmux.agentic import agentic_improve

            budget_str = os.environ.get("PDFMUX_BUDGET")
            budget = float(budget_str) if budget_str else None

            pages, extractor_name, passes = agentic_improve(
                pages,
                file_path,
                extractor_name,
                budget=budget,
            )
            if passes > 1:
                logger.info("Agentic extraction used %d passes", passes)
        except Exception as e:
            logger.debug("Agentic improvement skipped: %s", e)

    # Step 2.75: Record telemetry (opt-in via PDFMUX_TELEMETRY=local)
    try:
        from pdfmux.router.learning import TelemetryCollector, is_telemetry_enabled

        if is_telemetry_enabled():
            collector = TelemetryCollector()
            page_type = _classify_to_page_type(classification)
            for p in pages:
                collector.record_extraction(
                    page_type=page_type,
                    extractor=p.extractor,
                    confidence=p.confidence,
                    latency_ms=0,  # per-page timing not tracked yet
                    cost_usd=p.cost_usd,
                )
    except Exception:
        pass  # telemetry should never break extraction

    # Step 3: Compute confidence
    unrecovered = sum(
        1 for p in pages if p.quality in (PageQuality.BAD, PageQuality.EMPTY) and not p.ocr_applied
    )
    confidence, warnings = compute_document_confidence(
        pages,
        ocr_page_count=len(ocr_pages),
        unrecovered_count=unrecovered,
    )

    # Step 4: Build the merged text
    merged_text = "\n\n".join(p.text for p in pages if p.text.strip())

    # Step 5: Post-process
    from pdfmux.postprocess import clean_text

    cleaned = clean_text(merged_text)

    # Step 6: Structured extraction (tables, key-values, schema mapping)
    structured_tables = None
    structured_kvs = None
    structured_output = None

    if fmt == OutputFormat.JSON or schema:
        structured_tables, structured_kvs, structured_output = _extract_structured(
            pages, schema
        )

    # Step 7: Format output
    formatted = _format_output(
        text=cleaned,
        output_format=fmt,
        source_path=file_path,
        show_confidence=show_confidence,
        extractor=extractor_name,
        confidence=confidence,
        page_count=classification.page_count,
        warnings=warnings,
        classification=classification,
        ocr_pages=ocr_pages,
        tables=structured_tables,
        key_values=structured_kvs,
        structured=structured_output,
    )

    # Cleanup: close cached PDF handles
    try:
        from pdfmux.pdf_cache import close_doc

        close_doc(file_path)
    except Exception:
        pass

    return ConversionResult(
        text=formatted,
        format=output_format,
        confidence=confidence,
        extractor_used=extractor_name,
        page_count=classification.page_count,
        warnings=warnings,
        classification=classification,
        ocr_pages=ocr_pages,
    )


def process_batch(
    file_paths: list[str | Path],
    output_format: str = "markdown",
    quality: str = "standard",
    workers: int = 4,
) -> Iterator[tuple[Path, ConversionResult | Exception]]:
    """Process multiple PDFs concurrently.

    Yields (path, result_or_error) tuples as each PDF completes.
    Errors are caught per-file — one failure doesn't stop the batch.

    Args:
        file_paths: List of PDF file paths.
        output_format: Output format for all files.
        quality: Quality preset for all files.
        workers: Number of concurrent workers.

    Yields:
        (Path, ConversionResult) on success.
        (Path, Exception) on failure.
    """

    def _process_one(path: Path) -> ConversionResult:
        return process(file_path=path, output_format=output_format, quality=quality)

    paths = [Path(p) for p in file_paths]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, p): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                yield path, result
            except Exception as e:
                yield path, e


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_and_extract(
    file_path: Path,
    classification: PDFClassification,
    quality: Quality,
) -> tuple[list[PageResult], str, list[int]]:
    """Route to the appropriate extractor using the RouterEngine.

    Uses the router for intelligent per-page-type decisions,
    with fallback to proven extraction patterns.

    Returns:
        (pages, extractor_name, ocr_page_numbers)
    """
    # Map Quality to router Strategy
    from pdfmux.router.strategies import Strategy

    strategy_map = {
        Quality.FAST: Strategy.ECONOMY,
        Quality.STANDARD: Strategy.BALANCED,
        Quality.HIGH: Strategy.PREMIUM,
    }
    strategy = strategy_map.get(quality, Strategy.BALANCED)

    # Fast/Economy mode: PyMuPDF only, skip audit
    if quality == Quality.FAST:
        from pdfmux.extractors.fast import FastExtractor

        ext = FastExtractor()
        pages = list(ext.extract(
            file_path,
            enhance_tables=classification.has_tables,
        ))
        return pages, ext.name, []

    # High/Premium mode: segment-aware LLM extraction
    if quality == Quality.HIGH:
        try:
            from pdfmux.segment import detect_segments, is_mixed_content

            # Check if any pages have mixed content worth segment-routing
            mixed_pages = []
            for page_num in range(classification.page_count):
                segs = detect_segments(file_path, page_num)
                if is_mixed_content(segs):
                    mixed_pages.append(page_num)

            if mixed_pages:
                logger.info(
                    "PREMIUM mode: %d pages with mixed content detected, "
                    "using segment-aware extraction",
                    len(mixed_pages),
                )
        except Exception as e:
            logger.debug("Segment detection skipped: %s", e)

        pages, name = _try_llm_extractor(file_path)
        return pages, name, []

    # Standard/Balanced mode: use RouterEngine for intelligent routing
    try:
        from pdfmux.router.engine import RouterEngine

        router = RouterEngine()

        # Determine primary page type from classification
        page_type = _classify_to_page_type(classification)
        budget_str = os.environ.get("PDFMUX_BUDGET")
        budget = float(budget_str) if budget_str else None

        decision = router.select(page_type, strategy, budget)
        logger.info("Router decision: %s (%s)", decision.extractor, decision.reason)

        # Execute the router's decision
        pages, name, ocr_pages = _execute_route_decision(
            file_path, classification, decision
        )
        if pages:
            return pages, name, ocr_pages

    except Exception as e:
        logger.debug("Router failed, using legacy routing: %s", e)

    # Fallback: legacy routing (if router fails or returns empty)
    return _legacy_route_and_extract(file_path, classification)


def _classify_to_page_type(classification: PDFClassification) -> str:
    """Convert PDFClassification to a router page type string."""
    if classification.is_scanned:
        return "scanned"
    if classification.has_tables:
        return "tables"
    if classification.is_graphical:
        return "graphical"
    if classification.is_mixed:
        return "mixed"
    return "digital"


def _execute_route_decision(
    file_path: Path,
    classification: PDFClassification,
    decision,
) -> tuple[list[PageResult], str, list[int]]:
    """Execute a RouteDecision by calling the appropriate extractor."""
    extractor_name = decision.extractor

    if extractor_name == "opendataloader":
        try:
            from pdfmux.extractors.opendataloader import OpenDataLoaderExtractor

            odl = OpenDataLoaderExtractor()
            if odl.available():
                pages = list(odl.extract(file_path))
                if pages and any(len(p.text.strip()) > 10 for p in pages):
                    return pages, odl.name, []
        except Exception:
            pass

    elif extractor_name == "docling":
        if classification.has_tables:
            pages, name = _try_targeted_table_extraction(file_path, classification)
            return pages, name, []

    elif extractor_name == "llm":
        pages, name = _try_llm_extractor(file_path)
        return pages, name, []

    elif extractor_name == "pymupdf":
        pass  # fall through to multipass below

    # Default: multi-pass extraction with table overlay
    pages, name, ocr_pages = _multipass_extract(file_path, classification)
    pages, overlay_applied = _overlay_docling_tables(file_path, pages)
    if overlay_applied:
        name += " + docling-tables"
    return pages, name, ocr_pages


def _legacy_route_and_extract(
    file_path: Path,
    classification: PDFClassification,
) -> tuple[list[PageResult], str, list[int]]:
    """Legacy routing — used as fallback if RouterEngine fails."""
    # OpenDataLoader → use when available
    try:
        from pdfmux.extractors.opendataloader import OpenDataLoaderExtractor

        odl = OpenDataLoaderExtractor()
        if odl.available():
            pages = list(odl.extract(file_path))
            if pages and any(len(p.text.strip()) > 10 for p in pages):
                return pages, odl.name, []
    except Exception:
        pass

    # Tables → targeted Docling
    if classification.has_tables and not classification.is_graphical:
        pages, name = _try_targeted_table_extraction(file_path, classification)
        return pages, name, []

    # Standard → multi-pass + Docling table overlay
    pages, name, ocr_pages = _multipass_extract(file_path, classification)
    pages, overlay_applied = _overlay_docling_tables(file_path, pages)
    if overlay_applied:
        name += " + docling-tables"
    return pages, name, ocr_pages


def _try_table_extractor(file_path: Path) -> tuple[list[PageResult], str]:
    """Try Docling, fall back to PyMuPDF."""
    try:
        from pdfmux.extractors.tables import TableExtractor

        ext = TableExtractor()
        if ext.available():
            pages = list(ext.extract(file_path))
            return pages, ext.name
    except Exception:
        pass

    logger.info("Docling not available, falling back to PyMuPDF for tables")
    from pdfmux.extractors.fast import FastExtractor

    ext = FastExtractor()
    pages = list(ext.extract(file_path))
    return pages, ext.name


def _try_llm_extractor(file_path: Path) -> tuple[list[PageResult], str]:
    """Try Gemini Flash, fall back through the chain."""
    try:
        from pdfmux.extractors.llm import LLMExtractor

        ext = LLMExtractor()
        if ext.available():
            pages = list(ext.extract(file_path))
            return pages, ext.name
    except Exception:
        pass

    # Fall back to OCR
    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ext = RapidOCRExtractor()
        if ext.available():
            pages = list(ext.extract(file_path))
            return pages, ext.name
    except Exception:
        pass

    logger.info("No premium extractor available, falling back to PyMuPDF")
    from pdfmux.extractors.fast import FastExtractor

    ext = FastExtractor()
    pages = list(ext.extract(file_path))
    return pages, ext.name


def _try_targeted_table_extraction(
    file_path: Path,
    classification: PDFClassification,
) -> tuple[list[PageResult], str]:
    """Hybrid extraction: Docling for table pages, fast for the rest.

    For documents with <50 total pages, uses full-document Docling.
    For larger documents, identifies table-candidate pages and extracts
    only those with Docling while using PyMuPDF for the rest.
    """
    if classification.page_count <= 50:
        return _try_table_extractor(file_path)

    try:
        from pdfmux.extractors.tables import TableExtractor

        ext = TableExtractor()
        if not ext.available():
            raise ImportError

        table_pages = _identify_table_pages(file_path)

        if not table_pages or len(table_pages) > 100:
            return _try_table_extractor(file_path)

        # Hybrid: fast for non-table pages, Docling for table pages
        from pdfmux.extractors.fast import FastExtractor

        fast = FastExtractor()
        fast_pages = {p.page_num: p for p in fast.extract(file_path)}

        docling_pages = {p.page_num: p for p in ext.extract_pages(file_path, table_pages)}

        # Merge: prefer Docling results for table pages
        merged = []
        for page_num in sorted(fast_pages.keys()):
            if page_num in docling_pages:
                merged.append(docling_pages[page_num])
            else:
                merged.append(fast_pages[page_num])

        n_docling = len(docling_pages)
        return merged, f"pymupdf4llm + docling ({n_docling} table pages)"

    except Exception:
        logger.info("Targeted table extraction failed, falling back")
        return _try_table_extractor(file_path)


def _identify_table_pages(file_path: Path) -> list[int]:
    """Identify pages likely to contain tables using lightweight heuristics."""
    try:
        from pdfmux.pdf_cache import get_doc

        doc = get_doc(file_path)
    except ImportError:
        import fitz

        doc = fitz.open(str(file_path))
    table_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split("\n")

        number_dense = 0
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 20:
                continue
            non_space = stripped.replace(" ", "")
            if not non_space:
                continue
            numeric = sum(1 for c in non_space if c in "0123456789$,%.()-")
            if numeric / len(non_space) >= 0.30:
                number_dense += 1

        if number_dense >= 3:
            table_pages.append(page_num)

    return table_pages


# ---------------------------------------------------------------------------
# Docling table overlay — merge Docling tables into PyMuPDF text
# ---------------------------------------------------------------------------


def _extract_table_blocks(text: str) -> list[str]:
    """Extract markdown pipe-table blocks from text."""
    lines = text.split("\n")
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(stripped)
        else:
            if len(current) >= 2:  # header + separator (or data row)
                blocks.append("\n".join(current))
            current = []

    if len(current) >= 2:
        blocks.append("\n".join(current))

    return blocks


def _is_toc_table(table_block: str) -> bool:
    """Detect if a table is likely a Table of Contents (should not be injected).

    TOC tables typically have 2-3 columns, 8+ rows, with page numbers in the
    last column. Short data tables (<8 rows) are never classified as TOCs.
    """
    import re

    lines = table_block.split("\n")
    data_rows = [l for l in lines if not re.match(r"^\|[\s\-:|]+\|$", l.strip())]

    # Short tables can't be TOCs — real TOCs have many chapters/sections
    if len(data_rows) < 8:
        return False

    # Count columns and collect last-column values
    # Strict page number: pure digits or pure roman numerals only
    page_num_re = re.compile(r"^(\d{1,4}|[ivxlc]{1,6})$", re.IGNORECASE)
    last_col_values: list[str] = []
    max_cols = 0

    for row in data_rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        max_cols = max(max_cols, len(cells))
        if cells:
            last_col_values.append(cells[-1])

    # Only check 2-3 column tables (data tables typically have 4+)
    if max_cols > 3:
        return False

    if not last_col_values:
        return False

    # Check if last column is predominantly page numbers
    page_num_count = sum(
        1 for v in last_col_values
        if page_num_re.match(v.strip())
    )
    return page_num_count / len(last_col_values) > 0.6


def _overlay_docling_tables(
    file_path: Path,
    pages: list[PageResult],
) -> tuple[list[PageResult], bool]:
    """Try Docling on pages without tables; merge any tables found.

    Runs Docling extraction and extracts ONLY markdown table blocks.
    Appends those to PyMuPDF text for pages where PyMuPDF found no tables.
    Deduplicates: skips tables whose content already appears in the text
    (e.g. TOCs that Docling formats as tables).

    Returns:
        (enhanced_pages, overlay_was_applied)
    """
    try:
        from pdfmux.extractors.tables import TableExtractor

        ext = TableExtractor()
        if not ext.available():
            return pages, False
    except Exception:
        return pages, False

    # Find pages that have no tables in current text
    pages_without_tables: list[int] = []
    for p in pages:
        if not _extract_table_blocks(p.text):
            pages_without_tables.append(p.page_num)

    if not pages_without_tables:
        return pages, False

    # Run Docling on pages without tables
    try:
        docling_results = list(ext.extract_pages(file_path, pages_without_tables))
    except Exception:
        logger.debug("Docling table overlay failed, continuing without")
        return pages, False

    # Build lookup: page_num → list of data table blocks from Docling
    # Filter out TOC-like tables (chapter lists with page numbers)
    docling_tables: dict[int, list[str]] = {}
    for dp in docling_results:
        tables = _extract_table_blocks(dp.text)
        if not tables:
            continue

        data_tables = [t for t in tables if not _is_toc_table(t)]
        if data_tables:
            docling_tables[dp.page_num] = data_tables

    if not docling_tables:
        return pages, False

    # Merge: append Docling tables to PyMuPDF text
    enhanced: list[PageResult] = []
    for p in pages:
        if p.page_num in docling_tables:
            table_md = "\n\n".join(docling_tables[p.page_num])
            enhanced.append(
                PageResult(
                    page_num=p.page_num,
                    text=p.text.rstrip() + "\n\n" + table_md,
                    confidence=p.confidence,
                    quality=p.quality,
                    extractor=p.extractor,
                    image_count=p.image_count,
                    ocr_applied=p.ocr_applied,
                    tables=p.tables,
                )
            )
        else:
            enhanced.append(p)

    n_enhanced = len(docling_tables)
    logger.info(f"Docling table overlay: added tables to {n_enhanced} pages")
    return enhanced, True


# ---------------------------------------------------------------------------
# Multi-pass pipeline
# ---------------------------------------------------------------------------


def _compute_ocr_budget(classification: PDFClassification) -> float:
    """Compute OCR budget ratio based on document classification.

    Rules:
        - If >50% of pages are graphical: budget = 1.0 (OCR all)
        - If >25% graphical: budget = graphical_ratio + 0.10 (generous)
        - Otherwise: use default OCR_BUDGET_RATIO (0.30)
    """
    if classification.page_count == 0:
        return OCR_BUDGET_RATIO

    graphical_ratio = len(classification.graphical_pages) / classification.page_count

    if graphical_ratio >= IMAGE_HEAVY_THRESHOLD:
        return 1.0
    elif graphical_ratio > 0.25:
        return min(1.0, graphical_ratio + 0.10)
    else:
        return OCR_BUDGET_RATIO


def _multipass_extract(
    file_path: Path,
    classification: PDFClassification,
) -> tuple[list[PageResult], str, list[int]]:
    """Multi-pass extraction: fast → audit → OCR bad pages → merge.

    1. Fast-extract every page and audit quality
    2. If all pages good → return immediately (zero overhead)
    3. For bad/empty pages → re-extract with RapidOCR
    4. For unrecovered pages → try LLM
    5. Merge good fast text + OCR/LLM text in page order
    """
    # Pass 1: Fast extract + audit
    audit = audit_document(file_path)

    # Convert audit results to PageResult objects
    fast_pages: list[PageResult] = []
    for pa in audit.pages:
        confidence = score_page(pa.text, pa.image_count)
        fast_pages.append(
            PageResult(
                page_num=pa.page_num,
                text=pa.text,
                confidence=confidence,
                quality=PageQuality(pa.quality),
                extractor="pymupdf4llm",
                image_count=pa.image_count,
            )
        )

    # Fast path — all pages good
    if not audit.needs_ocr:
        return fast_pages, "pymupdf4llm", []

    # Pass 2: Re-extract bad/empty pages with OCR
    all_pages_needing_ocr = audit.bad_pages + audit.empty_pages
    ocr_results: dict[int, str] = {}
    ocr_name = ""
    budget_warnings: list[str] = []

    # OCR budget: dynamic based on classification
    effective_budget = _compute_ocr_budget(classification)
    max_ocr_pages = max(1, int(classification.page_count * effective_budget))
    if len(all_pages_needing_ocr) > max_ocr_pages:
        # Prioritize "bad" pages (some text) over "empty" pages
        prioritized = sorted(
            all_pages_needing_ocr,
            key=lambda pn: (0 if audit.pages[pn].quality == "bad" else 1, pn),
        )
        pages_needing_ocr = prioritized[:max_ocr_pages]
        skipped = len(all_pages_needing_ocr) - max_ocr_pages
        logger.warning(
            f"OCR budget: processing {max_ocr_pages} of {len(all_pages_needing_ocr)} "
            f"pages (budget={effective_budget:.0%} of {classification.page_count}). "
            f"Skipping {skipped} pages."
        )
        budget_warnings.append(
            f"{skipped} pages skipped due to OCR budget. Use --quality high to process all pages."
        )
    else:
        pages_needing_ocr = all_pages_needing_ocr

    logger.info(
        f"Multi-pass: {len(pages_needing_ocr)} pages need re-extraction "
        f"(bad={len(audit.bad_pages)}, empty={len(audit.empty_pages)})"
    )

    # Region OCR for "bad" pages — preserve existing good text
    bad_pages_set = set(audit.bad_pages)
    region_ocr_pages = [p for p in pages_needing_ocr if p in bad_pages_set]
    full_ocr_pages = [p for p in pages_needing_ocr if p not in bad_pages_set]

    if region_ocr_pages:
        try:
            from pdfmux.regions import region_ocr_page

            for page_num in region_ocr_pages:
                page_audit = audit.pages[page_num]
                merged, n_regions = region_ocr_page(
                    file_path,
                    page_num,
                    page_audit.text,
                )
                if n_regions > 0 and len(merged.strip()) > len(page_audit.text.strip()):
                    ocr_results[page_num] = merged
                    logger.info(f"Region OCR page {page_num}: recovered {n_regions} regions")
        except ImportError:
            logger.debug("Region OCR requires RapidOCR — falling back to full-page OCR")
            full_ocr_pages = pages_needing_ocr

    # Full-page OCR for empty pages (and bad pages that region OCR didn't help)
    still_need_full_ocr = [p for p in full_ocr_pages if p not in ocr_results]
    still_need_full_ocr.extend(p for p in region_ocr_pages if p not in ocr_results)

    # Try RapidOCR for remaining pages — now with parallel dispatch
    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ocr = RapidOCRExtractor()
        if ocr.available() and still_need_full_ocr:
            from pdfmux.parallel import parallel_ocr

            ocr_raw = parallel_ocr(file_path, still_need_full_ocr, ocr)
            for page_num, page_ocr in ocr_raw.items():
                if not page_ocr.success:
                    continue
                ocr_text = page_ocr.text
                page_audit = audit.pages[page_num]

                if page_audit.quality == "empty":
                    if len(ocr_text.strip()) > 10:
                        ocr_results[page_num] = ocr_text
                elif page_audit.quality == "bad":
                    if len(ocr_text.strip()) > len(page_audit.text.strip()):
                        ocr_results[page_num] = ocr_text

            ocr_name = "rapidocr"
    except Exception:
        logger.info("RapidOCR not available, trying legacy OCR")
        try:
            from pdfmux.extractors.ocr import OCRExtractor

            ocr_legacy = OCRExtractor()
            if ocr_legacy.available():
                for page_num in still_need_full_ocr:
                    ocr_pages_result = list(ocr_legacy.extract(file_path, pages=[page_num]))
                    ocr_text = ocr_pages_result[0].text if ocr_pages_result else ""
                    page_audit = audit.pages[page_num]

                    if page_audit.quality == "empty":
                        if len(ocr_text.strip()) > 10:
                            ocr_results[page_num] = ocr_text
                    elif page_audit.quality == "bad":
                        if len(ocr_text.strip()) > len(page_audit.text.strip()):
                            ocr_results[page_num] = ocr_text

                ocr_name = "surya"
        except Exception:
            logger.info("No OCR installed")

    # Pass 3: Try LLM on unrecovered pages
    still_bad = [p for p in pages_needing_ocr if p not in ocr_results]
    if still_bad:
        try:
            from pdfmux.extractors.llm import LLMExtractor

            llm = LLMExtractor()
            if llm.available():
                for page_num in still_bad:
                    llm_pages = list(llm.extract(file_path, pages=[page_num]))
                    llm_text = llm_pages[0].text if llm_pages else ""
                    page_audit = audit.pages[page_num]

                    if page_audit.quality == "empty":
                        if len(llm_text.strip()) > 10:
                            ocr_results[page_num] = llm_text
                    elif page_audit.quality == "bad":
                        if len(llm_text.strip()) > len(page_audit.text.strip()):
                            ocr_results[page_num] = llm_text

                if not ocr_name:
                    ocr_name = "gemini"
                else:
                    ocr_name += " + gemini"
        except Exception:
            pass

    # Merge: replace fast pages with OCR results where available
    merged_pages: list[PageResult] = []
    ocr_page_list: list[int] = sorted(ocr_results.keys())

    # Re-inject headings into OCR-replaced pages (OCR strips heading markers)
    from pdfmux.headings import inject_headings as _inject_headings

    try:
        from pdfmux.pdf_cache import get_doc as _get_doc

        _fitz_doc = _get_doc(file_path)
    except ImportError:
        import fitz as _fitz

        _fitz_doc = _fitz.open(str(file_path))

    for fp in fast_pages:
        if fp.page_num in ocr_results:
            ocr_text = ocr_results[fp.page_num]
            if fp.page_num < len(_fitz_doc):
                ocr_text = _inject_headings(ocr_text, _fitz_doc[fp.page_num])
            merged_pages.append(
                PageResult(
                    page_num=fp.page_num,
                    text=ocr_text,
                    confidence=score_page(ocr_text, fp.image_count),
                    quality=PageQuality.GOOD,
                    extractor=ocr_name or "ocr",
                    image_count=fp.image_count,
                    ocr_applied=True,
                )
            )
        else:
            merged_pages.append(fp)


    # Build extractor name
    n_ocr = len(ocr_page_list)
    n_unrecovered = len(still_bad) - sum(1 for p in still_bad if p in ocr_results)

    if n_ocr > 0:
        name = f"pymupdf4llm + {ocr_name} ({n_ocr} pages re-extracted)"
    else:
        name = "pymupdf4llm"

    if n_unrecovered > 0:
        logger.warning(
            f"Multi-pass: {n_unrecovered} pages could not be recovered. "
            f"Install pdfmux[ocr] for better results."
        )

    logger.info(
        f"Multi-pass complete: {len(audit.good_pages)} good, "
        f"{n_ocr} re-extracted, {n_unrecovered} unrecovered"
    )

    return merged_pages, name, ocr_page_list


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _extract_structured(
    pages: list[PageResult],
    schema_path: str | None = None,
) -> tuple[list[dict] | None, list[dict] | None, dict | None]:
    """Extract structured data (tables, key-values) from pages.

    Args:
        pages: Extracted page results.
        schema_path: Optional schema file for schema-guided mapping.

    Returns:
        (tables_list, kv_list, schema_mapped_output)
    """
    from pdfmux.kv_extract import extract_key_values
    from pdfmux.normalize import auto_normalize
    from pdfmux.types import ExtractedTable

    # Collect all tables from pages
    all_tables: list[ExtractedTable] = []
    for page in pages:
        all_tables.extend(page.tables)

    # Serialize tables for JSON output
    tables_json = None
    if all_tables:
        tables_json = [
            {
                "page": t.page_num + 1,  # 1-indexed for user-facing
                "headers": list(t.headers),
                "rows": [list(row) for row in t.rows],
                **({"label": t.label} if t.label else {}),
            }
            for t in all_tables
        ]

    # Extract key-value pairs from all pages
    all_kvs = []
    for page in pages:
        kvs = extract_key_values(page.text, page_num=page.page_num)
        all_kvs.extend(kvs)

    # Deduplicate by key (keep first occurrence)
    seen_keys: set[str] = set()
    deduped_kvs = []
    for kv in all_kvs:
        norm = kv.key.lower()
        if norm not in seen_keys:
            deduped_kvs.append(kv)
            seen_keys.add(norm)

    # Serialize key-values with normalization
    kvs_json = None
    if deduped_kvs:
        kvs_json = []
        for kv in deduped_kvs:
            normalized = auto_normalize(kv.key, kv.value)
            entry: dict = {
                "key": kv.key,
                "value": kv.value,
                "page": kv.page_num + 1,
            }
            if normalized != kv.value:
                entry["normalized"] = normalized
            kvs_json.append(entry)

    # Schema-guided mapping
    structured_output = None
    if schema_path:
        try:
            from pdfmux.schema import load_schema, map_to_schema

            schema = load_schema(schema_path)
            structured_output = map_to_schema(
                list(all_tables), deduped_kvs, schema
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Schema mapping failed: {e}")

    return tables_json, kvs_json, structured_output


def _format_output(
    text: str,
    output_format: OutputFormat,
    source_path: Path,
    show_confidence: bool,
    extractor: str,
    confidence: float,
    page_count: int,
    warnings: list[str],
    classification: PDFClassification,
    *,
    ocr_pages: list[int] | None = None,
    tables: list[dict] | None = None,
    key_values: list[dict] | None = None,
    structured: dict | None = None,
) -> str:
    """Format the processed text into the requested output format."""
    if output_format == OutputFormat.MARKDOWN:
        from pdfmux.formatters.markdown import format_markdown

        result = format_markdown(text, source=str(source_path))
        if show_confidence:
            confidence_note = (
                f"\n\n---\n*Conversion confidence: {confidence:.0%} ({page_count} pages)*"
            )
            if warnings:
                confidence_note += "\n" + "\n".join(f"- ⚠ {w}" for w in warnings)
            result += confidence_note
        return result

    if output_format == OutputFormat.JSON:
        from pdfmux.formatters.json_fmt import format_json

        return format_json(
            text=text,
            source=str(source_path),
            page_count=page_count,
            confidence=confidence,
            extractor=extractor,
            warnings=warnings,
            ocr_pages=ocr_pages,
            tables=tables,
            key_values=key_values,
            structured=structured,
        )

    if output_format == OutputFormat.LLM:
        from pdfmux.formatters.json_fmt import format_llm

        return format_llm(
            text=text,
            source=str(source_path),
            confidence=confidence,
            extractor=extractor,
            ocr_applied=bool(ocr_pages),
        )

    if output_format == OutputFormat.CSV:
        from pdfmux.formatters.csv_fmt import format_csv

        return format_csv(text)

    raise FormatError(f"Unknown output format: {output_format}")
