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
from pdfmux.errors import FormatError
from pdfmux.types import (
    OutputFormat,
    PageQuality,
    PageResult,
    Quality,
)

logger = logging.getLogger(__name__)


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
) -> ConversionResult:
    """Process a PDF through the tiered extraction pipeline.

    This is the main entry point. Returns a ConversionResult with
    extracted text formatted per output_format.

    Args:
        file_path: Path to the PDF file.
        output_format: "markdown" | "json" | "csv" | "llm".
        quality: "fast" | "standard" | "high".
        show_confidence: Whether to include confidence in markdown output.

    Returns:
        ConversionResult with extracted text and metadata.

    Raises:
        FileError: If file doesn't exist or isn't a PDF.
        FormatError: If output_format is invalid.
    """
    file_path = Path(file_path)

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
        qual = Quality.STANDARD

    # Step 1: Classify the PDF
    classification = classify(file_path)

    # Step 2: Route to the best extractor, get page results
    pages, extractor_name, ocr_pages = _route_and_extract(file_path, classification, qual)

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
    merged_text = "\n\n---\n\n".join(p.text for p in pages if p.text.strip())

    # Step 5: Post-process
    from pdfmux.postprocess import clean_text

    cleaned = clean_text(merged_text)

    # Step 6: Format output
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
    )

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
    """Route to the appropriate extractor.

    Returns:
        (pages, extractor_name, ocr_page_numbers)
    """
    # Fast mode: PyMuPDF only, skip audit
    if quality == Quality.FAST:
        from pdfmux.extractors.fast import FastExtractor

        ext = FastExtractor()
        pages = list(ext.extract(file_path))
        return pages, ext.name, []

    # High mode: LLM for everything
    if quality == Quality.HIGH:
        pages, name = _try_llm_extractor(file_path)
        return pages, name, []

    # Tables → Docling (unless also graphical)
    if classification.has_tables and not classification.is_graphical:
        pages, name = _try_table_extractor(file_path)
        return pages, name, []

    # Standard → multi-pass
    return _multipass_extract(file_path, classification)


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


# ---------------------------------------------------------------------------
# Multi-pass pipeline
# ---------------------------------------------------------------------------


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
    pages_needing_ocr = audit.bad_pages + audit.empty_pages
    ocr_results: dict[int, str] = {}
    ocr_name = ""

    logger.info(
        f"Multi-pass: {len(pages_needing_ocr)} pages need re-extraction "
        f"(bad={len(audit.bad_pages)}, empty={len(audit.empty_pages)})"
    )

    # Try RapidOCR first
    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ocr = RapidOCRExtractor()
        if ocr.available():
            for page_num in pages_needing_ocr:
                ocr_text = ocr.extract_page(file_path, page_num)
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
                for page_num in pages_needing_ocr:
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

    for fp in fast_pages:
        if fp.page_num in ocr_results:
            ocr_text = ocr_results[fp.page_num]
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
        )

    if output_format == OutputFormat.LLM:
        from pdfmux.formatters.json_fmt import format_llm

        return format_llm(
            text=text,
            source=str(source_path),
            confidence=confidence,
        )

    if output_format == OutputFormat.CSV:
        from pdfmux.formatters.csv_fmt import format_csv

        return format_csv(text)

    raise FormatError(f"Unknown output format: {output_format}")
