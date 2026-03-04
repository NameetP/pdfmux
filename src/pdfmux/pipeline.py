"""Pipeline — tiered routing logic that picks the best extractor per PDF.

This is the core of Pdfmux. Instead of using one extraction method,
we detect the PDF type and route to the optimal extractor:

  Standard mode → multi-pass: fast extract → audit → OCR bad pages → merge
  Has tables    → Docling (free, 0.3-3s/page)
  Fast mode     → PyMuPDF only (free, 0.01s/page)
  High mode     → Gemini Flash ($0.01-0.05)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pdfmux.detect import PDFClassification, classify
from pdfmux.extractors.fast import FastExtractor
from pdfmux.formatters.markdown import format_markdown
from pdfmux.postprocess import ProcessedResult, clean_and_score

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Full result of converting a PDF."""

    text: str
    format: str
    confidence: float
    extractor_used: str
    page_count: int
    warnings: list[str]
    classification: PDFClassification
    ocr_pages: list[int] = field(default_factory=list)


def process(
    file_path: str | Path,
    output_format: str = "markdown",
    quality: str = "standard",
    show_confidence: bool = False,
) -> ConversionResult:
    """Process a PDF through the tiered extraction pipeline.

    Args:
        file_path: Path to the PDF file.
        output_format: Output format — "markdown" (default), "json", "csv".
        quality: Quality preset — "fast" (rule-based only), "standard" (auto),
                 "high" (force ML-based).
        show_confidence: Whether to include confidence info in output.

    Returns:
        ConversionResult with extracted text and metadata.
    """
    file_path = Path(file_path)

    # Step 1: Classify the PDF
    classification = classify(file_path)

    # Step 2: Route to the best extractor
    extractor, raw_text, ocr_pages = _route_and_extract(
        file_path, classification, quality
    )

    # Step 3: Post-process and score confidence
    # Multi-pass tracks OCR results directly. For non-multi-pass paths,
    # flag when extraction is likely incomplete (graphical PDF + fast extractor)
    fast_on_graphical = (
        classification.is_graphical
        and "fast" in extractor.lower()
        and not ocr_pages  # multi-pass already handled it
    )
    processed = clean_and_score(
        raw_text,
        classification.page_count,
        extraction_limited=fast_on_graphical,
        graphical_page_count=(
            len(classification.graphical_pages) if fast_on_graphical else 0
        ),
        ocr_page_count=len(ocr_pages),
    )

    # Step 4: Format output
    formatted = _format_output(
        processed, output_format, file_path, show_confidence, extractor, classification,
        ocr_pages=ocr_pages,
    )

    return ConversionResult(
        text=formatted,
        format=output_format,
        confidence=processed.confidence,
        extractor_used=extractor,
        page_count=classification.page_count,
        warnings=processed.warnings,
        classification=classification,
        ocr_pages=ocr_pages,
    )


def _route_and_extract(
    file_path: Path,
    classification: PDFClassification,
    quality: str,
) -> tuple[str, str, list[int]]:
    """Route to the appropriate extractor based on PDF classification.

    Returns:
        Tuple of (extractor_name, raw_text, ocr_pages).
    """
    # Fast mode: always use PyMuPDF, skip audit
    if quality == "fast":
        ext = FastExtractor()
        return ext.name, ext.extract(file_path), []

    # High mode: use LLM for everything (if available)
    if quality == "high":
        name, text = _try_llm_extractor(file_path)
        return name, text, []

    # Tables-detected PDFs route to Docling first — UNLESS also graphical.
    # Graphical PDFs need multi-pass OCR more than table formatting,
    # and Docling can't OCR images anyway.
    if classification.has_tables and not classification.is_graphical:
        name, text = _try_table_extractor(file_path)
        return name, text, []

    # Standard mode — ALL PDFs go through multi-pass.
    # The audit step costs ~0 when all pages are good (fast path returns immediately).
    # This handles digital, graphical, scanned, and mixed PDFs uniformly.
    return _multipass_extract(file_path, classification)


def _try_table_extractor(file_path: Path) -> tuple[str, str]:
    """Try Docling for tables, fall back to PyMuPDF."""
    try:
        from pdfmux.extractors.tables import TableExtractor

        ext = TableExtractor()
        return ext.name, ext.extract(file_path)
    except ImportError:
        logger.info("Docling not installed, falling back to PyMuPDF for tables")
        ext = FastExtractor()
        return ext.name, ext.extract(file_path)


def _try_ocr_extractor(file_path: Path, pages: list[int] | None = None) -> tuple[str, str]:
    """Try OCR for scanned pages — RapidOCR first, then Surya, fall back to PyMuPDF."""
    # Try RapidOCR first (lightweight, ~200MB, Apache 2.0)
    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ext = RapidOCRExtractor()
        return ext.name, ext.extract(file_path, pages=pages)
    except ImportError:
        pass

    # Fall back to Surya (heavy, ~5GB, GPL)
    try:
        from pdfmux.extractors.ocr import OCRExtractor

        ext = OCRExtractor()
        return ext.name, ext.extract(file_path, pages=pages)
    except ImportError:
        logger.info("No OCR installed (RapidOCR or Surya), falling back to PyMuPDF")
        ext = FastExtractor()
        raw = ext.extract(file_path)
        if len(raw.strip()) < 100:
            raise RuntimeError(
                "PDF appears to be scanned/image-based. "
                "Install OCR support with: pip install pdfmux[ocr]"
            )
        return ext.name, raw


def _try_llm_extractor(file_path: Path) -> tuple[str, str]:
    """Try Gemini Flash LLM, fall back to best available."""
    try:
        from pdfmux.extractors.llm import LLMExtractor

        ext = LLMExtractor()
        return ext.name, ext.extract(file_path)
    except (ImportError, RuntimeError) as e:
        logger.info(f"LLM extractor unavailable ({e}), falling back")
        try:
            from pdfmux.extractors.ocr import OCRExtractor

            ext_ocr = OCRExtractor()
            return ext_ocr.name, ext_ocr.extract(file_path)
        except ImportError:
            ext_fast = FastExtractor()
            return ext_fast.name, ext_fast.extract(file_path)


def _multipass_extract(
    file_path: Path,
    classification: PDFClassification,
) -> tuple[str, str, list[int]]:
    """Multi-pass extraction: fast → audit → OCR bad pages → merge.

    Pipeline:
    1. Fast-extract every page (via audit) and score quality
    2. If all pages are good → return fast text immediately (zero overhead)
    3. For bad/empty pages → re-extract with RapidOCR
    4. For pages OCR couldn't recover → try LLM
    5. Merge good fast text + OCR/LLM text in page order

    Returns:
        Tuple of (extractor_name, merged_text, ocr_pages).
    """
    from pdfmux.audit import audit_document

    # Pass 1: Fast extract + audit every page
    audit = audit_document(file_path)

    # Fast path — no bad pages, return immediately (zero overhead)
    if not audit.needs_ocr:
        full_text = "\n\n---\n\n".join(
            p.text for p in audit.pages if p.text.strip()
        )
        return "pymupdf4llm (fast)", full_text, []

    # Pass 2: Re-extract bad/empty pages with OCR
    pages_needing_ocr = audit.bad_pages + audit.empty_pages
    ocr_results: dict[int, str] = {}
    ocr_name = ""

    logger.info(
        f"Multi-pass: {len(pages_needing_ocr)} pages need re-extraction "
        f"(bad={len(audit.bad_pages)}, empty={len(audit.empty_pages)})"
    )

    # Try RapidOCR (lightweight, preferred)
    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ocr = RapidOCRExtractor()
        for page_num in pages_needing_ocr:
            ocr_text = ocr.extract_page(file_path, page_num)
            page_audit = audit.pages[page_num]

            if page_audit.quality == "empty":
                # Any OCR text is an improvement over nothing
                if len(ocr_text.strip()) > 10:
                    ocr_results[page_num] = ocr_text
            elif page_audit.quality == "bad":
                # Only replace if OCR got more than fast extraction
                if len(ocr_text.strip()) > len(page_audit.text.strip()):
                    ocr_results[page_num] = ocr_text

        ocr_name = "rapidocr"
    except ImportError:
        logger.info("RapidOCR not installed, trying legacy OCR")
        # Fall back to legacy Surya OCR
        try:
            from pdfmux.extractors.ocr import OCRExtractor

            ocr_legacy = OCRExtractor()
            for page_num in pages_needing_ocr:
                ocr_text = ocr_legacy.extract(file_path, pages=[page_num])
                page_audit = audit.pages[page_num]

                if page_audit.quality == "empty":
                    if len(ocr_text.strip()) > 10:
                        ocr_results[page_num] = ocr_text
                elif page_audit.quality == "bad":
                    if len(ocr_text.strip()) > len(page_audit.text.strip()):
                        ocr_results[page_num] = ocr_text

            ocr_name = "surya"
        except ImportError:
            logger.info("No OCR installed")

    # Pass 3: Try LLM on pages that OCR couldn't recover
    still_bad = [p for p in pages_needing_ocr if p not in ocr_results]
    if still_bad:
        try:
            from pdfmux.extractors.llm import LLMExtractor

            llm = LLMExtractor()
            for page_num in still_bad:
                llm_text = llm.extract(file_path, pages=[page_num])
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
        except (ImportError, RuntimeError):
            pass

    # Merge in page order: good pages keep fast text, bad/empty pages use OCR
    merged_parts: list[str] = []
    ocr_page_list: list[int] = sorted(ocr_results.keys())

    for page_audit in audit.pages:
        if page_audit.page_num in ocr_results:
            merged_parts.append(ocr_results[page_audit.page_num])
        elif page_audit.text.strip():
            merged_parts.append(page_audit.text)
        # else: empty page with no OCR recovery — skip silently

    merged_text = "\n\n---\n\n".join(merged_parts)

    # Build extractor name for reporting
    n_ocr = len(ocr_page_list)
    n_unrecovered = len(still_bad) - sum(
        1 for p in still_bad if p in ocr_results
    )

    if n_ocr > 0:
        name = f"pymupdf4llm + {ocr_name} ({n_ocr} pages re-extracted)"
    else:
        name = "pymupdf4llm (fast)"

    if n_unrecovered > 0:
        logger.warning(
            f"Multi-pass: {n_unrecovered} pages could not be recovered. "
            f"Install pdfmux[ocr] for better results."
        )

    logger.info(
        f"Multi-pass complete: {len(audit.good_pages)} good, "
        f"{n_ocr} re-extracted, {n_unrecovered} unrecovered"
    )

    return name, merged_text, ocr_page_list


def _format_output(
    processed: ProcessedResult,
    output_format: str,
    source_path: Path,
    show_confidence: bool,
    extractor: str,
    classification: PDFClassification,
    *,
    ocr_pages: list[int] | None = None,
) -> str:
    """Format the processed text into the requested output format."""
    if output_format == "markdown":
        text = format_markdown(processed.text, source=str(source_path))
        if show_confidence:
            confidence_note = (
                f"\n\n---\n"
                f"*Conversion confidence: {processed.confidence:.0%} "
                f"({processed.page_count} pages)*"
            )
            if processed.warnings:
                confidence_note += "\n" + "\n".join(f"- ⚠ {w}" for w in processed.warnings)
            text += confidence_note
        return text

    if output_format == "json":
        from pdfmux.formatters.json_fmt import format_json

        return format_json(
            text=processed.text,
            source=str(source_path),
            page_count=processed.page_count,
            confidence=processed.confidence,
            extractor=extractor,
            warnings=processed.warnings,
            ocr_pages=ocr_pages,
        )

    if output_format == "llm":
        from pdfmux.formatters.json_fmt import format_llm

        return format_llm(
            text=processed.text,
            source=str(source_path),
            confidence=processed.confidence,
        )

    if output_format == "csv":
        from pdfmux.formatters.csv_fmt import format_csv

        return format_csv(processed.text)

    raise ValueError(f"Unknown output format: {output_format}")
