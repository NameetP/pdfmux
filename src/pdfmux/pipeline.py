"""Pipeline — tiered routing logic that picks the best extractor per PDF.

This is the core of Pdfmux. Instead of using one extraction method,
we detect the PDF type and route to the optimal extractor:

  Digital, clean → PyMuPDF (free, 0.01s/page)
  Has tables → Docling (free, 0.3-3s/page)
  Scanned → OCR pipeline (free, 1-5s/page)
  Complex/fallback → Gemini Flash ($0.01-0.05)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    extractor, raw_text = _route_and_extract(file_path, classification, quality)

    # Step 3: Post-process and score confidence
    processed = clean_and_score(raw_text, classification.page_count)

    # Step 4: Format output
    formatted = _format_output(
        processed, output_format, file_path, show_confidence, extractor, classification
    )

    return ConversionResult(
        text=formatted,
        format=output_format,
        confidence=processed.confidence,
        extractor_used=extractor,
        page_count=classification.page_count,
        warnings=processed.warnings,
        classification=classification,
    )


def _route_and_extract(
    file_path: Path,
    classification: PDFClassification,
    quality: str,
) -> tuple[str, str]:
    """Route to the appropriate extractor based on PDF classification.

    Returns:
        Tuple of (extractor_name, raw_text).
    """
    # Fast mode: always use PyMuPDF
    if quality == "fast":
        ext = FastExtractor()
        return ext.name, ext.extract(file_path)

    # High mode: use LLM for everything (if available)
    if quality == "high":
        return _try_llm_extractor(file_path)

    # Standard mode: route based on classification
    if classification.is_digital and not classification.has_tables:
        ext = FastExtractor()
        return ext.name, ext.extract(file_path)

    if classification.has_tables:
        return _try_table_extractor(file_path)

    if classification.is_scanned:
        return _try_ocr_extractor(file_path, classification.scanned_pages)

    if classification.is_mixed:
        return _handle_mixed_pdf(file_path, classification)

    # Default fallback
    ext = FastExtractor()
    return ext.name, ext.extract(file_path)


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
    """Try OCR for scanned pages, fall back to PyMuPDF."""
    try:
        from pdfmux.extractors.ocr import OCRExtractor

        ext = OCRExtractor()
        return ext.name, ext.extract(file_path, pages=pages)
    except ImportError:
        logger.info("Surya OCR not installed, falling back to PyMuPDF")
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


def _handle_mixed_pdf(file_path: Path, classification: PDFClassification) -> tuple[str, str]:
    """Handle mixed PDFs: fast extract digital pages, OCR scanned pages."""
    parts: list[str] = []
    extractor_name = "pymupdf4llm (fast)"

    if classification.digital_pages:
        ext = FastExtractor()
        digital_text = ext.extract(file_path, pages=classification.digital_pages)
        parts.append(digital_text)

    if classification.scanned_pages:
        try:
            from pdfmux.extractors.ocr import OCRExtractor

            ext_ocr = OCRExtractor()
            ocr_text = ext_ocr.extract(file_path, pages=classification.scanned_pages)
            parts.append(ocr_text)
            extractor_name = "pymupdf4llm + surya (mixed)"
        except ImportError:
            logger.info("Skipping scanned pages (OCR not installed)")

    return extractor_name, "\n\n".join(parts)


def _format_output(
    processed: ProcessedResult,
    output_format: str,
    source_path: Path,
    show_confidence: bool,
    extractor: str,
    classification: PDFClassification,
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
        )

    if output_format == "csv":
        from pdfmux.formatters.csv_fmt import format_csv

        return format_csv(processed.text)

    raise ValueError(f"Unknown output format: {output_format}")
