"""Streaming extraction — yield each page as it lands.

For large documents (100+ pages) the standard pipeline blocks until every
page is processed and merged. ``process_streaming`` instead emits a
sequence of :class:`StreamEvent` objects:

    classified  → page_count, page_types     (always first)
    page        → page_num, text, confidence, quality, extractor (one per page)
    warning     → free-form message          (zero or more)
    complete    → total_confidence, ocr_pages, extractor_used, warnings (last)

The streaming version reuses extraction logic from :mod:`pdfmux.audit`,
:mod:`pdfmux.extractors.fast`, and the OCR/LLM extractors that the
multi-pass pipeline uses, but yields per-page rather than building a
single ``ConversionResult``.

Example::

    for ev in process_streaming("report.pdf"):
        if ev.type == "page":
            print(ev.data["page_num"], ev.data["text"][:80])

The CLI wraps this generator and prints one JSON object per line
(``pdfmux stream report.pdf``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pdfmux.audit import audit_document, compute_document_confidence, score_page
from pdfmux.detect import PDFClassification, classify
from pdfmux.errors import FileError
from pdfmux.types import PageQuality, PageResult, Quality

logger = logging.getLogger(__name__)


StreamEventType = Literal["classified", "page", "warning", "complete"]


@dataclass(frozen=True)
class StreamEvent:
    """One emission from :func:`process_streaming`.

    The payload schema depends on ``type``:

    * ``classified`` — ``{"page_count": int, "page_types": list[str]}``
    * ``page`` — ``{"page_num": int, "text": str, "confidence": float,
      "quality": str, "extractor": str, "ocr_applied": bool}``
    * ``warning`` — ``{"message": str}``
    * ``complete`` — ``{"total_confidence": float, "ocr_pages": list[int],
      "extractor_used": str, "warnings": list[str], "page_count": int}``
    """

    type: StreamEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this event."""
        return {"type": self.type, "data": self.data}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


def _classification_to_page_types(c: PDFClassification) -> list[str]:
    """Return one label per page in document order."""
    types: list[str] = []
    digital = set(c.digital_pages or [])
    scanned = set(c.scanned_pages or [])
    graphical = set(c.graphical_pages or [])
    empty = set(getattr(c, "empty_pages", []) or [])

    for i in range(c.page_count):
        if i in scanned:
            types.append("scanned")
        elif i in graphical:
            types.append("graphical")
        elif i in empty:
            types.append("empty")
        elif i in digital:
            types.append("digital")
        else:
            types.append("unknown")
    return types


def _ocr_page(file_path: Path, page_num: int, current_text: str) -> tuple[str, str] | None:
    """Re-extract a single page with OCR/LLM fall-throughs.

    Returns ``(text, extractor_name)`` on success, ``None`` if no extractor
    produced better text than ``current_text``.
    """
    # 1) RapidOCR
    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ocr = RapidOCRExtractor()
        if ocr.available():
            results = list(ocr.extract(file_path, pages=[page_num]))
            if results and results[0].text.strip():
                if len(results[0].text.strip()) > len(current_text.strip()):
                    return results[0].text, "rapidocr"
    except Exception as e:
        logger.debug("RapidOCR streaming page %d failed: %s", page_num, e)

    # 2) Surya / legacy OCR
    try:
        from pdfmux.extractors.ocr import OCRExtractor

        ocr_legacy = OCRExtractor()
        if ocr_legacy.available():
            results = list(ocr_legacy.extract(file_path, pages=[page_num]))
            if results and results[0].text.strip():
                if len(results[0].text.strip()) > len(current_text.strip()):
                    return results[0].text, "surya"
    except Exception as e:
        logger.debug("Surya streaming page %d failed: %s", page_num, e)

    # 3) LLM Vision
    try:
        from pdfmux.extractors.llm import LLMExtractor

        llm = LLMExtractor()
        if llm.available():
            results = list(llm.extract(file_path, pages=[page_num]))
            if results and results[0].text.strip():
                if len(results[0].text.strip()) > len(current_text.strip()):
                    return results[0].text, "gemini"
    except Exception as e:
        logger.debug("LLM streaming page %d failed: %s", page_num, e)

    return None


def process_streaming(
    file_path: str | Path,
    quality: str = "standard",
    *,
    use_cache: bool = False,
) -> Iterator[StreamEvent]:
    """Stream extraction events for a PDF, page-by-page.

    Args:
        file_path: Path to the PDF file.
        quality: ``"fast"`` skips OCR re-extraction; ``"standard"`` and
            ``"high"`` re-extract bad/empty pages just like the regular
            pipeline.
        use_cache: When True, consult the smart result cache for a
            ``markdown`` extraction first; on hit, emit a single
            ``classified`` event followed by one ``page`` event per
            paragraph-joined chunk and a ``complete`` event. Default
            False — streaming consumers usually want raw events.

    Yields:
        :class:`StreamEvent` instances. The first is always
        ``classified`` and the last is always ``complete``.

    Raises:
        FileError: If the file is missing or invalid.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileError(f"PDF not found: {file_path}")

    try:
        qual = Quality(quality)
    except ValueError:
        qual = Quality.STANDARD

    # Optional cache short-circuit (markdown only — streaming consumers
    # don't typically request structured output).
    if use_cache:
        try:
            from pdfmux.result_cache import get_default_cache

            cache = get_default_cache()
            if cache.enabled:
                cached = cache.get(file_path, quality, "markdown", None)
                if cached is not None:
                    yield from _events_from_cached(cached)
                    return
        except Exception as e:  # pragma: no cover
            logger.debug("Streaming cache lookup skipped: %s", e)

    # 1. Classify (synchronous, fast).
    classification = classify(file_path)
    page_types = _classification_to_page_types(classification)
    yield StreamEvent(
        type="classified",
        data={
            "page_count": classification.page_count,
            "page_types": page_types,
        },
    )

    # 2. Audit + emit fast-pass pages.
    audit = audit_document(file_path)

    page_results: dict[int, PageResult] = {}
    ocr_pages: list[int] = []
    warnings: list[str] = []
    extractors_used: set[str] = set()
    extractors_used.add("pymupdf4llm")

    for pa in audit.pages:
        confidence = score_page(pa.text, pa.image_count)
        pr = PageResult(
            page_num=pa.page_num,
            text=pa.text,
            confidence=confidence,
            quality=PageQuality(pa.quality),
            extractor="pymupdf4llm",
            image_count=pa.image_count,
        )
        page_results[pa.page_num] = pr

        # Yield this page IF it's good — bad/empty pages will be re-emitted
        # below after OCR. Yielding good pages immediately gives consumers
        # responsive feedback while OCR runs.
        if pa.quality == "good" or qual == Quality.FAST:
            yield StreamEvent(
                type="page",
                data={
                    "page_num": pa.page_num,
                    "text": pr.text,
                    "confidence": pr.confidence,
                    "quality": pr.quality.value,
                    "extractor": pr.extractor,
                    "ocr_applied": pr.ocr_applied,
                    "image_count": pr.image_count,
                },
            )

    # 3. Re-extract bad/empty pages (skipped in fast mode).
    if qual != Quality.FAST and audit.needs_ocr:
        bad_or_empty = sorted(set(audit.bad_pages + audit.empty_pages))

        # Apply OCR budget (mirrors the legacy pipeline).
        budget_ratio = float(os.environ.get("PDFMUX_OCR_BUDGET", "0.30"))
        graphical_ratio = (
            len(classification.graphical_pages) / classification.page_count
            if classification.page_count
            else 0
        )
        if graphical_ratio >= 0.50:
            budget_ratio = 1.0
        elif graphical_ratio > 0.25:
            budget_ratio = min(1.0, graphical_ratio + 0.10)
        max_ocr_pages = max(1, int(classification.page_count * budget_ratio))

        if len(bad_or_empty) > max_ocr_pages:
            skipped = len(bad_or_empty) - max_ocr_pages
            msg = (
                f"{skipped} pages skipped due to OCR budget. "
                "Use --quality high to process all pages."
            )
            warnings.append(msg)
            yield StreamEvent(type="warning", data={"message": msg})

            prioritized = sorted(
                bad_or_empty,
                key=lambda pn: (0 if audit.pages[pn].quality == "bad" else 1, pn),
            )
            bad_or_empty = prioritized[:max_ocr_pages]

        for page_num in bad_or_empty:
            pa = audit.pages[page_num]
            ocr_attempt = _ocr_page(file_path, page_num, pa.text)
            if ocr_attempt is None:
                # Emit the original (low-confidence) page so consumers always
                # see one event per page.
                pr = page_results[page_num]
                yield StreamEvent(
                    type="page",
                    data={
                        "page_num": pr.page_num,
                        "text": pr.text,
                        "confidence": pr.confidence,
                        "quality": pr.quality.value,
                        "extractor": pr.extractor,
                        "ocr_applied": False,
                        "image_count": pr.image_count,
                    },
                )
                continue

            ocr_text, ocr_name = ocr_attempt
            extractors_used.add(ocr_name)
            new_pr = PageResult(
                page_num=page_num,
                text=ocr_text,
                confidence=score_page(ocr_text, pa.image_count),
                quality=PageQuality.GOOD,
                extractor=ocr_name,
                image_count=pa.image_count,
                ocr_applied=True,
            )
            page_results[page_num] = new_pr
            ocr_pages.append(page_num)
            yield StreamEvent(
                type="page",
                data={
                    "page_num": new_pr.page_num,
                    "text": new_pr.text,
                    "confidence": new_pr.confidence,
                    "quality": new_pr.quality.value,
                    "extractor": new_pr.extractor,
                    "ocr_applied": True,
                    "image_count": new_pr.image_count,
                },
            )

    # 4. Compute totals + final event.
    pages_in_order = [
        page_results[i] for i in range(classification.page_count) if i in page_results
    ]
    total_confidence, conf_warnings = compute_document_confidence(
        pages_in_order,
        ocr_page_count=len(ocr_pages),
        unrecovered_count=0,
    )
    for w in conf_warnings:
        if w not in warnings:
            warnings.append(w)
            yield StreamEvent(type="warning", data={"message": w})

    extractor_used = " + ".join(sorted(extractors_used))
    yield StreamEvent(
        type="complete",
        data={
            "total_confidence": total_confidence,
            "ocr_pages": sorted(ocr_pages),
            "extractor_used": extractor_used,
            "warnings": warnings,
            "page_count": classification.page_count,
        },
    )


def _events_from_cached(cached: Any) -> Iterator[StreamEvent]:
    """Replay a cached ConversionResult as a stream of events."""
    page_count = getattr(cached, "page_count", 0)
    classification = getattr(cached, "classification", None)
    if classification is not None:
        page_types = _classification_to_page_types(classification)
    else:
        page_types = ["unknown"] * page_count

    yield StreamEvent(
        type="classified",
        data={"page_count": page_count, "page_types": page_types},
    )

    # Cached results store merged text; split on form-feed-like markers if
    # any, otherwise emit a single synthetic page so consumers still get a
    # complete envelope.
    text = getattr(cached, "text", "")
    yield StreamEvent(
        type="page",
        data={
            "page_num": 0,
            "text": text,
            "confidence": getattr(cached, "confidence", 0.0),
            "quality": "good",
            "extractor": getattr(cached, "extractor_used", "cache"),
            "ocr_applied": bool(getattr(cached, "ocr_pages", []) or []),
            "image_count": 0,
            "from_cache": True,
        },
    )

    yield StreamEvent(
        type="complete",
        data={
            "total_confidence": getattr(cached, "confidence", 0.0),
            "ocr_pages": list(getattr(cached, "ocr_pages", []) or []),
            "extractor_used": getattr(cached, "extractor_used", "cache"),
            "warnings": list(getattr(cached, "warnings", []) or []),
            "page_count": page_count,
            "from_cache": True,
        },
    )
