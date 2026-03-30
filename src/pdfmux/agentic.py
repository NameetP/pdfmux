"""Agentic multi-pass extraction — verify and re-extract on failure.

The core differentiator: after initial extraction, audit each page's
confidence. Low-confidence pages are re-extracted with a better backend
from the fallback chain. This is what Reducto charges $0.015/page for.

Flow:
    PDF → extract (pass 1) → audit confidence per page
      → for low-confidence pages: re-extract with next backend (pass 2)
      → if still low: escalate to LLM (pass 3)
      → return best result per page
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)

# Confidence threshold for re-extraction
CONFIDENCE_THRESHOLD = float(os.environ.get("PDFMUX_CONFIDENCE_THRESHOLD", "0.70"))

# Maximum re-extraction passes per page
MAX_PASSES = int(os.environ.get("PDFMUX_MAX_PASSES", "3"))


def agentic_improve(
    pages: list[PageResult],
    file_path: Path,
    extractor_name: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_passes: int = MAX_PASSES,
    budget: float | None = None,
) -> tuple[list[PageResult], str, int]:
    """Improve extraction quality by re-extracting low-confidence pages.

    Args:
        pages: Initial extraction results.
        file_path: Path to the PDF.
        extractor_name: Name of the initial extractor.
        confidence_threshold: Pages below this get re-extracted.
        max_passes: Maximum re-extraction attempts per page.
        budget: Maximum cost in USD (None = unlimited).

    Returns:
        (improved_pages, updated_extractor_name, total_passes)
    """
    low_confidence_pages = [
        (i, p) for i, p in enumerate(pages)
        if p.confidence < confidence_threshold
        and p.quality != PageQuality.EMPTY
        and len(p.text.strip()) > 5
        and not p.ocr_applied  # skip pages already re-extracted by multipass
    ]

    if not low_confidence_pages:
        logger.debug("All pages above confidence threshold (%.2f)", confidence_threshold)
        return pages, extractor_name, 1

    logger.info(
        "%d pages below %.0f%% confidence — starting agentic re-extraction",
        len(low_confidence_pages),
        confidence_threshold * 100,
    )

    improved = list(pages)
    total_cost = 0.0
    passes_used = 1
    extractors_used = {extractor_name}

    # Get fallback extractors
    fallbacks = _get_fallback_extractors(extractor_name)

    for pass_num in range(2, max_passes + 1):
        if not low_confidence_pages:
            break

        if not fallbacks:
            logger.debug("No more fallback extractors available")
            break

        fallback_name = fallbacks.pop(0)

        # Budget check
        if budget is not None:
            estimated_cost = len(low_confidence_pages) * _estimate_cost(fallback_name)
            if total_cost + estimated_cost > budget:
                logger.info(
                    "Budget limit reached ($%.2f). Stopping re-extraction.",
                    budget,
                )
                break

        logger.info(
            "Pass %d: re-extracting %d pages with %s",
            pass_num,
            len(low_confidence_pages),
            fallback_name,
        )

        page_nums = [p.page_num for _, p in low_confidence_pages]
        start_time = time.perf_counter()

        try:
            re_extracted = _extract_pages_with(file_path, fallback_name, page_nums)
        except Exception as e:
            logger.warning("Fallback extractor %s failed: %s", fallback_name, e)
            continue

        elapsed = time.perf_counter() - start_time
        logger.info("Pass %d took %.1fs for %d pages", pass_num, elapsed, len(page_nums))

        # Compare and keep the better result per page
        still_low = []
        for idx, original_page in low_confidence_pages:
            re_page = _find_page(re_extracted, original_page.page_num)
            if re_page and re_page.confidence > original_page.confidence:
                improved[idx] = re_page
                logger.debug(
                    "Page %d improved: %.2f → %.2f (%s → %s)",
                    original_page.page_num,
                    original_page.confidence,
                    re_page.confidence,
                    original_page.extractor,
                    re_page.extractor,
                )
            else:
                # Still low — try next fallback
                current = improved[idx]
                if current.confidence < confidence_threshold:
                    still_low.append((idx, current))

        extractors_used.add(fallback_name)
        passes_used = pass_num
        low_confidence_pages = still_low

    # Build extractor name
    if len(extractors_used) > 1:
        name = " + ".join(sorted(extractors_used))
        name += f" ({passes_used} passes)"
    else:
        name = extractor_name

    improved_count = sum(
        1 for orig, imp in zip(pages, improved)
        if imp.confidence > orig.confidence
    )
    if improved_count > 0:
        logger.info(
            "Agentic extraction improved %d/%d pages over %d passes",
            improved_count,
            len(pages),
            passes_used,
        )

    return improved, name, passes_used


def _get_fallback_extractors(current: str) -> list[str]:
    """Get ordered list of fallback extractors to try.

    Priority: OCR → LLM (most expensive last).
    Skip the extractor already used.
    """
    # Ordered by cost: free → cheap → expensive
    chain = []

    try:
        from pdfmux.extractors import available_extractors

        available = {name for name, _ in available_extractors()}
    except Exception:
        available = {"fast"}

    # Build fallback chain based on what's available
    preferred_order = [
        "opendataloader",
        "docling",
        "rapidocr",
        "surya",
        "llm",
    ]

    for name in preferred_order:
        if name in available and name != current and name != "fast":
            chain.append(name)

    return chain


def _extract_pages_with(
    file_path: Path, extractor_name: str, page_nums: list[int]
) -> list[PageResult]:
    """Extract specific pages with a named extractor."""
    if extractor_name == "llm":
        from pdfmux.extractors.llm import LLMExtractor

        ext = LLMExtractor()
        if not ext.available():
            raise RuntimeError("LLM extractor not available")
        return list(ext.extract(file_path, pages=page_nums))

    from pdfmux.extractors import get_extractor

    ext = get_extractor(extractor_name)
    return list(ext.extract(file_path, pages=page_nums))


def _find_page(pages: list[PageResult], page_num: int) -> PageResult | None:
    """Find a page result by page number."""
    for p in pages:
        if p.page_num == page_num:
            return p
    return None


def _estimate_cost(extractor_name: str) -> float:
    """Estimate per-page cost for an extractor."""
    costs = {
        "pymupdf": 0.0,
        "fast": 0.0,
        "opendataloader": 0.0,
        "docling": 0.0,
        "rapidocr": 0.0,
        "surya": 0.0,
        "llm": 0.01,
    }
    return costs.get(extractor_name, 0.0)
