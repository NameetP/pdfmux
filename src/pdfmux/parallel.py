"""Parallel page processing for OCR re-extraction.

Uses ThreadPoolExecutor because OCR is CPU-bound via ONNX runtime,
which releases the GIL during inference. Thread overhead is lower
than process overhead for our workload (shared file handle, small payloads).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pdfmux.progress import should_log

logger = logging.getLogger(__name__)

# Report every OCR'd page up to this many; beyond it, throttle (see progress.py).
OCR_EVERY_PAGE_BELOW = 100

# Default: 4 threads. ONNX runtime uses internal threading per inference,
# so we don't want too many concurrent inferences competing for CPU.
# 4 is the sweet spot on 4-8 core machines.
DEFAULT_WORKERS = 4


@dataclass(frozen=True)
class PageOCRResult:
    """Result of OCR'ing a single page."""

    page_num: int
    text: str
    runtime_seconds: float
    success: bool
    error: str | None = None


def parallel_ocr(
    file_path: Path,
    page_nums: list[int],
    extractor,  # duck-typed: has extract_page(path, page_num) -> str
    *,
    max_workers: int = DEFAULT_WORKERS,
) -> dict[int, PageOCRResult]:
    """OCR multiple pages in parallel using ThreadPoolExecutor.

    Args:
        file_path: Path to PDF.
        page_nums: 0-indexed page numbers to OCR.
        extractor: Object with extract_page(file_path, page_num) -> str.
        max_workers: Thread pool size.

    Returns:
        Dict mapping page_num -> PageOCRResult.
    """
    results: dict[int, PageOCRResult] = {}

    if not page_nums:
        return results

    # Clamp workers to page count (no point having 4 threads for 2 pages)
    workers = min(max_workers, len(page_nums))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_page = {
            pool.submit(_ocr_one_page, extractor, file_path, pn): pn for pn in page_nums
        }

        total = len(page_nums)
        done = 0
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                results[page_num] = result
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num}: {e}")
                results[page_num] = PageOCRResult(
                    page_num=page_num,
                    text="",
                    runtime_seconds=0.0,
                    success=False,
                    error=str(e),
                )
            done += 1
            # OCR is the slow stage (seconds per page), so report every page
            # up to 100 pages and ~2% steps beyond that.
            if should_log(done, total, every_page_below=OCR_EVERY_PAGE_BELOW):
                logger.info("OCR: %d/%d pages", done, total)

    return results


def _ocr_one_page(extractor, file_path: Path, page_num: int) -> PageOCRResult:
    """OCR a single page. Runs inside a thread."""
    start = time.perf_counter()
    text = extractor.extract_page(file_path, page_num)
    elapsed = time.perf_counter() - start
    return PageOCRResult(
        page_num=page_num,
        text=text,
        runtime_seconds=elapsed,
        success=True,
    )
