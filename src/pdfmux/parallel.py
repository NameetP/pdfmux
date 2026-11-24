"""Parallel page processing for OCR re-extraction.

Uses ThreadPoolExecutor because OCR is CPU-bound via ONNX runtime,
which releases the GIL during inference. Thread overhead is lower
than process overhead for our workload (shared file handle, small payloads).
"""

from __future__ import annotations

import logging
import os
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
_WORKERS_ENV = "PDFMUX_OCR_WORKERS"

# One RapidOCR engine plateaus at ~1.3 GB on a 2000 px page, and each extra
# concurrent inference adds activation memory on top. Under a tight memory
# limit a page can fail with an allocation error only because its neighbours
# were running at the same time. Those pages are retried one at a time.
_ALLOC_MARKERS = ("bad_alloc", "failed to allocate", "out of memory", "cannot allocate")


def ocr_workers() -> int:
    """Concurrent OCR pages; ``PDFMUX_OCR_WORKERS`` overrides the default of 4.

    Set it to 1 on memory-constrained hosts: peak drops from ~1.6 GB to ~1.2 GB.
    """
    raw = os.environ.get(_WORKERS_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", _WORKERS_ENV, raw)
        else:
            if value >= 1:
                return value
            logger.warning("Ignoring %s=%r (must be >= 1)", _WORKERS_ENV, raw)
    return DEFAULT_WORKERS


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _ALLOC_MARKERS)


@dataclass(frozen=True)
class PageOCRResult:
    """Result of OCR'ing a single page."""

    page_num: int
    text: str
    runtime_seconds: float
    success: bool
    error: str | None = None
    # OCRLine entries (text, score, bbox) when the extractor reports them.
    lines: tuple = ()


def parallel_ocr(
    file_path: Path,
    page_nums: list[int],
    extractor,  # duck-typed: has extract_page(path, page_num) -> str
    *,
    max_workers: int | None = None,
) -> dict[int, PageOCRResult]:
    """OCR multiple pages in parallel using ThreadPoolExecutor.

    Args:
        file_path: Path to PDF.
        page_nums: 0-indexed page numbers to OCR.
        extractor: Object with extract_page(file_path, page_num) -> str.
        max_workers: Thread pool size (default: ``ocr_workers()``).

    Returns:
        Dict mapping page_num -> PageOCRResult.
    """
    results: dict[int, PageOCRResult] = {}

    if not page_nums:
        return results

    # Clamp workers to page count (no point having 4 threads for 2 pages)
    workers = min(max_workers if max_workers is not None else ocr_workers(), len(page_nums))
    retry_serial: list[int] = []

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
                if workers > 1 and _is_memory_error(e):
                    retry_serial.append(page_num)
                    done += 1
                    continue
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

    if retry_serial:
        logger.warning(
            "OCR ran out of memory on %d page(s) with %d workers; retrying them one at a time "
            "(set %s=1 to avoid this)",
            len(retry_serial),
            workers,
            _WORKERS_ENV,
        )
        for page_num in sorted(retry_serial):
            try:
                results[page_num] = _ocr_one_page(extractor, file_path, page_num)
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num} on serial retry: {e}")
                results[page_num] = PageOCRResult(
                    page_num=page_num,
                    text="",
                    runtime_seconds=0.0,
                    success=False,
                    error=str(e),
                )

    return results


def _ocr_one_page(extractor, file_path: Path, page_num: int) -> PageOCRResult:
    """OCR a single page. Runs inside a thread."""
    start = time.perf_counter()
    lines: tuple = ()
    if hasattr(extractor, "extract_page_lines"):
        lines = tuple(extractor.extract_page_lines(file_path, page_num))
        text = "\n".join(ln.text if hasattr(ln, "text") else ln[0] for ln in lines)
    else:
        text = extractor.extract_page(file_path, page_num)
    elapsed = time.perf_counter() - start
    return PageOCRResult(
        page_num=page_num,
        text=text,
        runtime_seconds=elapsed,
        success=True,
        lines=lines,
    )
