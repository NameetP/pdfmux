"""Tests for parallel OCR dispatch."""

from __future__ import annotations

from pathlib import Path

from pdfmux.parallel import parallel_ocr


class _FakeExtractor:
    """Fake extractor that returns deterministic text per page."""

    def extract_page(self, file_path: Path, page_num: int) -> str:
        return f"OCR text for page {page_num}"


class _FailingExtractor:
    """Extractor that raises on specific pages."""

    def __init__(self, fail_pages: set[int]) -> None:
        self._fail_pages = fail_pages

    def extract_page(self, file_path: Path, page_num: int) -> str:
        if page_num in self._fail_pages:
            raise RuntimeError(f"OCR engine error on page {page_num}")
        return f"OCR text for page {page_num}"


class TestParallelOCR:
    """Tests for parallel_ocr function."""

    def test_same_results_as_serial(self, tmp_path: Path) -> None:
        """Parallel dispatch produces identical results to serial extraction."""
        fake = _FakeExtractor()
        pdf = tmp_path / "dummy.pdf"
        pdf.touch()

        pages = [0, 1, 2, 3, 4]
        results = parallel_ocr(pdf, pages, fake, max_workers=2)

        assert len(results) == 5
        for pn in pages:
            assert results[pn].page_num == pn
            assert results[pn].text == f"OCR text for page {pn}"
            assert results[pn].success is True
            assert results[pn].runtime_seconds >= 0.0
            assert results[pn].error is None

    def test_failure_isolation(self, tmp_path: Path) -> None:
        """One failing page doesn't crash other pages."""
        failing = _FailingExtractor(fail_pages={2})
        pdf = tmp_path / "dummy.pdf"
        pdf.touch()

        pages = [0, 1, 2, 3]
        results = parallel_ocr(pdf, pages, failing, max_workers=2)

        assert len(results) == 4
        # Good pages succeeded
        assert results[0].success is True
        assert results[1].success is True
        assert results[3].success is True
        # Failed page has error info
        assert results[2].success is False
        assert results[2].text == ""
        assert "OCR engine error" in results[2].error

    def test_worker_clamping(self, tmp_path: Path) -> None:
        """Workers clamped to page count — 2 pages can't use 4 workers."""
        fake = _FakeExtractor()
        pdf = tmp_path / "dummy.pdf"
        pdf.touch()

        pages = [0, 1]
        # Request 4 workers for 2 pages — should still work fine
        results = parallel_ocr(pdf, pages, fake, max_workers=4)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True

    def test_empty_page_list(self, tmp_path: Path) -> None:
        """Empty page list returns empty dict without creating pool."""
        fake = _FakeExtractor()
        pdf = tmp_path / "dummy.pdf"
        pdf.touch()

        results = parallel_ocr(pdf, [], fake)
        assert results == {}
