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


class _OOMUnderConcurrencyExtractor:
    """Raises an ONNX-style allocation error while another page is in flight."""

    def __init__(
        self, message: str = "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : std::bad_alloc"
    ):
        import threading

        self._lock = threading.Lock()
        self._in_flight = 0
        self._message = message
        self.calls: list[int] = []

    def extract_page(self, file_path: Path, page_num: int) -> str:
        import time

        with self._lock:
            self._in_flight += 1
            concurrent = self._in_flight > 1
            self.calls.append(page_num)
        try:
            time.sleep(0.02)
            if concurrent:
                raise RuntimeError(self._message)
            return f"OCR text for page {page_num}"
        finally:
            with self._lock:
                self._in_flight -= 1


class TestMemoryPressure:
    def test_alloc_failures_are_retried_serially(self, tmp_path: Path) -> None:
        ext = _OOMUnderConcurrencyExtractor()
        results = parallel_ocr(tmp_path / "x.pdf", [0, 1, 2, 3], ext, max_workers=4)
        assert all(r.success for r in results.values())
        assert {pn: r.text for pn, r in results.items()} == {
            pn: f"OCR text for page {pn}" for pn in range(4)
        }

    def test_python_memoryerror_is_retried(self, tmp_path: Path) -> None:
        class _Ext:
            def __init__(self) -> None:
                self.seen: set[int] = set()

            def extract_page(self, file_path: Path, page_num: int) -> str:
                if page_num not in self.seen:
                    self.seen.add(page_num)
                    raise MemoryError()
                return "ok"

        results = parallel_ocr(tmp_path / "x.pdf", [0, 1], _Ext(), max_workers=2)
        assert all(r.success and r.text == "ok" for r in results.values())

    def test_non_memory_errors_are_not_retried(self, tmp_path: Path) -> None:
        ext = _FailingExtractor({1})
        results = parallel_ocr(tmp_path / "x.pdf", [0, 1, 2], ext, max_workers=3)
        assert not results[1].success
        assert "OCR engine error" in (results[1].error or "")

    def test_serial_retry_failure_is_recorded(self, tmp_path: Path) -> None:
        class _AlwaysOOM:
            def extract_page(self, file_path: Path, page_num: int) -> str:
                raise MemoryError("still too big")

        results = parallel_ocr(tmp_path / "x.pdf", [0, 1], _AlwaysOOM(), max_workers=2)
        assert [r.success for r in results.values()] == [False, False]
        assert set(results) == {0, 1}

    def test_single_worker_does_not_retry(self, tmp_path: Path) -> None:
        calls: list[int] = []

        class _Ext:
            def extract_page(self, file_path: Path, page_num: int) -> str:
                calls.append(page_num)
                raise MemoryError()

        results = parallel_ocr(tmp_path / "x.pdf", [0], _Ext(), max_workers=1)
        assert calls == [0]
        assert not results[0].success


class TestOcrWorkersEnv:
    def test_default(self, monkeypatch) -> None:
        from pdfmux.parallel import DEFAULT_WORKERS, ocr_workers

        monkeypatch.delenv("PDFMUX_OCR_WORKERS", raising=False)
        assert ocr_workers() == DEFAULT_WORKERS

    def test_override_drives_pool_size(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PDFMUX_OCR_WORKERS", "1")
        ext = _OOMUnderConcurrencyExtractor()
        results = parallel_ocr(tmp_path / "x.pdf", [0, 1, 2], ext)
        # One worker means no concurrency, so no allocation failures and no retries.
        assert all(r.success for r in results.values())
        assert sorted(ext.calls) == [0, 1, 2]

    def test_invalid_values_fall_back(self, monkeypatch) -> None:
        from pdfmux.parallel import DEFAULT_WORKERS, ocr_workers

        for bad in ("zero", "0", "-2"):
            monkeypatch.setenv("PDFMUX_OCR_WORKERS", bad)
            assert ocr_workers() == DEFAULT_WORKERS
