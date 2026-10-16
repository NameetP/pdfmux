"""Per-page progress reporting for long documents (NameetP/pdfmux#15).

``pdfmux convert -v`` on a 5000-page manual used to print nothing for hours.
The extract, audit, and OCR stages now emit throttled INFO lines through the
``pdfmux`` logger, and the fast extractor is windowed so the first line appears
after the first 50 pages rather than after the whole file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
import pytest

from pdfmux.audit import audit_document
from pdfmux.extractors import fast as fast_mod
from pdfmux.extractors.fast import FastExtractor
from pdfmux.parallel import parallel_ocr
from pdfmux.progress import MIN_INTERVAL, SMALL_DOC_PAGES, log_interval, should_log


@pytest.fixture
def long_pdf(tmp_path: Path) -> Path:
    """A 120-page digital PDF: spans three fast-extractor windows (50/50/20)."""
    pdf_path = tmp_path / "long.pdf"
    doc = fitz.open()
    for i in range(120):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"MARKER-{i:04d}\n\nThis is page {i + 1} of a long manual. "
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs.",
            fontsize=11,
        )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class _InstantOCR:
    def extract_page(self, file_path: Path, page_num: int) -> str:
        return f"ocr {page_num}"


def _lines(caplog: pytest.LogCaptureFixture, prefix: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


class TestCadence:
    def test_small_documents_report_every_page(self) -> None:
        assert log_interval(1) == 1
        assert log_interval(SMALL_DOC_PAGES) == 1
        assert all(should_log(n, SMALL_DOC_PAGES) for n in range(1, SMALL_DOC_PAGES + 1))

    def test_large_documents_are_throttled_to_about_two_percent(self) -> None:
        assert log_interval(5000) == 100
        assert log_interval(100) == MIN_INTERVAL  # 100 // 50 = 2, clamped up
        assert sum(should_log(n, 5000) for n in range(1, 5001)) == 50

    def test_last_page_always_reports(self) -> None:
        # 123 // 50 = 2 -> clamped to 5; 123 is not a multiple of 5.
        assert not should_log(122, 123)
        assert should_log(123, 123)
        assert should_log(124, 123)  # over-count still closes the stage

    def test_zero_and_negative_never_report(self) -> None:
        assert not should_log(0, 10)
        assert not should_log(1, 0)
        assert not should_log(-1, 10)

    def test_every_page_below_override(self) -> None:
        assert log_interval(60, every_page_below=100) == 1
        assert log_interval(101, every_page_below=100) == MIN_INTERVAL


class TestFastExtractorWindowing:
    def test_every_page_yielded_once_in_order_with_its_own_text(self, long_pdf: Path) -> None:
        pages = list(FastExtractor().extract(long_pdf))
        assert [p.page_num for p in pages] == list(range(120))
        for p in pages:
            assert f"MARKER-{p.page_num:04d}" in p.text
            assert p.text.count("MARKER-") == 1, "a page must not carry a neighbour's text"

    def test_window_boundaries_are_seamless(self, long_pdf: Path) -> None:
        assert fast_mod.PAGE_WINDOW == 50
        pages = {p.page_num: p for p in FastExtractor().extract(long_pdf)}
        for n in (49, 50, 99, 100, 119):
            assert f"MARKER-{n:04d}" in pages[n].text

    def test_page_subset_is_honoured_and_sorted(self, long_pdf: Path) -> None:
        pages = list(FastExtractor().extract(long_pdf, pages=[75, 3, 119, 3, 500]))
        assert [p.page_num for p in pages] == [3, 75, 119]

    def test_progress_lines_arrive_per_window(
        self, long_pdf: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="pdfmux")
        list(FastExtractor().extract(long_pdf))
        # 120 pages -> interval max(5, 120 // 50) = 5; windows close at 50, 100, 120.
        assert _lines(caplog, "Extracting:") == [
            "Extracting: 50/120 pages",
            "Extracting: 100/120 pages",
            "Extracting: 120/120 pages",
        ]

    def test_short_chunk_list_falls_back_instead_of_dropping_pages(
        self, long_pdf: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        real = fast_mod.pymupdf4llm.to_markdown

        def lossy(*args, **kwargs):
            return real(*args, **kwargs)[:-1]  # drop the last chunk of every window

        monkeypatch.setattr(fast_mod.pymupdf4llm, "to_markdown", lossy)
        caplog.set_level(logging.WARNING, logger="pdfmux")
        pages = list(FastExtractor().extract(long_pdf, pages=list(range(10))))
        assert [p.page_num for p in pages] == list(range(10))
        assert "MARKER-0009" in pages[9].text  # recovered via raw fitz fallback
        assert any(
            "returned 9 chunks for 10 requested pages" in m for m in _lines(caplog, "pymupdf4llm")
        )

    def test_small_document_still_single_call(
        self, multi_page_pdf: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="pdfmux")
        pages = list(FastExtractor().extract(multi_page_pdf))
        assert [p.page_num for p in pages] == list(range(5))
        assert _lines(caplog, "Extracting:") == ["Extracting: 5/5 pages"]


class TestAuditProgress:
    def test_audit_reports_per_window(
        self, long_pdf: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="pdfmux")
        audit = audit_document(long_pdf)
        assert len(audit.pages) == 120
        assert _lines(caplog, "Auditing:") == [
            "Auditing: 50/120 pages",
            "Auditing: 100/120 pages",
            "Auditing: 120/120 pages",
        ]


class TestOCRProgress:
    def test_ocr_reports_every_page_up_to_100(
        self, multi_page_pdf: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="pdfmux")
        results = parallel_ocr(multi_page_pdf, [0, 1, 2, 3], _InstantOCR(), max_workers=2)
        assert len(results) == 4
        assert _lines(caplog, "OCR:") == [f"OCR: {n}/4 pages" for n in (1, 2, 3, 4)]

    def test_ocr_counts_failures_as_done(
        self, multi_page_pdf: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _Failing:
            def extract_page(self, file_path: Path, page_num: int) -> str:
                raise RuntimeError("boom")

        caplog.set_level(logging.INFO, logger="pdfmux")
        results = parallel_ocr(multi_page_pdf, [0, 1], _Failing(), max_workers=2)
        assert all(not r.success for r in results.values())
        assert _lines(caplog, "OCR:") == ["OCR: 1/2 pages", "OCR: 2/2 pages"]

    def test_ocr_throttles_above_100(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="pdfmux")
        results = parallel_ocr(tmp_path / "x.pdf", list(range(250)), _InstantOCR(), max_workers=4)
        assert len(results) == 250
        lines = _lines(caplog, "OCR:")
        assert lines[-1] == "OCR: 250/250 pages"
        assert len(lines) == 50  # every 5 of 250
