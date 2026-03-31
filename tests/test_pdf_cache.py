"""Tests for PDF document cache."""

from __future__ import annotations

from pathlib import Path

from pdfmux.pdf_cache import cache_stats, close_all, close_doc, get_doc


class TestPdfCache:
    def test_get_doc_returns_document(self, digital_pdf: Path):
        doc = get_doc(digital_pdf)
        assert doc is not None
        assert len(doc) > 0
        close_all()

    def test_same_doc_returned_twice(self, digital_pdf: Path):
        doc1 = get_doc(digital_pdf)
        doc2 = get_doc(digital_pdf)
        assert doc1 is doc2
        close_all()

    def test_close_doc(self, digital_pdf: Path):
        doc = get_doc(digital_pdf)
        assert not doc.is_closed
        close_doc(digital_pdf)
        assert doc.is_closed

    def test_close_all(self, digital_pdf: Path):
        doc = get_doc(digital_pdf)
        assert not doc.is_closed
        close_all()
        assert doc.is_closed

    def test_cache_stats(self, digital_pdf: Path):
        close_all()
        stats = cache_stats()
        assert stats["cached_docs"] == 0

        get_doc(digital_pdf)
        stats = cache_stats()
        assert stats["cached_docs"] == 1
        assert stats["open_docs"] == 1

        close_all()

    def test_reopen_after_close(self, digital_pdf: Path):
        doc1 = get_doc(digital_pdf)
        close_doc(digital_pdf)

        doc2 = get_doc(digital_pdf)
        assert doc2 is not doc1
        assert not doc2.is_closed
        close_all()
