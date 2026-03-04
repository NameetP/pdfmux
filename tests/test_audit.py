"""Tests for the per-page quality audit module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfmux.audit import (
    EMPTY_TEXT_THRESHOLD,
    GOOD_TEXT_THRESHOLD,
    DocumentAudit,
    PageAudit,
    _classify_page,
    audit_document,
)


class TestClassifyPage:
    """Tests for the page classification logic."""

    def test_good_page_high_text(self) -> None:
        """Page with plenty of text is 'good'."""
        quality, reason = _classify_page(500, 3)
        assert quality == "good"

    def test_good_page_threshold_boundary(self) -> None:
        """Page with exactly GOOD_TEXT_THRESHOLD chars is 'good'."""
        quality, reason = _classify_page(GOOD_TEXT_THRESHOLD, 5)
        assert quality == "good"

    def test_good_page_no_images(self) -> None:
        """Page with some text but no images is 'good' (nothing to OCR)."""
        quality, reason = _classify_page(80, 0)
        assert quality == "good"

    def test_bad_page_low_text_with_images(self) -> None:
        """Page with low text + images is 'bad' (text likely in images)."""
        quality, reason = _classify_page(100, 3)
        assert quality == "bad"

    def test_bad_page_minimal_text_with_images(self) -> None:
        """Page just above empty threshold with images is 'bad'."""
        quality, reason = _classify_page(30, 1)
        assert quality == "bad"

    def test_empty_page_no_text(self) -> None:
        """Page with virtually no text is 'empty'."""
        quality, reason = _classify_page(5, 0)
        assert quality == "empty"

    def test_empty_page_no_text_with_images(self) -> None:
        """Page with no text but images is 'empty'."""
        quality, reason = _classify_page(10, 3)
        assert quality == "empty"

    def test_empty_threshold_boundary(self) -> None:
        """Page at the empty threshold boundary."""
        # Just below threshold = empty
        quality, _ = _classify_page(EMPTY_TEXT_THRESHOLD - 1, 0)
        assert quality == "empty"

    def test_edge_case_minimal_no_images(self) -> None:
        """Edge case: between empty and minimal threshold, no images."""
        quality, _ = _classify_page(25, 0)
        assert quality == "good"  # No images to OCR


class TestDocumentAudit:
    """Tests for DocumentAudit dataclass properties."""

    def test_good_pages_property(self) -> None:
        """good_pages returns page numbers of good pages."""
        pages = [
            PageAudit(0, "text", 300, 0, "good", "ok"),
            PageAudit(1, "", 5, 2, "empty", "no text"),
            PageAudit(2, "text", 400, 1, "good", "ok"),
        ]
        audit = DocumentAudit(pages=pages, total_pages=3)
        assert audit.good_pages == [0, 2]

    def test_bad_pages_property(self) -> None:
        """bad_pages returns page numbers of bad pages."""
        pages = [
            PageAudit(0, "x" * 100, 100, 3, "bad", "low text"),
            PageAudit(1, "x" * 300, 300, 0, "good", "ok"),
        ]
        audit = DocumentAudit(pages=pages, total_pages=2)
        assert audit.bad_pages == [0]

    def test_empty_pages_property(self) -> None:
        """empty_pages returns page numbers of empty pages."""
        pages = [
            PageAudit(0, "", 0, 2, "empty", "no text"),
            PageAudit(1, "x" * 300, 300, 0, "good", "ok"),
        ]
        audit = DocumentAudit(pages=pages, total_pages=2)
        assert audit.empty_pages == [0]

    def test_needs_ocr_true(self) -> None:
        """needs_ocr is True when bad or empty pages exist."""
        pages = [
            PageAudit(0, "x" * 300, 300, 0, "good", "ok"),
            PageAudit(1, "", 5, 2, "empty", "no text"),
        ]
        audit = DocumentAudit(pages=pages, total_pages=2)
        assert audit.needs_ocr is True

    def test_needs_ocr_false(self) -> None:
        """needs_ocr is False when all pages are good."""
        pages = [
            PageAudit(0, "x" * 300, 300, 0, "good", "ok"),
            PageAudit(1, "x" * 400, 400, 1, "good", "ok"),
        ]
        audit = DocumentAudit(pages=pages, total_pages=2)
        assert audit.needs_ocr is False


class TestAuditDocument:
    """Tests for the full audit_document function."""

    def test_digital_pdf_all_good(self, digital_pdf: Path) -> None:
        """Digital PDF should have all pages classified as good."""
        audit = audit_document(digital_pdf)
        assert audit.total_pages == 2
        assert len(audit.good_pages) == 2
        assert audit.needs_ocr is False

    def test_empty_pdf(self, empty_pdf: Path) -> None:
        """Empty PDF should have empty pages."""
        audit = audit_document(empty_pdf)
        assert audit.total_pages == 1
        assert len(audit.empty_pages) == 1
        assert audit.needs_ocr is True

    def test_multi_page_pdf(self, multi_page_pdf: Path) -> None:
        """Multi-page digital PDF should have all pages good."""
        audit = audit_document(multi_page_pdf)
        assert audit.total_pages == 5
        assert len(audit.good_pages) == 5
        assert audit.needs_ocr is False

    def test_page_audit_frozen(self, digital_pdf: Path) -> None:
        """PageAudit should be immutable (frozen dataclass)."""
        audit = audit_document(digital_pdf)
        with pytest.raises(AttributeError):
            audit.pages[0].quality = "bad"  # type: ignore[misc]
