"""Tests for PDF type detection."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdfmux.detect import classify
from pdfmux.errors import FileError


def test_classify_digital_pdf(digital_pdf: Path) -> None:
    """Digital PDFs should be classified as digital."""
    result = classify(digital_pdf)
    assert result.is_digital
    assert not result.is_scanned
    assert result.page_count == 2
    assert result.confidence > 0.5


def test_classify_empty_pdf(empty_pdf: Path) -> None:
    """Empty PDFs should still be classified without errors."""
    result = classify(empty_pdf)
    assert result.page_count == 1
    # Empty pages are treated as digital (no images)
    assert result.is_digital


def test_classify_multi_page(multi_page_pdf: Path) -> None:
    """Multi-page PDFs should report correct page count."""
    result = classify(multi_page_pdf)
    assert result.page_count == 5
    assert result.is_digital


def test_classify_nonexistent_file() -> None:
    """Should raise FileError for missing files."""
    with pytest.raises(FileError):
        classify("/nonexistent/file.pdf")


def test_classify_non_pdf(tmp_path: Path) -> None:
    """Should raise FileError for non-PDF files."""
    txt = tmp_path / "test.txt"
    txt.write_text("not a pdf")
    with pytest.raises(FileError):
        classify(txt)


# ---------------------------------------------------------------------------
# Regression tests for classifier bugs B1-B4
# (found by the 2026-09-02 pdf-inspector head-to-head eval)
# ---------------------------------------------------------------------------

_BODY = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump over the fence."
)


def _scan_page_png(label: str) -> bytes:
    """Render a one-page text doc to PNG bytes — a synthetic scan image."""
    src = fitz.open()
    page = src.new_page()
    page.insert_text((72, 72), f"{label}\n\n{_BODY}", fontsize=11)
    png = page.get_pixmap(dpi=100).tobytes("png")
    src.close()
    return png


def _figure_png() -> bytes:
    """A small colored-figure PNG for slide-deck style pages."""
    src = fitz.open()
    page = src.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(10, 10, 290, 190), color=(0.2, 0.5, 0.9), fill=(0.2, 0.5, 0.9))
    png = page.get_pixmap(dpi=96).tobytes("png")
    src.close()
    return png


def test_classify_mixed_text_and_scan_is_mixed_not_graphical(tmp_path: Path) -> None:
    """B1: a genuinely mixed text+raster doc must be mixed, not graphical.

    Bare scan pages (full-page raster, no text layer) also satisfied the
    graphical page rule, so `is_graphical` fired on every mixed doc and the
    higher-precedence graphical route made the mixed route unreachable.
    """
    pdf_path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        if i % 2 == 0:
            page.insert_text((72, 72), f"Chapter {i + 1}\n\n{_BODY}", fontsize=11)
        else:
            page.insert_image(page.rect, stream=_scan_page_png(f"Scan {i + 1}"))
    doc.save(str(pdf_path))
    doc.close()

    result = classify(pdf_path)
    assert result.is_mixed
    assert not result.is_graphical
    assert not result.is_scanned
    assert result.scanned_pages == [1, 3]


def test_classify_short_caption_deck_is_graphical_not_scanned(tmp_path: Path) -> None:
    """B2: image-heavy slides with short captions must not read as scanned.

    Pages with images and captions under 50 chars fell into the scanned
    bucket, so slide decks classified as scanned documents.
    """
    pdf_path = tmp_path / "deck.pdf"
    figure = _figure_png()
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        for k in range(3):
            rect = fitz.Rect(60 + k * 170, 120, 200 + k * 170, 260)
            page.insert_image(rect, stream=figure)
        page.insert_text((72, 80), f"Figure deck slide {i + 1} — quarterly chart", fontsize=14)
    doc.save(str(pdf_path))
    doc.close()

    result = classify(pdf_path)
    assert result.is_graphical
    assert not result.is_scanned


def test_classify_bare_scan_still_scanned(tmp_path: Path) -> None:
    """B1/B2 guard: a pure scan (raster pages, no text layer) stays scanned."""
    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_image(page.rect, stream=_scan_page_png(f"Page {i + 1}"))
    doc.save(str(pdf_path))
    doc.close()

    result = classify(pdf_path)
    assert result.is_scanned
    assert not result.is_graphical
    assert result.scanned_pages == [0, 1, 2]


def test_classify_encrypted_pdf_raises_file_error(tmp_path: Path) -> None:
    """B3: an encrypted PDF must raise the documented FileError.

    Page access on a locked document leaked a bare ValueError instead of
    classify()'s FileError contract.
    """
    pdf_path = tmp_path / "locked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), _BODY, fontsize=11)
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()

    with pytest.raises(FileError) as exc_info:
        classify(pdf_path)
    assert exc_info.value.code == "PDF_ENCRYPTED"


def test_classify_html_as_pdf_raises_file_error(tmp_path: Path) -> None:
    """B4: HTML served with a .pdf extension must be rejected at classify.

    It previously passed classification as a digital document and relied on
    the downstream audit layer to catch it.
    """
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_text(
        "<html><head><title>Not a PDF</title></head>"
        "<body><h1>This is HTML</h1><p>Pretending to be a PDF.</p></body></html>",
        encoding="utf-8",
    )

    with pytest.raises(FileError) as exc_info:
        classify(pdf_path)
    assert exc_info.value.code == "PDF_INVALID"
