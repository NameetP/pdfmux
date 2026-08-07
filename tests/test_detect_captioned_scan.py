"""Regression tests for captioned-scan detection.

A scanned page is image-only in substance, but real-world scans routinely carry
a printed letterhead, stamp, reference number or "Page 1 of 4" footer rendered
as *real* text. Classifying on character count alone (`text_len > 50`) called
those pages digital, so they were never routed to OCR and extraction returned
the caption where the document body should have been — a silent failure, and
one that installing `pdfmux[ocr]` does not fix because the page never reaches
the OCR stage.

The discriminator is text-*area* coverage, not character count. Character count
cannot separate the two cases that matter: a scan with a 1000-character
letterhead and an OCR'd "searchable PDF" with a 1000-character text layer are
identical by length. They differ in whether the text spans the page.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdfmux.detect import classify

# Enough body text that the rendered page is unambiguously a full page of prose.
_BODY = (
    "Thermal decomposition of the polymer matrix begins at 240 C. "
    "Tensile strength retention after 1000 hours at 85 C and 85% RH "
    "exceeds 92% of the initial value measured per ASTM D638. "
)


def _text_page(path: Path) -> Path:
    """A single page of native text — the source we rasterize from."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(56, 56, 540, 760), _BODY * 8, fontsize=11, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def _scan_with_caption(src: Path, dst: Path, caption_chars: int) -> Path:
    """Rasterize `src`, then overlay a real text caption of the given length.

    This is what a scanned document with a printed letterhead looks like: the
    body is locked in the image, and a small band of genuine text sits on top.
    """
    src_doc = fitz.open(src)
    out = fitz.open()
    for page in src_doc:
        pix = page.get_pixmap(dpi=150)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(page.rect, pixmap=pix)
        if caption_chars:
            caption = ("SCANNED DOC REF 8841-B BATCH 44 " * 120)[:caption_chars]
            new.insert_textbox(fitz.Rect(40, 8, 570, 300), caption, fontsize=6, fontname="helv")
    out.save(str(dst))
    out.close()
    src_doc.close()
    return dst


def _searchable_pdf(src: Path, dst: Path) -> Path:
    """Rasterize `src` and re-lay its text invisibly on top.

    This is an OCR'd "searchable PDF" — image-backed, but with a genuine
    page-spanning text layer. It must stay digital: OCR would be wasted work.
    """
    src_doc = fitz.open(src)
    out = fitz.open()
    for page in src_doc:
        pix = page.get_pixmap(dpi=150)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(page.rect, pixmap=pix)
        for block in page.get_text("blocks"):
            if not block[4].strip():
                continue
            new.insert_textbox(
                fitz.Rect(block[:4]),
                block[4],
                fontsize=9,
                fontname="helv",
                render_mode=3,  # invisible — the OCR text-layer convention
            )
    out.save(str(dst))
    out.close()
    src_doc.close()
    return dst


@pytest.mark.parametrize("caption_chars", [0, 20, 40, 50, 60, 80, 120, 300, 600, 1200])
def test_captioned_scan_routed_to_ocr(tmp_path: Path, caption_chars: int) -> None:
    """A scan stays a scan no matter how long the printed caption is.

    The 60-1200 rows are the regression: every one of them was classified
    digital before the fix, because each carries more than 50 characters.
    """
    src = _text_page(tmp_path / "src.pdf")
    pdf = _scan_with_caption(src, tmp_path / f"cap{caption_chars}.pdf", caption_chars)

    # Guard against a silently broken fixture: assert the caption really embedded.
    doc = fitz.open(pdf)
    embedded = len(doc[0].get_text().strip())
    doc.close()
    assert embedded >= caption_chars * 0.9, (
        f"fixture did not embed the caption ({embedded} of {caption_chars} chars) — "
        "the text likely overflowed its rect, so this row proves nothing"
    )

    result = classify(pdf)
    assert result.scanned_pages == [0], (
        f"scan with a {caption_chars}-char caption was not routed to OCR; "
        f"scanned_pages={result.scanned_pages} digital_pages={result.digital_pages}"
    )
    assert result.is_scanned


def test_searchable_pdf_stays_digital(tmp_path: Path) -> None:
    """An OCR'd searchable PDF is image-backed but must NOT be re-OCR'd.

    This is the case a naive "has images and few characters" rule gets wrong,
    and the reason the fix keys on text area rather than character count.
    """
    src = _text_page(tmp_path / "src.pdf")
    pdf = _searchable_pdf(src, tmp_path / "searchable.pdf")

    result = classify(pdf)
    assert result.scanned_pages == []
    assert result.is_digital


def test_native_text_page_stays_digital(tmp_path: Path) -> None:
    """No images, plenty of text — untouched by the captioned-scan branch."""
    result = classify(_text_page(tmp_path / "native.pdf"))
    assert result.scanned_pages == []
    assert result.is_digital


def test_fully_blank_scan_still_detected(tmp_path: Path) -> None:
    """An image-only page with no text at all — the pre-existing path."""
    src = _text_page(tmp_path / "src.pdf")
    pdf = _scan_with_caption(src, tmp_path / "blank.pdf", 0)

    result = classify(pdf)
    assert result.scanned_pages == [0]
    assert result.is_scanned


def test_small_logo_does_not_make_a_text_page_scanned(tmp_path: Path) -> None:
    """A digital page with a letterhead *logo* is still digital.

    Guards the other direction: the image-coverage floor must keep ordinary
    branded documents — invoices, datasheets — out of the OCR path.
    """
    pdf = tmp_path / "logo.pdf"
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64))
    pix.clear_with(128)
    page.insert_image(fitz.Rect(40, 20, 104, 84), pixmap=pix)
    page.insert_textbox(fitz.Rect(56, 100, 540, 760), _BODY * 8, fontsize=11, fontname="helv")
    doc.save(str(pdf))
    doc.close()

    result = classify(pdf)
    assert result.scanned_pages == []
    assert result.is_digital
