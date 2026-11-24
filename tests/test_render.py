"""Tests for bounded page rendering (pdfmux.render).

Regression: phone-scanner PDFs place a 12 MP photo at 1 px = 1 pt, so a page is
3024 x 4032 pt. Rendering it at a fixed 200 DPI produced an 8400 x 11200 px
raster (282 MB) per page and four parallel OCR workers ran out of memory.
"""

from __future__ import annotations

import fitz
import pytest

from pdfmux import render
from pdfmux.render import DEFAULT_MAX_PIXELS, effective_dpi, max_render_pixels, render_pixmap

A4 = fitz.Rect(0, 0, 595, 842)
PHONE_SCAN = fitz.Rect(0, 0, 3024, 4032)


def _doc(width: float, height: float) -> fitz.Document:
    doc = fitz.open()
    doc.new_page(width=width, height=height)
    return doc


class TestEffectiveDpi:
    def test_a4_at_300_dpi_is_untouched(self):
        assert effective_dpi(A4, 300) == 300

    def test_letter_at_300_dpi_is_untouched(self):
        assert effective_dpi(fitz.Rect(0, 0, 612, 792), 300) == 300

    def test_a3_at_200_dpi_is_untouched(self):
        assert effective_dpi(fitz.Rect(0, 0, 842, 1191), 200) == 200

    def test_phone_scan_is_capped_to_budget(self):
        dpi = effective_dpi(PHONE_SCAN, 200)
        assert dpi < 200
        px = (PHONE_SCAN.width / 72 * dpi) * (PHONE_SCAN.height / 72 * dpi)
        assert px <= DEFAULT_MAX_PIXELS
        # Uses the budget rather than collapsing far below it.
        assert px > DEFAULT_MAX_PIXELS * 0.99

    def test_never_raises_dpi(self):
        assert effective_dpi(A4, 72, max_pixels=10**12) == 72

    def test_pathological_page_still_fits_budget(self):
        huge = fitz.Rect(0, 0, 200_000, 200_000)
        dpi = effective_dpi(huge, 200)
        assert 0 < dpi < 200
        assert (huge.width / 72 * dpi) ** 2 <= DEFAULT_MAX_PIXELS

    def test_empty_rect_keeps_requested_dpi(self):
        assert effective_dpi(fitz.Rect(0, 0, 0, 0), 200) == 200


class TestMaxRenderPixelsEnv:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("PDFMUX_MAX_RENDER_PIXELS", raising=False)
        assert max_render_pixels() == DEFAULT_MAX_PIXELS

    def test_override(self, monkeypatch):
        monkeypatch.setenv("PDFMUX_MAX_RENDER_PIXELS", "4000000")
        assert max_render_pixels() == 4_000_000

    @pytest.mark.parametrize("bad", ["lots", "0", "-5", "1.5"])
    def test_invalid_falls_back_to_default(self, monkeypatch, bad):
        monkeypatch.setenv("PDFMUX_MAX_RENDER_PIXELS", bad)
        assert max_render_pixels() == DEFAULT_MAX_PIXELS


class TestRenderPixmap:
    def test_a4_renders_at_requested_dpi(self):
        page = _doc(595, 842)[0]
        pix = render_pixmap(page, 200)
        assert (pix.width, pix.height) == (1653, 2339)

    def test_phone_scan_render_stays_in_budget(self):
        page = _doc(3024, 4032)[0]
        pix = render_pixmap(page, 200)
        assert pix.width * pix.height <= DEFAULT_MAX_PIXELS * 1.001  # rounding
        assert pix.width * pix.height < 8400 * 11200 / 5

    def test_explicit_budget(self):
        page = _doc(3024, 4032)[0]
        pix = render_pixmap(page, 200, max_pixels=1_000_000)
        # PyMuPDF rounds the raster outward by up to 1 px per edge.
        assert pix.width * pix.height <= 1_000_000 + pix.width + pix.height + 1

    def test_clip_is_budgeted_on_clip_not_page(self):
        page = _doc(3024, 4032)[0]
        clip = fitz.Rect(0, 0, 300, 300)  # small region of a huge page: no cap needed
        pix = render_pixmap(page, 200, clip=clip)
        assert (pix.width, pix.height) == (834, 834)

    def test_clip_outside_page_does_not_crash(self):
        page = _doc(595, 842)[0]
        pix = render_pixmap(page, 200, clip=fitz.Rect(5000, 5000, 5100, 5100))
        assert pix.width * pix.height >= 0


def test_ocr_paths_use_bounded_renderer():
    """No OCR/vision path may call page.get_pixmap directly (fixed-DPI bypasses the cap)."""
    import pathlib

    src = pathlib.Path(render.__file__).parent
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p.name != "render.py" and "get_pixmap(" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
