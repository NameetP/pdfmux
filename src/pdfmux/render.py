"""Bounded page rendering for OCR and vision paths.

Every OCR path renders a page to a raster at a fixed DPI. DPI is relative to
the page's *point* size, and scanned PDFs often have absurd point sizes: a
phone-scanner app or ``img2pdf`` places a 12 MP photo at 1 px = 1 pt, so the
page is 3024 x 4032 pt (42 x 56 inches). At 200 DPI that renders to
8400 x 11200 px = 282 MB of raw RGB per page, before PNG encoding and ONNX
inference. Four of those in ``parallel_ocr`` peaked at 1.5 GB and hit memory
limits in sandboxes and agent runtimes.

``render_pixmap`` keeps the requested DPI for normal pages and lowers it only
when the render would exceed ``max_pixels``. The OCR engines downscale large
inputs internally anyway, so the cap costs no accuracy on real documents.
"""

from __future__ import annotations

import logging
import math
import os

import fitz

logger = logging.getLogger(__name__)

# 16 MP: A4/Letter at 300 DPI (~8.7 MP) and A3 at 200 DPI (~7.8 MP) render
# untouched. Only oversized pages are scaled down. ~48 MB raw RGB per page.
DEFAULT_MAX_PIXELS = 16_000_000
_ENV_VAR = "PDFMUX_MAX_RENDER_PIXELS"


def max_render_pixels() -> int:
    """Pixel budget per render; ``PDFMUX_MAX_RENDER_PIXELS`` overrides it."""
    raw = os.environ.get(_ENV_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", _ENV_VAR, raw)
        else:
            if value > 0:
                return value
            logger.warning("Ignoring non-positive %s=%r", _ENV_VAR, raw)
    return DEFAULT_MAX_PIXELS


def effective_dpi(rect: fitz.Rect, dpi: float, max_pixels: int | None = None) -> float:
    """DPI to render ``rect`` at so the raster stays within ``max_pixels``."""
    budget = max_pixels if max_pixels is not None else max_render_pixels()
    width_in = max(rect.width, 0.0) / 72.0
    height_in = max(rect.height, 0.0) / 72.0
    area_in = width_in * height_in
    if area_in <= 0 or (width_in * dpi) * (height_in * dpi) <= budget:
        return float(dpi)
    # Shave a hair off so float rounding never lands a render one pixel over budget.
    return min(float(dpi), math.sqrt(budget / area_in) * (1 - 1e-9))


def render_pixmap(
    page: fitz.Page,
    dpi: float,
    *,
    clip: fitz.Rect | None = None,
    max_pixels: int | None = None,
) -> fitz.Pixmap:
    """Render ``page`` (or ``clip``) at ``dpi``, scaled down to fit the pixel budget."""
    target = fitz.Rect(clip) & page.rect if clip is not None else page.rect
    use_dpi = effective_dpi(target, dpi, max_pixels)
    if use_dpi < dpi:
        logger.info(
            "Page %d is %.0fx%.0f pt; rendering at %.0f DPI instead of %.0f to stay under %d px",
            page.number,
            target.width,
            target.height,
            use_dpi,
            dpi,
            max_pixels if max_pixels is not None else max_render_pixels(),
        )
    zoom = use_dpi / 72.0
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
