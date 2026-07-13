"""Marker adapter (open-source, runs locally).

Marker (VikParuchuri, GPL-3.0 for the code / model weights under their own terms)
converts PDFs to Markdown with a deep-learning pipeline. Installed via
`pip install marker-pdf`. No API key, but downloads model weights on first run.

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    markdown, _, _ = text_from_rendered(rendered)

Marker is licensed GPL-3.0 and its weights carry usage restrictions; we record
its license in the leaderboard so downstream users understand the terms. It runs
locally, so it can participate in CI where GPU/time budget allows (in the default
CI matrix it is opt-in via `MARKER_IN_CI=1` because model download is heavy).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .base import Adapter


class MarkerAdapter(Adapter):
    name = "marker"
    label = "Marker"
    homepage = "https://github.com/VikParuchuri/marker"
    license = "GPL-3.0"
    env_var = None  # local

    def available(self) -> bool:
        return importlib.util.find_spec("marker") is not None

    def unavailable_reason(self) -> str:
        return "Marker not installed. Run `pip install marker-pdf` to include it."

    def extract(self, pdf_path: Path) -> str:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        markdown, _, _ = text_from_rendered(rendered)
        return markdown
