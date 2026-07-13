"""Docling adapter (open-source, runs locally).

Docling (IBM / open source, MIT) converts documents to a structured object that
exports to Markdown. Installed via `pip install docling`. No API key.

    from docling.document_converter import DocumentConverter
    result = DocumentConverter().convert(pdf_path)
    markdown = result.document.export_to_markdown()

Because Docling runs locally, it participates in CI on the public corpus.
"""

from __future__ import annotations

from pathlib import Path

from .base import Adapter


class DoclingAdapter(Adapter):
    name = "docling"
    label = "Docling"
    homepage = "https://github.com/docling-project/docling"
    license = "MIT"
    env_var = None  # local

    def available(self) -> bool:
        try:
            import docling  # noqa: F401
        except Exception:
            return False
        return True

    def unavailable_reason(self) -> str:
        return "Docling not installed. Run `pip install docling` to include it."

    def extract(self, pdf_path: Path) -> str:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
