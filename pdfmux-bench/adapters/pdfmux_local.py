"""pdfmux adapter — runs locally, no API key.

This is the reference adapter and the only one that runs out-of-the-box: install
`pdfmux` (`pip install pdfmux`) and it works with no credentials. pdfmux is
included here as one competitor among many; it gets no scoring advantage — the
metrics in scoring.py are engine-agnostic and applied identically to every row.
"""

from __future__ import annotations

from pathlib import Path

from .base import Adapter


class PdfmuxAdapter(Adapter):
    name = "pdfmux"
    label = "pdfmux"
    homepage = "https://github.com/NameetP/pdfmux"
    license = "MIT"
    env_var = None  # local, no key

    def __init__(self, quality: str = "standard") -> None:
        # "fast" = PyMuPDF only; "standard" = + OCR fallback; "high" = + audit/LLM.
        self.quality = quality

    def available(self) -> bool:
        try:
            import pdfmux  # noqa: F401
        except Exception:
            return False
        return True

    def unavailable_reason(self) -> str:
        return "pdfmux is not installed. Run `pip install pdfmux` to include it."

    def extract(self, pdf_path: Path) -> str:
        import pdfmux

        # extract_text returns Markdown. use_cache is irrelevant here; we want the
        # real extraction path every run.
        return pdfmux.extract_text(str(pdf_path), quality=self.quality)
