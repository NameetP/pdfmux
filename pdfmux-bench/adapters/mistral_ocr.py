"""Mistral OCR adapter (closed API).

API shape (as of 2026-07; verify against https://docs.mistral.ai/capabilities/OCR)
----------------------------------------------------------------------------------
  Auth:     Authorization: Bearer $MISTRAL_API_KEY
  Endpoint: POST https://api.mistral.ai/v1/ocr
  Body:     {
              "model": "mistral-ocr-latest",
              "document": { "type": "document_url",
                            "document_url": "data:application/pdf;base64,<...>" }
            }
  Response: { "pages": [ { "index": 0, "markdown": "..." }, ... ] }

Mistral OCR accepts a base64 data URI directly, so no separate upload step is
needed. We concatenate per-page `markdown` in page order.
"""

from __future__ import annotations

import base64
from pathlib import Path

from .base import Adapter

OCR_URL = "https://api.mistral.ai/v1/ocr"
MODEL = "mistral-ocr-latest"
TIMEOUT = 300


class MistralOCRAdapter(Adapter):
    name = "mistral_ocr"
    label = "Mistral OCR"
    homepage = "https://mistral.ai"
    license = "closed-api"
    env_var = "MISTRAL_API_KEY"

    def extract(self, pdf_path: Path) -> str:
        import requests

        b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
        data_uri = f"data:application/pdf;base64,{b64}"

        resp = requests.post(
            OCR_URL,
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "document": {"type": "document_url", "document_url": data_uri},
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        pages = resp.json().get("pages", [])
        pages_sorted = sorted(pages, key=lambda p: p.get("index", 0))
        md = "\n\n".join(p.get("markdown", "") for p in pages_sorted)
        if not md.strip():
            raise ValueError("Mistral OCR returned no page markdown")
        return md
