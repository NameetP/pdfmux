"""Reducto adapter (closed API).

Reducto publishes RD-TableBench but has not refreshed it since ~2024. pdfmux-bench
includes Reducto as a first-class competitor so its current numbers sit next to
everyone else's, regenerable by anyone with a key.

API shape (as of 2026-07; verify against https://docs.reducto.ai)
-----------------------------------------------------------------
  Auth:      Authorization: Bearer $REDUCTO_API_KEY
  Upload:    POST https://platform.reducto.ai/upload   (multipart file)
             -> { "file_id": "reducto://..." }
  Parse:     POST https://platform.reducto.ai/parse
             body: { "document_url": "<file_id>", "options": {...} }
             -> { "result": { "type": "full", "chunks": [ { "content": "..." } ] } }

We concatenate chunk `content` (Reducto returns Markdown-ish blocks) into a
single Markdown string. If Reducto changes the response envelope, update
`_result_to_markdown` — the mapping is deliberately isolated.
"""

from __future__ import annotations

from pathlib import Path

from .base import Adapter

UPLOAD_URL = "https://platform.reducto.ai/upload"
PARSE_URL = "https://platform.reducto.ai/parse"
TIMEOUT = 300


class ReductoAdapter(Adapter):
    name = "reducto"
    label = "Reducto"
    homepage = "https://reducto.ai"
    license = "closed-api"
    env_var = "REDUCTO_API_KEY"

    def extract(self, pdf_path: Path) -> str:
        import requests  # gated dependency; see requirements.txt

        headers = {"Authorization": f"Bearer {self._key()}"}

        with pdf_path.open("rb") as fh:
            up = requests.post(
                UPLOAD_URL,
                headers=headers,
                files={"file": (pdf_path.name, fh, "application/pdf")},
                timeout=TIMEOUT,
            )
        up.raise_for_status()
        file_id = up.json()["file_id"]

        parse = requests.post(
            PARSE_URL,
            headers={**headers, "Content-Type": "application/json"},
            json={"document_url": file_id, "options": {"chunking": {"chunk_mode": "page"}}},
            timeout=TIMEOUT,
        )
        parse.raise_for_status()
        return _result_to_markdown(parse.json())


def _result_to_markdown(payload: dict) -> str:
    result = payload.get("result", payload)
    chunks = result.get("chunks") or []
    parts = []
    for ch in chunks:
        content = ch.get("content") or ch.get("text") or ""
        if content:
            parts.append(content)
    if parts:
        return "\n\n".join(parts)
    # Fallbacks for alternate envelopes.
    if isinstance(result.get("markdown"), str):
        return result["markdown"]
    raise ValueError("Reducto response contained no chunk content or markdown field")
