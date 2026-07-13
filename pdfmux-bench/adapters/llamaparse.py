"""LlamaParse adapter (closed API, LlamaCloud).

API shape (as of 2026-07; verify against https://docs.cloud.llamaindex.ai)
--------------------------------------------------------------------------
  Auth:    Authorization: Bearer $LLAMA_CLOUD_API_KEY
  Upload:  POST https://api.cloud.llamaindex.ai/api/v1/parsing/upload
           multipart file  ->  { "id": "<job_id>", "status": "PENDING" }
  Poll:    GET  https://api.cloud.llamaindex.ai/api/v1/parsing/job/<job_id>
           ->  { "status": "SUCCESS" | "PENDING" | "ERROR" }
  Result:  GET  https://api.cloud.llamaindex.ai/api/v1/parsing/job/<job_id>/result/markdown
           ->  { "markdown": "..." }

LlamaParse is asynchronous — upload returns a job id, then we poll until the job
succeeds and fetch the Markdown result.
"""

from __future__ import annotations

import time
from pathlib import Path

from .base import Adapter

BASE = "https://api.cloud.llamaindex.ai/api/v1/parsing"
POLL_INTERVAL = 3
MAX_WAIT = 300


class LlamaParseAdapter(Adapter):
    name = "llamaparse"
    label = "LlamaParse"
    homepage = "https://cloud.llamaindex.ai"
    license = "closed-api"
    env_var = "LLAMA_CLOUD_API_KEY"

    def extract(self, pdf_path: Path) -> str:
        import requests

        headers = {"Authorization": f"Bearer {self._key()}", "accept": "application/json"}

        with pdf_path.open("rb") as fh:
            up = requests.post(
                f"{BASE}/upload",
                headers=headers,
                files={"file": (pdf_path.name, fh, "application/pdf")},
                timeout=120,
            )
        up.raise_for_status()
        job_id = up.json()["id"]

        deadline = time.time() + MAX_WAIT
        while time.time() < deadline:
            st = requests.get(f"{BASE}/job/{job_id}", headers=headers, timeout=60)
            st.raise_for_status()
            status = st.json().get("status")
            if status == "SUCCESS":
                break
            if status == "ERROR":
                raise RuntimeError(f"LlamaParse job {job_id} failed: {st.json()}")
            time.sleep(POLL_INTERVAL)
        else:
            raise TimeoutError(f"LlamaParse job {job_id} did not finish in {MAX_WAIT}s")

        res = requests.get(f"{BASE}/job/{job_id}/result/markdown", headers=headers, timeout=60)
        res.raise_for_status()
        return res.json()["markdown"]
