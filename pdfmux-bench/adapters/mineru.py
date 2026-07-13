"""MinerU adapter (open-source, runs locally via CLI).

MinerU (OpenDataLab, AGPL-3.0) extracts Markdown + structured content from PDFs.
Installed via `pip install mineru` (or the older `magic-pdf`). It ships a CLI:

    mineru -p <input.pdf> -o <output_dir>

which writes `<stem>/<stem>.md` (plus layout/JSON artifacts) into the output dir.
We shell out to the CLI into a temp dir and read the resulting Markdown, which is
the most stable public interface across MinerU versions (the Python API has moved
between `magic_pdf` and `mineru` namespaces).

MinerU is AGPL-3.0 — recorded in the leaderboard. Runs locally; heavy model
download makes it opt-in in CI (`MINERU_IN_CI=1`).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import Adapter


class MinerUAdapter(Adapter):
    name = "mineru"
    label = "MinerU"
    homepage = "https://github.com/opendatalab/MinerU"
    license = "AGPL-3.0"
    env_var = None  # local CLI

    def available(self) -> bool:
        return shutil.which("mineru") is not None

    def unavailable_reason(self) -> str:
        return "MinerU CLI not found on PATH. Run `pip install mineru` to include it."

    def extract(self, pdf_path: Path) -> str:
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
            out_dir = Path(tmp)
            proc = subprocess.run(
                ["mineru", "-p", str(pdf_path), "-o", str(out_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"mineru CLI failed (exit {proc.returncode}): {proc.stderr[-500:]}"
                )
            # MinerU writes <stem>/.../<stem>.md — find the largest .md produced.
            md_files = sorted(out_dir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
            if not md_files:
                raise FileNotFoundError(
                    f"mineru produced no .md under {out_dir}; stdout: {proc.stdout[-300:]}"
                )
            return md_files[0].read_text(encoding="utf-8")
