"""Engine adapters for pdfmux-bench.

Register every engine here. `ALL_ADAPTERS` is the single source of truth the
harness iterates over. To add an engine: write `adapters/<name>.py` with an
`Adapter` subclass, import it here, and append an instance to `ALL_ADAPTERS`.
That's the whole contribution surface — see CONTRIBUTING.md.
"""

from __future__ import annotations

from .base import Adapter
from .docling import DoclingAdapter
from .llamaparse import LlamaParseAdapter
from .marker import MarkerAdapter
from .mineru import MinerUAdapter
from .mistral_ocr import MistralOCRAdapter
from .pdfmux_local import PdfmuxAdapter
from .reducto import ReductoAdapter

#: Order here is the default leaderboard tie-break order only; scores decide rank.
ALL_ADAPTERS: list[Adapter] = [
    PdfmuxAdapter(),
    ReductoAdapter(),
    LlamaParseAdapter(),
    MistralOCRAdapter(),
    DoclingAdapter(),
    MarkerAdapter(),
    MinerUAdapter(),
]

#: Engines that run locally with no API key — the set CI can execute.
OSS_LOCAL = {"pdfmux", "docling", "marker", "mineru"}


def get_adapters(names: list[str] | None = None) -> list[Adapter]:
    """Return adapters filtered by name, or all of them."""
    if not names:
        return list(ALL_ADAPTERS)
    by_name = {a.name: a for a in ALL_ADAPTERS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(
            f"Unknown engine(s): {', '.join(missing)}. "
            f"Available: {', '.join(by_name)}"
        )
    return [by_name[n] for n in names]


__all__ = ["Adapter", "ALL_ADAPTERS", "OSS_LOCAL", "get_adapters"]
