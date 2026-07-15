"""Single-tool MCP server — one tool, ``extract_pdf``.

This is a deliberately minimal wrapper around pdfmux's extraction pipeline,
built for MCP hosts and directories that reward a small, obvious tool surface
(Claude Desktop, Cursor, open-notebook, the Claude MCP directory). The full
multi-tool server lives in ``pdfmux.mcp_server`` (``pdfmux serve``); use that
when you also want triage, batch, structured-table, and streaming tools.

Usage (stdio — default, for Claude Desktop / Cursor / open-notebook):
    pdfmux-extract
    # or
    python -m pdfmux.mcp_extract

Config snippet:
    { "mcpServers": { "pdfmux": { "command": "pdfmux-extract" } } }

The one tool:
    extract_pdf — convert a PDF to AI-readable Markdown, with a confidence
                  score and warnings surfaced when extraction is uncertain.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from pdfmux.path_safety import check_path as _check_path
from pdfmux.pipeline import process

__all__ = ["mcp", "extract_pdf", "run_server", "run_http_server"]


mcp = FastMCP(
    name="pdfmux",
    instructions=(
        "pdfmux converts a PDF to clean, AI-readable Markdown. Call extract_pdf "
        "with the path to a PDF. It auto-detects whether the document is digital "
        "or scanned and picks the best extraction path, then reports a confidence "
        "score so the agent can tell a clean extraction from a degraded one "
        "instead of silently trusting garbage text."
    ),
)


@mcp.tool()
def extract_pdf(file_path: str, quality: str = "standard") -> str:
    """Convert a PDF to AI-readable Markdown.

    Auto-detects the document type (digital, scanned, or mixed) and picks the
    best extraction method. When the result is uncertain — a low-confidence page,
    an image-only scan, or a partial extraction — a short header is prepended with
    the confidence score, the extractor used, and any warnings, so you can decide
    whether to trust the text rather than silently ingesting a bad extraction.

    Args:
        file_path: Absolute path to the PDF file to extract.
        quality: "fast" (native text only, no OCR — cheapest), "standard"
            (native text + OCR fallback on image pages — the default), or
            "high" (OCR/vision on every page that needs it — most thorough).

    Returns:
        The document as Markdown. If confidence is below 80% or warnings were
        raised, a metadata header precedes the content.
    """
    p = _check_path(file_path)

    if not p.exists():
        raise ValueError(f"File not found: {file_path}")

    if quality not in ("fast", "standard", "high"):
        raise ValueError(f"quality must be 'fast', 'standard', or 'high' — got {quality!r}")

    result = process(file_path=str(p), output_format="markdown", quality=quality)

    parts: list[str] = []

    if result.confidence < 0.8 or result.warnings:
        header = [
            f"**Extraction confidence: {result.confidence:.0%}**",
            f"Extractor: {result.extractor_used}",
            f"Pages: {result.page_count}",
        ]
        if result.ocr_pages:
            header.append(f"OCR pages: {', '.join(str(pg + 1) for pg in result.ocr_pages)}")
        if result.warnings:
            header.append("")
            header.append("**Warnings:**")
            header.extend(f"- {w}" for w in result.warnings)
        header.append("")
        header.append("---")
        header.append("")
        parts.append("\n".join(header))

    parts.append(result.text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Server entry points
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Run the single-tool MCP server over stdio (JSON-RPC). Default transport."""
    mcp.run(transport="stdio")


def run_http_server(host: str | None = None, port: int = 8000) -> None:
    """Run the single-tool MCP server over Streamable HTTP transport.

    Default bind host is loopback (``127.0.0.1``). Override via the ``host``
    argument or ``PDFMUX_HTTP_HOST``; set ``0.0.0.0`` only when intentionally
    exposing the server to a non-loopback network.
    """
    if host is None:
        host = os.environ.get("PDFMUX_HTTP_HOST", "127.0.0.1")
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_server()
