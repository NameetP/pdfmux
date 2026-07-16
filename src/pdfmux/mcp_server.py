"""MCP server — expose pdfmux as a tool for AI agents.

Usage (stdio — default, backward compatible):
    pdfmux serve

Usage (HTTP — for Smithery / remote deployment):
    pdfmux serve --http
    pdfmux serve --http --port 9000

Then add to your Claude/Cursor config:
    { "mcpServers": { "pdfmux": { "command": "pdfmux", "args": ["serve"] } } }

Tools:
    convert_pdf        — extract text from a PDF (Markdown, JSON, LLM chunks)
    analyze_pdf        — quick triage: classify + audit without full extraction
    batch_convert      — convert all PDFs in a directory
    extract_structured — tables, key-values, schema mapping
    extract_streaming  — page-by-page NDJSON for large documents
    get_pdf_metadata   — page count, file size, type detection (instant, no extraction)
    verify_extraction  — audit ANY engine's extraction for silently-dropped pages
"""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from pdfmux.path_safety import (
    ALLOWED_DIRS,
)
from pdfmux.path_safety import (
    check_path as _check_path,
)
from pdfmux.path_safety import (
    is_path_allowed as _is_path_allowed,
)
from pdfmux.pipeline import process

__all__ = [
    "ALLOWED_DIRS",
    "_check_path",
    "_is_path_allowed",
    "mcp",
    "run_server",
    "run_http_server",
]


# ---------------------------------------------------------------------------
# Server instance — created once, started by run_server() or run_http_server()
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="pdfmux",
    instructions=(
        "pdfmux converts PDFs to AI-readable Markdown. "
        "Use convert_pdf for full extraction, analyze_pdf for quick triage, "
        "batch_convert for directories, extract_structured for tables/key-values, "
        "extract_streaming for page-by-page NDJSON streaming on large docs, "
        "and verify_extraction to audit any engine's extraction for silently-dropped pages."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_pdf_metadata(file_path: str) -> str:
    """Get PDF metadata instantly — page count, file size, document type, and whether it has tables. No extraction performed. Use this first to decide which tool to call next: convert_pdf for full text, analyze_pdf for quality audit, or extract_structured for tables."""
    p = _check_path(file_path)

    if not p.exists():
        raise ValueError(f"File not found: {file_path}")

    from pdfmux.detect import classify

    classification = classify(str(p))
    file_size = p.stat().st_size

    types = []
    if classification.is_digital:
        types.append("digital")
    if classification.is_scanned:
        types.append("scanned")
    if classification.is_mixed:
        types.append("mixed")
    if classification.is_graphical:
        types.append("graphical")
    if classification.has_tables:
        types.append("has_tables")

    metadata = {
        "file": str(file_path),
        "file_size_bytes": file_size,
        "file_size_human": (
            f"{file_size / 1024 / 1024:.1f} MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.0f} KB"
        ),
        "page_count": classification.page_count,
        "detected_types": types,
        "has_tables": classification.has_tables,
        "is_scanned": classification.is_scanned,
        "recommended_quality": ("high" if classification.is_scanned else "standard"),
        "recommended_tool": ("extract_structured" if classification.has_tables else "convert_pdf"),
    }

    return json.dumps(metadata, indent=2)


@mcp.tool()
def convert_pdf(
    file_path: str,
    format: str = "markdown",
    quality: str = "standard",
) -> str:
    """Convert a PDF to AI-readable Markdown. Automatically detects the PDF type and picks the best extraction method. Returns confidence score and warnings when extraction is limited."""
    p = _check_path(file_path)

    result = process(
        file_path=str(p),
        output_format=format,
        quality=quality,
    )

    parts: list[str] = []

    if result.confidence < 0.8 or result.warnings:
        meta_lines = [
            f"**Extraction confidence: {result.confidence:.0%}**",
            f"Extractor: {result.extractor_used}",
            f"Pages: {result.page_count}",
        ]
        if result.ocr_pages:
            meta_lines.append(f"OCR pages: {', '.join(str(p + 1) for p in result.ocr_pages)}")
        if result.warnings:
            meta_lines.append("")
            meta_lines.append("**Warnings:**")
            for w in result.warnings:
                meta_lines.append(f"- {w}")
        meta_lines.append("")
        meta_lines.append("---")
        meta_lines.append("")
        parts.append("\n".join(meta_lines))

    parts.append(result.text)
    return "\n".join(parts)


@mcp.tool()
def analyze_pdf(file_path: str) -> str:
    """Quick PDF triage — classify type and audit page quality without full extraction. Returns page count, type detection, per-page quality breakdown, and estimated extraction difficulty. Much cheaper than convert_pdf for initial assessment."""
    p = _check_path(file_path)

    from pdfmux.audit import audit_document
    from pdfmux.detect import classify

    classification = classify(str(p))
    audit = audit_document(str(p))

    types = []
    if classification.is_digital:
        types.append("digital")
    if classification.is_scanned:
        types.append("scanned")
    if classification.is_mixed:
        types.append("mixed")
    if classification.is_graphical:
        types.append("graphical")
    if classification.has_tables:
        types.append("tables")

    pages_info = []
    for pa in audit.pages:
        pages_info.append(
            {
                "page": pa.page_num + 1,
                "quality": pa.quality,
                "chars": pa.text_len,
                "images": pa.image_count,
                "reason": pa.reason,
            }
        )

    analysis = {
        "file": str(file_path),
        "page_count": classification.page_count,
        "detected_types": types,
        "detection_confidence": round(classification.confidence, 3),
        "needs_ocr": audit.needs_ocr,
        "good_pages": len(audit.good_pages),
        "bad_pages": len(audit.bad_pages),
        "empty_pages": len(audit.empty_pages),
        "pages": pages_info,
    }

    return json.dumps(analysis, indent=2)


@mcp.tool()
def batch_convert(directory: str, quality: str = "standard") -> str:
    """Convert all PDFs in a directory to Markdown. Returns a summary with per-file results."""
    p = _check_path(directory, label="directory")

    if not p.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    pdfs = list(p.glob("*.pdf")) + list(p.glob("*.PDF"))
    if not pdfs:
        return f"No PDF files found in {directory}"

    from pdfmux.pipeline import process_batch

    results = []
    for path, result_or_error in process_batch(pdfs, output_format="markdown", quality=quality):
        if isinstance(result_or_error, Exception):
            results.append({"file": path.name, "status": "error", "error": str(result_or_error)})
        else:
            results.append(
                {
                    "file": path.name,
                    "status": "success",
                    "pages": result_or_error.page_count,
                    "confidence": round(result_or_error.confidence, 3),
                    "extractor": result_or_error.extractor_used,
                    "chars": len(result_or_error.text),
                }
            )

    summary = {
        "directory": str(directory),
        "total_files": len(pdfs),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }

    return json.dumps(summary, indent=2)


@mcp.tool()
def extract_structured(
    file_path: str,
    schema: str | None = None,
    quality: str = "standard",
) -> str:
    """Extract structured data from a PDF — tables as JSON, key-value pairs, and optionally map to a JSON schema. Returns tables with headers/rows, detected key-value pairs with auto-normalization (dates, amounts, rates), and schema-mapped output if a schema is provided."""
    p = _check_path(file_path)

    result = process(
        file_path=str(p),
        output_format="json",
        quality=quality,
        schema=schema,
    )

    return result.text


@mcp.tool()
def extract_streaming(
    file_path: str,
    quality: str = "standard",
) -> str:
    """Stream extraction events for a PDF as NDJSON.

    Use for large documents (100+ pages) where waiting for the full
    extraction is impractical. The response body is newline-delimited
    JSON with one object per line:

        {"type":"classified","data":{"page_count":N,"page_types":[...]}}
        {"type":"page","data":{"page_num":0,"text":"...","confidence":0.92,...}}
        {"type":"warning","data":{"message":"..."}}        (zero or more)
        {"type":"complete","data":{"total_confidence":0.94,"ocr_pages":[...],...}}

    The first event is always ``classified``; the last is always ``complete``.
    Each ``page`` event arrives as soon as that page is extracted, including
    OCR re-extraction in standard/high quality modes.
    """
    p = _check_path(file_path)

    from pdfmux.streaming import process_streaming

    lines: list[str] = []
    for ev in process_streaming(str(p), quality=quality):
        lines.append(json.dumps(ev.to_dict(), ensure_ascii=False))
    return "\n".join(lines)


@mcp.tool()
def verify_extraction(
    file_path: str,
    extracted_text: str | None = None,
    engine: str = "external",
    fmt: str = "auto",
) -> str:
    """Audit an extraction of a PDF for silently-dropped pages — the failure where an extractor returns nothing for a page that has real text while reporting success. Pass extracted_text with another engine's output (Reducto, Mistral OCR, LlamaParse, Docling, an in-house parser — as JSON, Markdown, or plain text) to certify THAT engine against the source PDF; omit it to have pdfmux extract the document itself and certify its own read. Returns the per-page audit — each page marked usable / silently-empty / recovered / review / unverifiable — the "N of M pages silently dropped" headline, and an overall PASS/REVIEW/FAIL verdict with a tamper-evident signature. Reuses pdfmux's own audit pass as the ground truth."""
    p = _check_path(file_path)

    if not p.exists():
        raise ValueError(f"File not found: {file_path}")

    from pdfmux import verify_extraction as _verify_extraction

    if extracted_text is not None and extracted_text.strip():
        engine_label = engine
        extracted: str = extracted_text
        fmt_hint = fmt
    else:
        # No external extraction supplied — audit pdfmux's own read of the doc.
        engine_label = "pdfmux" if engine == "external" else engine
        extracted = process(
            file_path=str(p),
            output_format="json",
            quality="standard",
        ).text
        fmt_hint = "json"

    manifest = _verify_extraction(str(p), extracted, engine=engine_label, fmt=fmt_hint)
    manifest_dict = manifest.to_dict()

    def _status(page: dict) -> str:
        if page.get("silent_drop"):
            return "silently-empty"
        if "content_off_page" in page.get("flags", []):
            return "recovered"
        verdict = page.get("verdict")
        return "usable" if verdict == "pass" else verdict

    pages = []
    for pv in manifest.pages:
        row = pv.to_dict()
        pages.append(
            {
                "page": row["page"],
                "status": _status(row),
                "coverage": row["coverage"],
                "confidence": row["confidence"],
                "alignment": row["alignment"],
                "hallucination_risk": row["hallucination_risk"],
                "flags": row["flags"],
            }
        )

    n_dropped = len(manifest.silent_drops)
    report = {
        "source": manifest.source,
        "engine": manifest.engine,
        "verdict": manifest.verdict,
        "page_count": manifest.page_count,
        "page_aligned": manifest.page_aligned,
        "pages_silently_dropped": n_dropped,
        "silently_dropped_pages": list(manifest.silent_drops),
        "headline": f"{n_dropped} of {manifest.page_count} page(s) silently dropped",
        "summary": manifest.summary,
        "overall_confidence": manifest.confidence,
        "overall_coverage": manifest.coverage,
        "pages": pages,
        "signature": manifest.signature,
        "tool": manifest.tool,
        "limitations": manifest_dict["limitations"],
    }

    return json.dumps(report, indent=2)


# ---------------------------------------------------------------------------
# Server entry points
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Run the MCP server over stdio (JSON-RPC). Default transport."""
    mcp.run(transport="stdio")


def run_http_server(host: str | None = None, port: int = 8000) -> None:
    """Run the MCP server over Streamable HTTP transport.

    Default bind host is loopback (``127.0.0.1``) — safer than ``0.0.0.0``.
    Override via the ``host`` argument or the ``PDFMUX_HTTP_HOST`` env var.
    Set ``PDFMUX_HTTP_HOST=0.0.0.0`` (or pass ``--host 0.0.0.0`` on the CLI)
    when intentionally exposing the server to a non-loopback network.
    """
    if host is None:
        host = os.environ.get("PDFMUX_HTTP_HOST", "127.0.0.1")
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
