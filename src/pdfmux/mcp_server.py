"""MCP server — expose pdfmux as a tool for AI agents.

Usage:
    pdfmux serve

Then add to your Claude/Cursor config:
    { "mcpServers": { "pdfmux": { "command": "pdfmux", "args": ["serve"] } } }

Tools:
    convert_pdf   — extract text from a PDF (Markdown, JSON, LLM chunks)
    analyze_pdf   — quick triage: classify + audit without full extraction
    batch_convert — convert all PDFs in a directory
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pdfmux import __version__
from pdfmux.pipeline import process

# Security: restrict file access to allowed directories.
# Default: current working directory. Override via PDFMUX_ALLOWED_DIRS (colon-separated).
_ALLOWED_DIRS_ENV = os.environ.get("PDFMUX_ALLOWED_DIRS", "")
ALLOWED_DIRS: list[Path] = (
    [Path(d).resolve() for d in _ALLOWED_DIRS_ENV.split(":") if d.strip()]
    if _ALLOWED_DIRS_ENV
    else [Path.cwd().resolve()]
)


def _is_path_allowed(file_path: Path) -> bool:
    """Check if the given path is inside one of the allowed directories."""
    resolved = file_path.resolve()
    return any(resolved == d or d in resolved.parents for d in ALLOWED_DIRS)


def run_server() -> None:
    """Run the MCP server over stdio (JSON-RPC)."""
    _write_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = message.get("method", "")
        msg_id = message.get("id")

        if method == "initialize":
            _handle_initialize(msg_id)
        elif method == "tools/list":
            _handle_tools_list(msg_id)
        elif method == "tools/call":
            _handle_tools_call(msg_id, message.get("params", {}))
        elif msg_id is not None:
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )


def _handle_initialize(msg_id: int | str | None) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "pdfmux",
                    "version": __version__,
                },
            },
        }
    )


def _handle_tools_list(msg_id: int | str | None) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "convert_pdf",
                        "description": (
                            "Convert a PDF to AI-readable Markdown. "
                            "Automatically detects the PDF type and picks "
                            "the best extraction method. Returns confidence "
                            "score and warnings when extraction is limited."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Absolute path to the PDF file",
                                },
                                "format": {
                                    "type": "string",
                                    "description": "Output format: markdown (default), json",
                                    "enum": ["markdown", "json"],
                                    "default": "markdown",
                                },
                                "quality": {
                                    "type": "string",
                                    "description": (
                                        "Quality preset: fast, standard (default), high"
                                    ),
                                    "enum": ["fast", "standard", "high"],
                                    "default": "standard",
                                },
                            },
                            "required": ["file_path"],
                        },
                    },
                    {
                        "name": "analyze_pdf",
                        "description": (
                            "Quick PDF triage — classify type and audit page quality "
                            "without full extraction. Returns page count, type detection, "
                            "per-page quality breakdown, and estimated extraction difficulty. "
                            "Much cheaper than convert_pdf for initial assessment."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Absolute path to the PDF file",
                                },
                            },
                            "required": ["file_path"],
                        },
                    },
                    {
                        "name": "batch_convert",
                        "description": (
                            "Convert all PDFs in a directory to Markdown. "
                            "Returns a summary with per-file results."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "directory": {
                                    "type": "string",
                                    "description": "Absolute path to directory containing PDFs",
                                },
                                "quality": {
                                    "type": "string",
                                    "description": (
                                        "Quality preset: fast, standard (default), high"
                                    ),
                                    "enum": ["fast", "standard", "high"],
                                    "default": "standard",
                                },
                            },
                            "required": ["directory"],
                        },
                    },
                    {
                        "name": "extract_structured",
                        "description": (
                            "Extract structured data from a PDF — tables as JSON, "
                            "key-value pairs, and optionally map to a JSON schema. "
                            "Returns tables with headers/rows, detected key-value "
                            "pairs with auto-normalization (dates, amounts, rates), "
                            "and schema-mapped output if a schema is provided."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Absolute path to the PDF file",
                                },
                                "schema": {
                                    "type": "string",
                                    "description": (
                                        "Optional: path to a JSON schema file or "
                                        "preset name for schema-guided extraction"
                                    ),
                                },
                                "quality": {
                                    "type": "string",
                                    "description": (
                                        "Quality preset: fast, standard (default), high"
                                    ),
                                    "enum": ["fast", "standard", "high"],
                                    "default": "standard",
                                },
                            },
                            "required": ["file_path"],
                        },
                    },
                ]
            },
        }
    )


def _handle_tools_call(msg_id: int | str | None, params: dict) -> None:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "convert_pdf":
        _handle_convert_pdf(msg_id, arguments)
    elif tool_name == "analyze_pdf":
        _handle_analyze_pdf(msg_id, arguments)
    elif tool_name == "batch_convert":
        _handle_batch_convert(msg_id, arguments)
    elif tool_name == "extract_structured":
        _handle_extract_structured(msg_id, arguments)
    else:
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
            }
        )


def _handle_convert_pdf(msg_id: int | str | None, arguments: dict) -> None:
    file_path = arguments.get("file_path", "")
    fmt = arguments.get("format", "markdown")
    quality = arguments.get("quality", "standard")

    if not file_path:
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "file_path is required"},
            }
        )
        return

    if not _is_path_allowed(Path(file_path)):
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": (
                        f"Access denied: {file_path} is outside allowed directories. "
                        "Set PDFMUX_ALLOWED_DIRS to configure access."
                    ),
                },
            }
        )
        return

    try:
        result = process(
            file_path=file_path,
            output_format=fmt,
            quality=quality,
        )

        content_parts = []

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
            content_parts.append({"type": "text", "text": "\n".join(meta_lines)})

        content_parts.append({"type": "text", "text": result.text})

        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": content_parts},
            }
        )
    except Exception as e:
        error_code = getattr(e, "code", "UNKNOWN_ERROR")
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error converting PDF: {e}\nError code: {error_code}",
                        }
                    ],
                    "isError": True,
                },
            }
        )


def _handle_analyze_pdf(msg_id: int | str | None, arguments: dict) -> None:
    """Quick PDF triage — classify + audit without full extraction."""
    file_path = arguments.get("file_path", "")

    if not file_path:
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "file_path is required"},
            }
        )
        return

    if not _is_path_allowed(Path(file_path)):
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": (
                        f"Access denied: {file_path} is outside allowed directories. "
                        "Set PDFMUX_ALLOWED_DIRS to configure access."
                    ),
                },
            }
        )
        return

    try:
        from pdfmux.audit import audit_document
        from pdfmux.detect import classify

        classification = classify(file_path)
        audit = audit_document(file_path)

        # Build type detection summary
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

        # Per-page breakdown
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

        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(analysis, indent=2)}]},
            }
        )
    except Exception as e:
        error_code = getattr(e, "code", "UNKNOWN_ERROR")
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error analyzing PDF: {e}\nError code: {error_code}",
                        }
                    ],
                    "isError": True,
                },
            }
        )


def _handle_batch_convert(msg_id: int | str | None, arguments: dict) -> None:
    """Convert all PDFs in a directory."""
    directory = arguments.get("directory", "")
    quality = arguments.get("quality", "standard")

    if not directory:
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "directory is required"},
            }
        )
        return

    if not _is_path_allowed(Path(directory)):
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": (
                        f"Access denied: {directory} is outside allowed directories. "
                        "Set PDFMUX_ALLOWED_DIRS to configure access."
                    ),
                },
            }
        )
        return

    try:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": f"Not a directory: {directory}",
                    },
                }
            )
            return

        pdfs = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))
        if not pdfs:
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"No PDF files found in {directory}"}]
                    },
                }
            )
            return

        from pdfmux.pipeline import process_batch

        results = []
        for path, result_or_error in process_batch(pdfs, output_format="markdown", quality=quality):
            if isinstance(result_or_error, Exception):
                results.append(
                    {
                        "file": path.name,
                        "status": "error",
                        "error": str(result_or_error),
                    }
                )
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

        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]},
            }
        )
    except Exception as e:
        error_code = getattr(e, "code", "UNKNOWN_ERROR")
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error in batch convert: {e}\nError code: {error_code}",
                        }
                    ],
                    "isError": True,
                },
            }
        )


def _handle_extract_structured(msg_id: int | str | None, arguments: dict) -> None:
    """Extract structured data — tables, key-values, and schema mapping."""
    file_path = arguments.get("file_path", "")
    schema = arguments.get("schema")
    quality = arguments.get("quality", "standard")

    if not file_path:
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "file_path is required"},
            }
        )
        return

    if not _is_path_allowed(Path(file_path)):
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": (
                        f"Access denied: {file_path} is outside allowed directories. "
                        "Set PDFMUX_ALLOWED_DIRS to configure access."
                    ),
                },
            }
        )
        return

    try:
        result = process(
            file_path=file_path,
            output_format="json",
            quality=quality,
            schema=schema,
        )

        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": result.text}]},
            }
        )
    except Exception as e:
        error_code = getattr(e, "code", "UNKNOWN_ERROR")
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error extracting structured data: {e}\nError code: {error_code}",
                        }
                    ],
                    "isError": True,
                },
            }
        )


def _write_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()
