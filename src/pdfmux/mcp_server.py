"""MCP server — expose Pdfmux as a tool for AI agents.

Usage:
    pdfmux serve

Then add to your Claude/Cursor config:
    { "mcpServers": { "pdfmux": { "command": "pdfmux", "args": ["serve"] } } }
"""

from __future__ import annotations

import json
import sys

from pdfmux.pipeline import process


def run_server() -> None:
    """Run the MCP server over stdio.

    Implements the Model Context Protocol (MCP) for tool execution.
    Reads JSON-RPC messages from stdin, processes them, writes responses to stdout.
    """
    # Write server info
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
    """Handle the initialize request."""
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "pdfmux",
                    "version": "0.3.0",
                },
            },
        }
    )


def _handle_tools_list(msg_id: int | str | None) -> None:
    """Handle the tools/list request."""
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
                            "Automatically detects the PDF type (digital, "
                            "graphical, scanned, mixed) and picks the best "
                            "extraction method. Returns confidence score "
                            "and warnings when extraction is limited."
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
                                    "description": "Output format: markdown (default)",
                                    "enum": ["markdown"],
                                    "default": "markdown",
                                },
                                "quality": {
                                    "type": "string",
                                    "description": (
                                        "Quality preset: fast (rule-based), "
                                        "standard (auto), high (ML-based)"
                                    ),
                                    "enum": ["fast", "standard", "high"],
                                    "default": "standard",
                                },
                            },
                            "required": ["file_path"],
                        },
                    }
                ]
            },
        }
    )


def _handle_tools_call(msg_id: int | str | None, params: dict) -> None:
    """Handle a tools/call request."""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name != "convert_pdf":
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
            }
        )
        return

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

    try:
        result = process(
            file_path=file_path,
            output_format=fmt,
            quality=quality,
        )

        # Build response with extraction metadata and warnings
        content_parts = []

        # Add warnings/metadata header if confidence is below 80%
        if result.confidence < 0.8 or result.warnings:
            meta_lines = [
                f"**Extraction confidence: {result.confidence:.0%}**",
                f"Extractor: {result.extractor_used}",
                f"Pages: {result.page_count}",
            ]
            if result.ocr_pages:
                meta_lines.append(
                    f"OCR pages: {', '.join(str(p + 1) for p in result.ocr_pages)}"
                )
            if result.warnings:
                meta_lines.append("")
                meta_lines.append("**Warnings:**")
                for w in result.warnings:
                    meta_lines.append(f"- ⚠ {w}")
            meta_lines.append("")
            meta_lines.append("---")
            meta_lines.append("")
            content_parts.append(
                {
                    "type": "text",
                    "text": "\n".join(meta_lines),
                }
            )

        content_parts.append(
            {
                "type": "text",
                "text": result.text,
            }
        )

        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": content_parts,
                },
            }
        )
    except Exception as e:
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error converting PDF: {e}",
                        }
                    ],
                    "isError": True,
                },
            }
        )


def _write_message(message: dict) -> None:
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()
