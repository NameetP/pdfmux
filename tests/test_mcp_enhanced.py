"""Tests for enhanced MCP server — analyze_pdf and batch_convert tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pdfmux.mcp_server import (
    _handle_analyze_pdf,
    _handle_batch_convert,
)


def _allow_path(path: Path):
    """Context manager to allow a path in MCP server."""
    return patch("pdfmux.mcp_server.ALLOWED_DIRS", [path.resolve()])


class TestAnalyzePdf:
    """Tests for the analyze_pdf MCP tool."""

    def test_analyze_returns_metadata(self, digital_pdf: Path) -> None:
        """analyze_pdf should return classification and audit metadata."""
        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            _allow_path(digital_pdf.parent),
        ):
            _handle_analyze_pdf(msg_id=1, arguments={"file_path": str(digital_pdf)})

        assert len(captured) == 1
        msg = captured[0]
        assert msg["id"] == 1
        assert "result" in msg

        content = msg["result"]["content"][0]["text"]
        data = json.loads(content)

        assert "page_count" in data
        assert data["page_count"] == 2
        assert "detected_types" in data
        assert "digital" in data["detected_types"]
        assert "needs_ocr" in data
        assert "good_pages" in data
        assert "pages" in data
        assert len(data["pages"]) == 2

    def test_analyze_missing_file_path(self) -> None:
        """analyze_pdf without file_path should return error."""
        captured = []

        with patch("pdfmux.mcp_server._write_message", side_effect=captured.append):
            _handle_analyze_pdf(msg_id=2, arguments={})

        assert len(captured) == 1
        msg = captured[0]
        assert "error" in msg
        assert msg["error"]["code"] == -32602

    def test_analyze_page_quality_fields(self, digital_pdf: Path) -> None:
        """Each page in analyze output should have quality fields."""
        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            _allow_path(digital_pdf.parent),
        ):
            _handle_analyze_pdf(msg_id=3, arguments={"file_path": str(digital_pdf)})

        content = captured[0]["result"]["content"][0]["text"]
        data = json.loads(content)

        for page_info in data["pages"]:
            assert "page" in page_info
            assert "quality" in page_info
            assert "chars" in page_info
            assert "images" in page_info
            assert "reason" in page_info


class TestBatchConvert:
    """Tests for the batch_convert MCP tool."""

    def test_batch_converts_directory(self, tmp_path: Path, digital_pdf: Path) -> None:
        """batch_convert should process PDFs in a directory."""
        import shutil

        # Use a subdirectory to avoid fixture PDFs in tmp_path
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        dest = batch_dir / "test.pdf"
        shutil.copy2(digital_pdf, dest)

        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            _allow_path(tmp_path),
        ):
            _handle_batch_convert(msg_id=4, arguments={"directory": str(batch_dir)})

        assert len(captured) == 1
        msg = captured[0]
        assert "result" in msg

        content = msg["result"]["content"][0]["text"]
        data = json.loads(content)

        assert data["total_files"] == 1
        assert data["success"] == 1
        assert data["failed"] == 0
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "success"
        assert data["results"][0]["file"] == "test.pdf"

    def test_batch_empty_directory(self, tmp_path: Path) -> None:
        """batch_convert with no PDFs should report empty."""
        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            _allow_path(tmp_path),
        ):
            _handle_batch_convert(msg_id=5, arguments={"directory": str(tmp_path)})

        assert len(captured) == 1
        content = captured[0]["result"]["content"][0]["text"]
        assert "No PDF files found" in content

    def test_batch_missing_directory(self) -> None:
        """batch_convert without directory should return error."""
        captured = []

        with patch("pdfmux.mcp_server._write_message", side_effect=captured.append):
            _handle_batch_convert(msg_id=6, arguments={})

        assert len(captured) == 1
        msg = captured[0]
        assert "error" in msg
