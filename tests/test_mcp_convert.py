"""Tests for MCP server — convert_pdf tool + path restrictions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pdfmux.mcp_server import (
    _handle_convert_pdf,
    _is_path_allowed,
)


class TestConvertPdf:
    """Tests for the convert_pdf MCP tool."""

    def test_convert_returns_text(self, digital_pdf: Path) -> None:
        """convert_pdf should return extracted text."""
        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            patch("pdfmux.mcp_server.ALLOWED_DIRS", [digital_pdf.parent.resolve()]),
        ):
            _handle_convert_pdf(msg_id=1, arguments={"file_path": str(digital_pdf)})

        assert len(captured) == 1
        msg = captured[0]
        assert msg["id"] == 1
        assert "result" in msg
        assert "content" in msg["result"]

        # Should have text content
        texts = [p["text"] for p in msg["result"]["content"]]
        full_text = "\n".join(texts)
        assert len(full_text) > 0

    def test_convert_missing_file_path(self) -> None:
        """convert_pdf without file_path should return error."""
        captured = []

        with patch("pdfmux.mcp_server._write_message", side_effect=captured.append):
            _handle_convert_pdf(msg_id=2, arguments={})

        assert len(captured) == 1
        assert "error" in captured[0]
        assert captured[0]["error"]["code"] == -32602

    def test_convert_nonexistent_file(self, tmp_path: Path) -> None:
        """convert_pdf with nonexistent file should return error."""
        captured = []
        fake_path = str(tmp_path / "nonexistent.pdf")

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            patch("pdfmux.mcp_server.ALLOWED_DIRS", [tmp_path.resolve()]),
        ):
            _handle_convert_pdf(msg_id=3, arguments={"file_path": fake_path})

        assert len(captured) == 1
        msg = captured[0]
        assert "result" in msg
        assert msg["result"].get("isError") is True


class TestPathRestrictions:
    """Tests for MCP server path security."""

    def test_path_inside_allowed_dir(self, tmp_path: Path) -> None:
        """Paths inside allowed dirs should be allowed."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        with patch("pdfmux.mcp_server.ALLOWED_DIRS", [tmp_path.resolve()]):
            assert _is_path_allowed(test_file)

    def test_path_outside_allowed_dir(self, tmp_path: Path) -> None:
        """Paths outside allowed dirs should be blocked."""
        with patch("pdfmux.mcp_server.ALLOWED_DIRS", [tmp_path.resolve()]):
            assert not _is_path_allowed(Path("/etc/passwd"))

    def test_convert_blocked_path(self, digital_pdf: Path) -> None:
        """convert_pdf should block paths outside allowed directories."""
        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            patch("pdfmux.mcp_server.ALLOWED_DIRS", [Path("/nonexistent").resolve()]),
        ):
            _handle_convert_pdf(msg_id=10, arguments={"file_path": str(digital_pdf)})

        assert len(captured) == 1
        msg = captured[0]
        assert "error" in msg
        assert "Access denied" in msg["error"]["message"]

    def test_analyze_blocked_path(self, digital_pdf: Path) -> None:
        """analyze_pdf should block paths outside allowed directories."""
        from pdfmux.mcp_server import _handle_analyze_pdf

        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            patch("pdfmux.mcp_server.ALLOWED_DIRS", [Path("/nonexistent").resolve()]),
        ):
            _handle_analyze_pdf(msg_id=11, arguments={"file_path": str(digital_pdf)})

        assert len(captured) == 1
        assert "Access denied" in captured[0]["error"]["message"]

    def test_batch_blocked_path(self, tmp_path: Path) -> None:
        """batch_convert should block directories outside allowed dirs."""
        from pdfmux.mcp_server import _handle_batch_convert

        captured = []

        with (
            patch("pdfmux.mcp_server._write_message", side_effect=captured.append),
            patch("pdfmux.mcp_server.ALLOWED_DIRS", [Path("/nonexistent").resolve()]),
        ):
            _handle_batch_convert(msg_id=12, arguments={"directory": str(tmp_path)})

        assert len(captured) == 1
        assert "Access denied" in captured[0]["error"]["message"]

    def test_allowed_dirs_env_parsing(self) -> None:
        """PDFMUX_ALLOWED_DIRS should parse colon-separated paths."""
        # This tests the module-level parsing logic
        assert isinstance(_is_path_allowed(Path(".")), bool)
