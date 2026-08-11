"""Tests for MCP server — enhanced tools (batch_convert, extract_structured)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class TestBatchConvertTool:
    """Tests for the batch_convert MCP tool."""

    def test_batch_convert_exists(self) -> None:
        """batch_convert should be importable."""
        from pdfmux.mcp_server import batch_convert

        assert callable(batch_convert)

    def test_batch_convert_blocked_path(self) -> None:
        """batch_convert should reject dirs outside ALLOWED_DIRS."""
        import pytest

        from pdfmux.mcp_server import batch_convert

        with patch("pdfmux.path_safety.ALLOWED_DIRS", [Path("/nonexistent")]):
            with pytest.raises(ValueError, match="Access denied"):
                batch_convert(directory="/tmp")


class TestExtractStructuredTool:
    """Tests for the extract_structured MCP tool."""

    def test_extract_structured_exists(self) -> None:
        """extract_structured should be importable."""
        from pdfmux.mcp_server import extract_structured

        assert callable(extract_structured)

    def test_extract_structured_blocked_path(self) -> None:
        """extract_structured should reject paths outside ALLOWED_DIRS."""
        import pytest

        from pdfmux.mcp_server import extract_structured

        with patch("pdfmux.path_safety.ALLOWED_DIRS", [Path("/nonexistent")]):
            with pytest.raises(ValueError, match="Access denied"):
                extract_structured(file_path="/etc/passwd")


class TestMCPServerSetup:
    """Tests for MCP server configuration."""

    def test_run_server_exists(self) -> None:
        """run_server should be importable."""
        from pdfmux.mcp_server import run_server

        assert callable(run_server)

    def test_run_http_server_exists(self) -> None:
        """run_http_server should be importable."""
        from pdfmux.mcp_server import run_http_server

        assert callable(run_http_server)

    def test_server_uses_the_mcp_2x_import_surface(self) -> None:
        """The server instance must be an mcp 2.x ``MCPServer``.

        Dependency-facing guard. mcp 2.0.0 deleted ``mcp.server.fastmcp``; the
        1.x-only fix was an upper-bound pin (commit 6a4f78a) that re-breaks on
        every bump. If this fails with ImportError on ``mcp.server.mcpserver``,
        the installed ``mcp`` is <2.0.0 and the ``[serve]`` extra's floor was
        violated — do not re-pin, port the import.
        """
        from mcp.server.mcpserver import MCPServer

        from pdfmux import mcp_extract, mcp_server

        assert isinstance(mcp_server.mcp, MCPServer)
        assert isinstance(mcp_extract.mcp, MCPServer)

    def test_run_http_server_passes_host_and_port_to_run(self) -> None:
        """Bind address must reach ``run()``, not ``mcp.settings``.

        mcp 2.x moved host/port from the ``Settings`` model onto the transport
        kwargs of ``run()``; ``mcp.settings.host = ...`` now raises ValueError.
        Asserts the decision deterministically instead of binding a real socket,
        so a silent regression to the settings form fails here rather than at
        first HTTP launch.
        """
        from pdfmux import mcp_server

        with patch.object(mcp_server.mcp, "run") as mock_run:
            mcp_server.run_http_server(host="0.0.0.0", port=9123)

        mock_run.assert_called_once_with(transport="streamable-http", host="0.0.0.0", port=9123)

    def test_run_http_server_defaults_to_loopback(self) -> None:
        """No explicit host → loopback, still delivered via run() kwargs."""
        from pdfmux import mcp_server

        with patch.dict("os.environ", {}, clear=False) as _env:
            _env.pop("PDFMUX_HTTP_HOST", None)
            with patch.object(mcp_server.mcp, "run") as mock_run:
                mcp_server.run_http_server()

        mock_run.assert_called_once_with(transport="streamable-http", host="127.0.0.1", port=8000)
