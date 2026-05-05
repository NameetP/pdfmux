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
