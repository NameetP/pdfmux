"""Tests for MCP server — path restrictions + tool functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pdfmux.mcp_server import (
    ALLOWED_DIRS,
    _is_path_allowed,
)


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

    def test_is_path_allowed_returns_bool(self) -> None:
        """_is_path_allowed should return a boolean."""
        assert isinstance(_is_path_allowed(Path(".")), bool)

    def test_allowed_dirs_is_list(self) -> None:
        """ALLOWED_DIRS should be a list of Path objects."""
        assert isinstance(ALLOWED_DIRS, list)
        for d in ALLOWED_DIRS:
            assert isinstance(d, Path)


class TestConvertPdfTool:
    """Tests for the convert_pdf MCP tool function."""

    def test_convert_pdf_exists(self) -> None:
        """convert_pdf should be importable."""
        from pdfmux.mcp_server import convert_pdf

        assert callable(convert_pdf)

    def test_convert_pdf_blocked_path(self) -> None:
        """convert_pdf should reject paths outside ALLOWED_DIRS."""
        import pytest

        from pdfmux.mcp_server import convert_pdf

        with patch("pdfmux.mcp_server.ALLOWED_DIRS", [Path("/nonexistent")]):
            with pytest.raises(ValueError, match="Access denied"):
                convert_pdf(file_path="/etc/passwd")

    def test_convert_pdf_with_valid_path(self, digital_pdf: Path) -> None:
        """convert_pdf should extract text from a valid PDF."""
        from pdfmux.mcp_server import convert_pdf

        with patch(
            "pdfmux.mcp_server.ALLOWED_DIRS",
            [digital_pdf.parent.resolve()],
        ):
            result = convert_pdf(file_path=str(digital_pdf))
            # Should return extracted text (not an error)
            assert "Access denied" not in result
            assert len(result) > 0


class TestAnalyzePdfTool:
    """Tests for the analyze_pdf MCP tool function."""

    def test_analyze_pdf_exists(self) -> None:
        """analyze_pdf should be importable."""
        from pdfmux.mcp_server import analyze_pdf

        assert callable(analyze_pdf)

    def test_analyze_pdf_blocked_path(self) -> None:
        """analyze_pdf should reject paths outside ALLOWED_DIRS."""
        import pytest

        from pdfmux.mcp_server import analyze_pdf

        with patch("pdfmux.mcp_server.ALLOWED_DIRS", [Path("/nonexistent")]):
            with pytest.raises(ValueError, match="Access denied"):
                analyze_pdf(file_path="/etc/passwd")
