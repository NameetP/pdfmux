"""Tests for the MCP server tools."""

import json
import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal valid PDF for testing."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Hello, pdfmux MCP server test.", fontsize=12)
    page.insert_text((72, 100), "This is a test document with two lines.", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_pdf_with_table(tmp_path):
    """Create a PDF with a simple table for structured extraction testing."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Name    Age    City", fontsize=12)
    page.insert_text((72, 92), "Alice   30     NYC", fontsize=12)
    page.insert_text((72, 112), "Bob     25     LA", fontsize=12)
    pdf_path = tmp_path / "table_test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture(autouse=True)
def allow_tmp_dirs(tmp_path):
    """Allow access to tmp directories for all tests."""
    with patch.dict(os.environ, {"PDFMUX_ALLOWED_DIRS": str(tmp_path)}):
        # Re-import to pick up new ALLOWED_DIRS
        import pdfmux.mcp_server as mod

        mod.ALLOWED_DIRS = [tmp_path.resolve()]
        yield


# ---------------------------------------------------------------------------
# Path security tests
# ---------------------------------------------------------------------------


class TestPathSecurity:
    def test_allowed_path(self, tmp_path):
        from pdfmux.mcp_server import _is_path_allowed

        test_file = tmp_path / "test.pdf"
        test_file.touch()
        assert _is_path_allowed(test_file)

    def test_disallowed_path(self, tmp_path):
        from pdfmux.mcp_server import _check_path

        with pytest.raises(ValueError, match="Access denied"):
            _check_path("/etc/passwd")

    def test_empty_path_raises(self):
        from pdfmux.mcp_server import _check_path

        with pytest.raises(ValueError, match="required"):
            _check_path("")

    def test_path_traversal_blocked(self, tmp_path):
        from pdfmux.mcp_server import _check_path

        with pytest.raises(ValueError, match="Access denied"):
            _check_path(str(tmp_path / ".." / ".." / "etc" / "passwd"))


# ---------------------------------------------------------------------------
# get_pdf_metadata tests
# ---------------------------------------------------------------------------


class TestGetPdfMetadata:
    def test_basic_metadata(self, sample_pdf):
        from pdfmux.mcp_server import get_pdf_metadata

        result = json.loads(get_pdf_metadata(str(sample_pdf)))

        assert result["page_count"] == 1
        assert result["file_size_bytes"] > 0
        assert "file_size_human" in result
        assert "digital" in result["detected_types"]
        assert isinstance(result["has_tables"], bool)
        assert result["recommended_tool"] in ("convert_pdf", "extract_structured")
        assert result["recommended_quality"] in ("standard", "high")

    def test_file_not_found(self, tmp_path):
        from pdfmux.mcp_server import get_pdf_metadata

        with pytest.raises(ValueError, match="File not found"):
            get_pdf_metadata(str(tmp_path / "nonexistent.pdf"))

    def test_access_denied(self):
        from pdfmux.mcp_server import get_pdf_metadata

        with pytest.raises(ValueError, match="Access denied"):
            get_pdf_metadata("/etc/passwd")


# ---------------------------------------------------------------------------
# convert_pdf tests
# ---------------------------------------------------------------------------


class TestConvertPdf:
    def test_basic_conversion(self, sample_pdf):
        from pdfmux.mcp_server import convert_pdf

        result = convert_pdf(str(sample_pdf))

        assert "Hello" in result
        assert "pdfmux" in result
        assert isinstance(result, str)

    def test_json_format(self, sample_pdf):
        from pdfmux.mcp_server import extract_structured

        result = extract_structured(str(sample_pdf))
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_quality_fast(self, sample_pdf):
        from pdfmux.mcp_server import convert_pdf

        result = convert_pdf(str(sample_pdf), quality="fast")
        assert "Hello" in result

    def test_access_denied(self):
        from pdfmux.mcp_server import convert_pdf

        with pytest.raises(ValueError, match="Access denied"):
            convert_pdf("/etc/passwd")


# ---------------------------------------------------------------------------
# analyze_pdf tests
# ---------------------------------------------------------------------------


class TestAnalyzePdf:
    def test_basic_analysis(self, sample_pdf):
        from pdfmux.mcp_server import analyze_pdf

        result = json.loads(analyze_pdf(str(sample_pdf)))

        assert result["page_count"] == 1
        assert "detected_types" in result
        assert isinstance(result["needs_ocr"], bool)
        assert isinstance(result["pages"], list)
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page"] == 1

    def test_access_denied(self):
        from pdfmux.mcp_server import analyze_pdf

        with pytest.raises(ValueError, match="Access denied"):
            analyze_pdf("/etc/passwd")


# ---------------------------------------------------------------------------
# batch_convert tests
# ---------------------------------------------------------------------------


class TestBatchConvert:
    def test_batch_directory(self, sample_pdf, tmp_path):
        from pdfmux.mcp_server import batch_convert

        result = json.loads(batch_convert(str(tmp_path)))

        assert result["total_files"] >= 1
        assert result["success"] >= 1
        assert isinstance(result["results"], list)

    def test_empty_directory(self, tmp_path):
        from pdfmux.mcp_server import batch_convert

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        # Need to allow this path
        import pdfmux.mcp_server as mod

        mod.ALLOWED_DIRS.append(empty_dir.resolve())
        result = batch_convert(str(empty_dir))
        assert "No PDF files found" in result

    def test_not_a_directory(self, sample_pdf):
        from pdfmux.mcp_server import batch_convert

        with pytest.raises(ValueError, match="Not a directory"):
            batch_convert(str(sample_pdf))


# ---------------------------------------------------------------------------
# extract_structured tests
# ---------------------------------------------------------------------------


class TestExtractStructured:
    def test_basic_structured(self, sample_pdf):
        from pdfmux.mcp_server import extract_structured

        result = extract_structured(str(sample_pdf))
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_access_denied(self):
        from pdfmux.mcp_server import extract_structured

        with pytest.raises(ValueError, match="Access denied"):
            extract_structured("/etc/passwd")


# ---------------------------------------------------------------------------
# Server setup tests
# ---------------------------------------------------------------------------


class TestServerSetup:
    def test_mcp_instance_exists(self):
        from pdfmux.mcp_server import mcp

        assert mcp is not None
        assert mcp.name == "pdfmux"

    def test_tools_registered(self):
        from pdfmux.mcp_server import mcp

        # The FastMCP instance should have our tools
        tool_names = {name for name in mcp._tool_manager._tools}
        assert "convert_pdf" in tool_names
        assert "analyze_pdf" in tool_names
        assert "batch_convert" in tool_names
        assert "extract_structured" in tool_names
        assert "get_pdf_metadata" in tool_names
