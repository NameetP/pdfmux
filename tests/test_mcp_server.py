"""Tests for the MCP server tools."""

import json
import os
import tempfile
from pathlib import Path
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
        # ALLOWED_DIRS is module-level, evaluated at first import. We override
        # both the canonical binding (path_safety) and the re-exported alias
        # (mcp_server) so existing tests that read either keep working.
        import pdfmux.mcp_server as mod
        import pdfmux.path_safety as path_safety_mod

        original_path_safety = path_safety_mod.ALLOWED_DIRS
        original_mcp = mod.ALLOWED_DIRS
        path_safety_mod.ALLOWED_DIRS = [tmp_path.resolve()]
        mod.ALLOWED_DIRS = path_safety_mod.ALLOWED_DIRS
        try:
            yield
        finally:
            path_safety_mod.ALLOWED_DIRS = original_path_safety
            mod.ALLOWED_DIRS = original_mcp


# ---------------------------------------------------------------------------
# Path security tests
# ---------------------------------------------------------------------------


class TestPathSecurity:
    def test_allowed_path(self, tmp_path):
        from pdfmux.mcp_server import _is_path_allowed, ALLOWED_DIRS

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
        # Need to allow this path — append to the canonical binding so the
        # _is_path_allowed check (which lives in path_safety) sees it.
        import pdfmux.path_safety as path_safety_mod

        path_safety_mod.ALLOWED_DIRS.append(empty_dir.resolve())
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
        assert "extract_streaming" in tool_names
        assert "get_pdf_metadata" in tool_names
        assert "verify_extraction" in tool_names


# ---------------------------------------------------------------------------
# verify_extraction tests (Certify Anything — the 7th tool)
# ---------------------------------------------------------------------------


def _page1_lines() -> list[str]:
    # Distinct vocabulary from page 2, and comfortably over the 200-char
    # GOOD_TEXT_THRESHOLD so a dropped page reads as a genuine silent drop.
    return [
        f"Q1 revenue recognition and deferred income schedule row {i} totalling material amounts."
        for i in range(6)
    ]


def _page2_lines() -> list[str]:
    return [
        f"Appendix regulatory compliance attestation signature counterparty disclosure item {i}."
        for i in range(6)
    ]


@pytest.fixture
def two_page_pdf(tmp_path):
    """A 2-page PDF whose pages carry distinct, >200-char content each."""
    import fitz

    doc = fitz.open()
    for lines in (_page1_lines(), _page2_lines()):
        page = doc.new_page(width=612, height=792)
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 20
    pdf_path = tmp_path / "twopage.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestVerifyExtraction:
    _STATUSES = {"usable", "silently-empty", "recovered", "review", "fail", "unverifiable"}

    def test_self_audit_shape(self, sample_pdf):
        """No extraction supplied → pdfmux certifies its own read; assert shape."""
        from pdfmux.mcp_server import verify_extraction

        report = json.loads(verify_extraction(str(sample_pdf)))

        assert report["engine"] == "pdfmux"
        assert report["verdict"] in ("PASS", "REVIEW", "FAIL")
        assert report["page_count"] == 1
        assert "silently dropped" in report["headline"]
        assert isinstance(report["silently_dropped_pages"], list)
        assert isinstance(report["pages_silently_dropped"], int)
        assert isinstance(report["pages"], list) and len(report["pages"]) == 1
        page = report["pages"][0]
        assert page["page"] == 1
        assert page["status"] in self._STATUSES
        assert set(page) >= {"page", "status", "coverage", "confidence", "alignment"}
        assert isinstance(report["summary"], str) and report["summary"]
        assert report["signature"].startswith("sha256:")
        assert isinstance(report["limitations"], list) and report["limitations"]

    def test_detects_silent_drop(self, two_page_pdf):
        """A page-aligned extraction that keeps page 1 but drops page 2."""
        from pdfmux.mcp_server import verify_extraction

        extracted = json.dumps(
            [
                {"page": 1, "text": "\n".join(_page1_lines())},
                {"page": 2, "text": ""},
            ]
        )
        report = json.loads(
            verify_extraction(str(two_page_pdf), extracted_text=extracted, engine="reducto")
        )

        assert report["engine"] == "reducto"
        assert report["page_count"] == 2
        assert report["pages_silently_dropped"] >= 1
        assert 2 in report["silently_dropped_pages"]
        page2 = next(p for p in report["pages"] if p["page"] == 2)
        assert page2["status"] == "silently-empty"
        assert report["verdict"] == "FAIL"
        assert report["headline"].startswith(str(report["pages_silently_dropped"]))

    def test_clean_extraction_no_drops(self, two_page_pdf):
        """Both pages present → nothing silently dropped."""
        from pdfmux.mcp_server import verify_extraction

        extracted = json.dumps(
            [
                {"page": 1, "text": "\n".join(_page1_lines())},
                {"page": 2, "text": "\n".join(_page2_lines())},
            ]
        )
        report = json.loads(
            verify_extraction(str(two_page_pdf), extracted_text=extracted, engine="llamaparse")
        )

        assert report["engine"] == "llamaparse"
        assert report["page_count"] == 2
        assert report["pages_silently_dropped"] == 0
        assert report["silently_dropped_pages"] == []

    def test_access_denied(self):
        from pdfmux.mcp_server import verify_extraction

        with pytest.raises(ValueError, match="Access denied"):
            verify_extraction("/etc/passwd")

    def test_file_not_found(self, tmp_path):
        from pdfmux.mcp_server import verify_extraction

        with pytest.raises(ValueError, match="File not found"):
            verify_extraction(str(tmp_path / "nonexistent.pdf"))
