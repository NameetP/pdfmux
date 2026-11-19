"""Regression tests for GHSA-332c-cq58-rx3x.

Every entry point that walks a PDF's pages must enforce the size and page
limits BEFORE classify()/audit_document() run, and the MCP triage tools must
also run under the hard timeout. Before the fix, get_pdf_metadata, analyze_pdf,
extract_streaming and verify_extraction(extracted_text=...) skipped all three,
and process() checked the page count only after classify() had already walked
every page.

The "must not be called" assertions are the point: a guard that rejects the
file only after the expensive walk does not prevent the denial of service.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from pdfmux.errors import FileError, OCRTimeoutError


def _boom(*_a, **_k):
    raise AssertionError("per-page work ran before the resource guard")


def _slow_classify(path: str):
    time.sleep(5)


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Resource limit test page {i + 1}.")
    path = tmp_path / "three.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def allow_tmp(tmp_path: Path):
    """Point the MCP path sandbox at tmp_path for the duration of a test."""
    import pdfmux.mcp_server as mcp_mod
    import pdfmux.path_safety as ps

    saved_ps, saved_mcp = ps.ALLOWED_DIRS, getattr(mcp_mod, "ALLOWED_DIRS", None)
    ps.ALLOWED_DIRS = [tmp_path.resolve()]
    if saved_mcp is not None:
        mcp_mod.ALLOWED_DIRS = ps.ALLOWED_DIRS
    try:
        yield
    finally:
        ps.ALLOWED_DIRS = saved_ps
        if saved_mcp is not None:
            mcp_mod.ALLOWED_DIRS = saved_mcp


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


class TestCheckResourceLimits:
    def test_returns_page_count_within_limits(self, three_page_pdf: Path) -> None:
        from pdfmux.pipeline import check_resource_limits

        assert check_resource_limits(three_page_pdf) == 3

    def test_rejects_too_many_pages(self, three_page_pdf: Path) -> None:
        from pdfmux.pipeline import check_resource_limits

        with patch("pdfmux.pipeline.MAX_PAGE_COUNT", 2):
            with pytest.raises(FileError, match="Too many pages") as exc:
                check_resource_limits(three_page_pdf)
        assert exc.value.code == "PDF_TOO_LARGE"

    def test_rejects_too_large(self, three_page_pdf: Path) -> None:
        from pdfmux.pipeline import check_resource_limits

        with patch("pdfmux.pipeline.MAX_FILE_SIZE_MB", 0):
            with pytest.raises(FileError, match="File too large") as exc:
                check_resource_limits(three_page_pdf)
        assert exc.value.code == "PDF_TOO_LARGE"

    def test_missing_and_non_pdf_are_left_to_caller(self, tmp_path: Path) -> None:
        from pdfmux.pipeline import check_resource_limits

        assert check_resource_limits(tmp_path / "nope.pdf") == 0
        junk = tmp_path / "junk.pdf"
        junk.write_bytes(b"not a pdf at all")
        assert check_resource_limits(junk) == 0

    def test_is_cheap_on_a_large_page_count(self, tmp_path: Path) -> None:
        """The guard reads the page tree only, so it must stay fast at scale."""
        import pymupdf

        from pdfmux.pipeline import check_resource_limits

        doc = pymupdf.open()
        for _ in range(3000):
            doc.new_page()
        path = tmp_path / "many.pdf"
        doc.save(str(path))
        doc.close()

        t0 = time.perf_counter()
        assert check_resource_limits(path) == 3000
        assert time.perf_counter() - t0 < 1.0


# ---------------------------------------------------------------------------
# process(): page check must now precede classify()
# ---------------------------------------------------------------------------


def test_process_rejects_before_classify(three_page_pdf: Path) -> None:
    from pdfmux.pipeline import process

    with patch("pdfmux.pipeline.MAX_PAGE_COUNT", 2), patch("pdfmux.pipeline.classify", _boom):
        with pytest.raises(FileError, match="Too many pages"):
            process(three_page_pdf, use_cache=False)


# ---------------------------------------------------------------------------
# Every MCP entry point that bypasses process()
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("allow_tmp")
class TestMcpTriageToolsAreBounded:
    def test_get_pdf_metadata_page_cap(self, three_page_pdf: Path) -> None:
        from pdfmux.mcp_server import get_pdf_metadata

        with patch("pdfmux.pipeline.MAX_PAGE_COUNT", 2), patch("pdfmux.detect.classify", _boom):
            with pytest.raises(FileError, match="Too many pages"):
                get_pdf_metadata(str(three_page_pdf))

    def test_get_pdf_metadata_size_cap(self, three_page_pdf: Path) -> None:
        from pdfmux.mcp_server import get_pdf_metadata

        with patch("pdfmux.pipeline.MAX_FILE_SIZE_MB", 0), patch("pdfmux.detect.classify", _boom):
            with pytest.raises(FileError, match="File too large"):
                get_pdf_metadata(str(three_page_pdf))

    def test_analyze_pdf_page_cap(self, three_page_pdf: Path) -> None:
        from pdfmux.mcp_server import analyze_pdf

        with (
            patch("pdfmux.pipeline.MAX_PAGE_COUNT", 2),
            patch("pdfmux.detect.classify", _boom),
            patch("pdfmux.audit.audit_document", _boom),
        ):
            with pytest.raises(FileError, match="Too many pages"):
                analyze_pdf(str(three_page_pdf))

    def test_analyze_pdf_missing_file(self, tmp_path: Path) -> None:
        from pdfmux.mcp_server import analyze_pdf

        with pytest.raises(ValueError, match="File not found"):
            analyze_pdf(str(tmp_path / "missing.pdf"))

    def test_extract_streaming_page_cap(self, three_page_pdf: Path) -> None:
        from pdfmux.mcp_server import extract_streaming

        with (
            patch("pdfmux.pipeline.MAX_PAGE_COUNT", 2),
            patch("pdfmux.streaming.classify", _boom),
            patch("pdfmux.streaming.audit_document", _boom),
        ):
            with pytest.raises(FileError, match="Too many pages"):
                extract_streaming(str(three_page_pdf))

    def test_verify_extraction_with_external_text_page_cap(self, three_page_pdf: Path) -> None:
        from pdfmux.mcp_server import verify_extraction

        with (
            patch("pdfmux.pipeline.MAX_PAGE_COUNT", 2),
            patch("pdfmux.verifier.audit_document", _boom),
        ):
            with pytest.raises(FileError, match="Too many pages"):
                verify_extraction(str(three_page_pdf), extracted_text="some external text")

    def test_triage_runs_under_hard_timeout(self, three_page_pdf: Path, monkeypatch) -> None:
        """A wedged classify() must return control at the deadline, not hang."""
        import pdfmux.mcp_server as mcp_mod

        monkeypatch.setenv("PDFMUX_TIMEOUT_ISOLATION", "thread")  # no pickling of the stub
        monkeypatch.setattr(mcp_mod, "_classify", _slow_classify)
        with patch("pdfmux.pipeline.EXTRACTION_TIMEOUT_S", 1):
            t0 = time.perf_counter()
            with pytest.raises(OCRTimeoutError, match="timed out"):
                mcp_mod.get_pdf_metadata(str(three_page_pdf))
            assert time.perf_counter() - t0 < 4

    def test_normal_pdf_still_works(self, three_page_pdf: Path) -> None:
        """Default limits and the default (process) isolation backend: real results."""
        import json

        from pdfmux.mcp_server import analyze_pdf, get_pdf_metadata

        meta = json.loads(get_pdf_metadata(str(three_page_pdf)))
        assert meta["page_count"] == 3
        assert "3" in analyze_pdf(str(three_page_pdf))
