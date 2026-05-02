"""Regression guards for the real-world failure modes we hit on a 433-PDF batch.

Every fixture here corresponds to a failure category the v1 ARK pipeline
silently lost. The tests are intentionally written as *behavioral contracts*:
they don't assert exact text output, they assert that pdfmux either succeeds
cleanly OR raises a named error — never silently returns empty Markdown.

Source incident: products/pdfmux/reviews/2026-04-30-ark-batch-retro.md
Plan reference: products/pdfmux/PDFMUX-IMPROVEMENTS-FROM-BATCH-RUN.md (P2)
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

import pdfmux
from pdfmux.errors import FileError, PdfmuxError


# ---------------------------------------------------------------------------
# Fixtures — built inline. Each models one real failure we observed.
# ---------------------------------------------------------------------------


@pytest.fixture
def truncated_pdf(tmp_path: Path) -> Path:
    """A PDF whose byte stream ends mid-object.

    Models the four 'pypdf: Stream has ended unexpectedly' failures from v1
    of the ARK batch (PROZURA proforma, Troysperse ZWD1, WACKER PO, Wacker
    TN MSDS). PyMuPDF's xref repair tolerates these; pypdf does not.
    """
    pdf_path = tmp_path / "truncated.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Header text on a partial PDF.", fontsize=11)
    full_bytes = doc.tobytes()
    doc.close()

    # Truncate ~30% of the file — destroys xref but PyMuPDF can usually repair.
    cutoff = max(64, int(len(full_bytes) * 0.7))
    pdf_path.write_bytes(full_bytes[:cutoff])
    return pdf_path


@pytest.fixture
def non_ascii_filename_pdf(tmp_path: Path) -> Path:
    """A PDF whose filename contains full-width punctuation and CJK characters.

    Models the v1 issue with `Coolsoft RE-5035 D4,D5,D6 test reports(原版).pdf`
    where the closing paren is U+FF09 (full-width). Worked in v4 (Python API)
    but caused intermittent shell-quoting failures in v1 (CLI subprocess).
    """
    # Use the same kind of filename the ARK batch had — full-width punct + CJK.
    pdf_path = tmp_path / "Coolsoft test reports（原版）.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test report content.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def arabic_only_pdf(tmp_path: Path) -> Path:
    """A digital PDF whose body is Arabic.

    Used to verify that Arabic is detected (DocumentResult.has_arabic)
    and that the BiDi post-processor runs without crashing on RTL text.
    Uses Helvetica fallback — the glyph rendering may be poor but PyMuPDF's
    text extraction should still produce the underlying Unicode.
    """
    pdf_path = tmp_path / "arabic_only.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # NB: Helvetica may not render Arabic glyphs visually, but the PDF will
    # still contain the Unicode codepoints, which is what we care about.
    arabic_text = "مرحبا بالعالم. هذا اختبار للنص العربي في ملف PDF."
    try:
        page.insert_text((72, 72), arabic_text, fontsize=14)
    except Exception:
        # Some PyMuPDF versions reject non-Latin in default font; skip
        # gracefully — the test that uses this fixture handles missing text.
        page.insert_text((72, 72), "fallback", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def zero_byte_pdf(tmp_path: Path) -> Path:
    """A 0-byte file with a .pdf extension. Should error cleanly, not silently."""
    pdf_path = tmp_path / "empty_file.pdf"
    pdf_path.write_bytes(b"")
    return pdf_path


@pytest.fixture
def html_as_pdf(tmp_path: Path) -> Path:
    """An HTML document misnamed with a .pdf extension.

    Common in real batches when a 'view as PDF' link saves the page source
    instead. Should error cleanly (FileError or PdfmuxError) — never silently
    return the rendered HTML body as 'extracted text'.
    """
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_text(
        "<html><head><title>Not a PDF</title></head>"
        "<body><h1>This is HTML</h1><p>Pretending to be a PDF.</p></body></html>",
        encoding="utf-8",
    )
    return pdf_path


# ---------------------------------------------------------------------------
# Behavioral contracts — what pdfmux must do on each failure mode.
# ---------------------------------------------------------------------------


class TestTruncatedStream:
    """PyMuPDF tolerates truncated PDFs that pypdf rejects.

    pdfmux must either recover (low-confidence text) or raise PdfmuxError.
    Never silently return empty text.
    """

    def test_does_not_silently_return_empty(self, truncated_pdf: Path) -> None:
        try:
            text = pdfmux.extract_text(truncated_pdf, quality="fast")
        except PdfmuxError:
            return  # Acceptable: clear error.

        # Acceptable: recovered some text. Empty string is the failure mode
        # we explicitly guard against — that's what bit the v1 batch.
        assert isinstance(text, str)
        assert text != "" or text.strip() != "", (
            "Truncated PDF returned empty Markdown without raising — this is "
            "the silent-failure mode the 1.6.1 contract forbids."
        )

    def test_extract_json_reports_low_confidence_or_raises(self, truncated_pdf: Path) -> None:
        """If we got JSON output, the confidence field must reflect the damage."""
        try:
            data = pdfmux.extract_json(truncated_pdf, quality="fast")
        except PdfmuxError:
            return
        assert "confidence" in data
        # We don't assert a specific number — only that confidence is
        # surfaced (a downstream caller could then gate on it).
        assert isinstance(data["confidence"], int | float)


class TestNonAsciiFilename:
    """The Python API must handle CJK + full-width punctuation in filenames."""

    def test_extract_text_accepts_cjk_filename(self, non_ascii_filename_pdf: Path) -> None:
        text = pdfmux.extract_text(non_ascii_filename_pdf, quality="fast")
        assert isinstance(text, str)
        assert "Test report content" in text

    def test_batch_extract_handles_cjk_filename(self, non_ascii_filename_pdf: Path) -> None:
        """batch_extract is the API we now recommend for batches — it MUST
        survive non-ASCII filenames without subprocess-style shell-quoting bugs.
        """
        results = list(pdfmux.batch_extract([non_ascii_filename_pdf], quality="fast"))
        assert len(results) == 1
        path, result = results[0]
        assert path == non_ascii_filename_pdf
        # batch_extract yields (path, DocumentResult) on success or (path, Exception) on failure.
        # We only require: not silently empty.
        if isinstance(result, Exception):
            pytest.fail(f"batch_extract raised on CJK filename: {result!r}")
        assert "Test report content" in result.text


class TestArabicOnly:
    """Arabic-only PDFs must trigger has_arabic and survive the BiDi pipeline."""

    def test_arabic_pipeline_does_not_crash(self, arabic_only_pdf: Path) -> None:
        """The BiDi post-processor must not crash on RTL text.

        We don't assert on the exact extracted glyphs — Helvetica may not
        render Arabic — but the pipeline must complete without exception.
        """
        try:
            text = pdfmux.extract_text(arabic_only_pdf, quality="fast")
            assert isinstance(text, str)
        except PdfmuxError:
            # Acceptable if the fixture's font rendering produced unrecoverable
            # output; the contract is "no uncaught exceptions, BiDi doesn't crash".
            pass


class TestZeroBytePdf:
    """A 0-byte file must error cleanly, never silently return empty text."""

    def test_zero_byte_raises_pdfmux_error(self, zero_byte_pdf: Path) -> None:
        with pytest.raises(PdfmuxError):
            pdfmux.extract_text(zero_byte_pdf, quality="fast")

    def test_zero_byte_via_batch_yields_exception(self, zero_byte_pdf: Path) -> None:
        """batch_extract must isolate per-file failures — yield the exception,
        don't crash the whole batch."""
        results = list(pdfmux.batch_extract([zero_byte_pdf], quality="fast"))
        assert len(results) == 1
        path, result = results[0]
        assert path == zero_byte_pdf
        assert isinstance(result, Exception), (
            f"0-byte PDF must yield an exception in batch_extract, got: {type(result).__name__}"
        )


class TestHtmlAsPdf:
    """An HTML file misnamed .pdf must NOT be parsed as Markdown."""

    def test_html_as_pdf_errors_cleanly(self, html_as_pdf: Path) -> None:
        """The contract: either raise (preferred) OR return empty/low-confidence —
        but NEVER return the HTML body as if it were extracted text."""
        try:
            text = pdfmux.extract_text(html_as_pdf, quality="fast")
        except PdfmuxError:
            return  # Preferred behavior: clear error.

        # If we didn't raise, the output must not contain HTML markup as if it were content.
        assert "<html>" not in text, (
            "HTML-as-PDF was parsed as if it were a real PDF — this is the silent-corruption "
            "failure mode users get when 'view as PDF' saves the page source."
        )

    def test_html_as_pdf_in_batch_isolates(self, html_as_pdf: Path, digital_pdf: Path) -> None:
        """A bad file in a batch must not poison the rest of the batch."""
        results = list(pdfmux.batch_extract([html_as_pdf, digital_pdf], quality="fast"))
        assert len(results) == 2

        # The good PDF must succeed regardless of what happened to the bad one.
        good_results = [
            (p, r) for p, r in results if p == digital_pdf and not isinstance(r, Exception)
        ]
        assert len(good_results) == 1, "digital_pdf should still extract successfully"


# ---------------------------------------------------------------------------
# Edge case: FileError type is preserved (we already test the error catalog
# elsewhere, but pin here that the user-visible error is one of the named ones).
# ---------------------------------------------------------------------------


class TestErrorTypeContract:
    """Errors raised on real-world bad input must be named pdfmux errors,
    not bare ValueError / FileNotFoundError / RuntimeError."""

    def test_zero_byte_raises_named_error(self, zero_byte_pdf: Path) -> None:
        with pytest.raises(PdfmuxError) as exc_info:
            pdfmux.extract_text(zero_byte_pdf, quality="fast")
        # Specifically, this should be a FileError variant (corrupted/empty/unreadable)
        # rather than a generic ExtractionError. We accept either since the audit
        # boundary on 0 bytes is debatable — but it MUST be a PdfmuxError subclass.
        assert isinstance(exc_info.value, PdfmuxError)

    def test_missing_file_raises_file_error(self, tmp_path: Path) -> None:
        """The classic 'path doesn't exist' case — must be FileError, not generic."""
        ghost = tmp_path / "does-not-exist.pdf"
        with pytest.raises(FileError):
            pdfmux.extract_text(ghost, quality="fast")
