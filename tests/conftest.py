"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import pytest

# Generous, but finite. Tests that drive the CLI run the real extraction path
# (`pdfmux analyze` calls `process(quality="standard")`, which loads Docling /
# Marker / opendataloader), so on a contended machine they can take minutes.
# A genuinely wedged extractor is still caught: test_timeout_isolation.py
# overrides this with its own sub-second value to exercise that path.
TEST_EXTRACTION_TIMEOUT_S = 1800


@pytest.fixture(autouse=True)
def _pin_extraction_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the suite inheriting a wall-clock dependency from the environment.

    ``pipeline.EXTRACTION_TIMEOUT_S`` is read from ``PDFMUX_TIMEOUT`` **at import
    time** and defaults to 300s. That made every CLI test implicitly a
    performance test: on 2026-07-27 two full-suite runs of byte-identical code
    disagreed (762 passed vs 2 failed) purely because the machine was 6-8x
    contended and extraction crept past the deadline.

    Note this patches the module attribute, not the environment variable.
    Setting ``PDFMUX_TIMEOUT`` here would do nothing at all — the constant has
    already been computed by the time any test runs, so the env var only has
    effect if it is set before the process starts.
    """
    from pdfmux import pipeline

    monkeypatch.setattr(pipeline, "EXTRACTION_TIMEOUT_S", TEST_EXTRACTION_TIMEOUT_S)


def assert_cli_ok(result: Any, *, context: str = "") -> None:
    """Assert a Typer ``CliRunner`` result succeeded, and say why if it did not.

    ``assert result.exit_code == 0`` on its own reports a bare ``AssertionError:
    assert 1 == 0``, which cannot distinguish "the command is broken" from "the
    extraction timed out on a busy machine". That ambiguity is what made the
    2026-07-27 flake expensive to diagnose, so surface the output and the
    exception, and name the timeout explicitly when it is the cause.
    """
    if result.exit_code == 0:
        return

    from pdfmux import pipeline
    from pdfmux.errors import OCRTimeoutError

    exc = result.exception
    parts = [f"CLI exited {result.exit_code}"]
    if context:
        parts.append(f" ({context})")
    if isinstance(exc, OCRTimeoutError):
        # Report the value actually in force, not the constant — a test may
        # have overridden it, and saying "1800s" when it was 1s misleads.
        parts.append(
            f"\n\nTIMED OUT — this is an environment problem, not a code failure."
            f"\nExtraction exceeded EXTRACTION_TIMEOUT_S"
            f" (in force: {pipeline.EXTRACTION_TIMEOUT_S}s;"
            f" tests pin {TEST_EXTRACTION_TIMEOUT_S}s unless overridden)."
            f"\nIf this fires at the pinned value, the machine is heavily"
            f" loaded or an extractor wedged."
        )
    if exc is not None:
        parts.append(f"\n\nexception: {type(exc).__name__}: {exc}")
    parts.append(f"\n\n--- CLI output ---\n{result.output}")

    raise AssertionError("".join(parts))


@pytest.fixture
def digital_pdf(tmp_path: Path) -> Path:
    """Create a simple digital PDF for testing."""
    pdf_path = tmp_path / "digital_simple.pdf"
    doc = fitz.open()

    # Page 1: Simple text
    page = doc.new_page()
    text = (
        "# Introduction\n\n"
        "This is a sample PDF document created for testing purposes. "
        "It contains multiple paragraphs of text that should be "
        "extractable by the fast PyMuPDF extractor.\n\n"
        "## Section 1\n\n"
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
    )
    page.insert_text((72, 72), text, fontsize=11)

    # Page 2: More text
    page2 = doc.new_page()
    text2 = (
        "## Section 2\n\n"
        "This is the second page of our test document. "
        "It demonstrates that the extractor can handle multi-page PDFs "
        "and maintain structure across pages.\n\n"
        "- Item one\n"
        "- Item two\n"
        "- Item three"
    )
    page2.insert_text((72, 72), text2, fontsize=11)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create an empty PDF for edge case testing."""
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    """Create a 5-page digital PDF."""
    pdf_path = tmp_path / "multi_page.pdf"
    doc = fitz.open()

    for i in range(5):
        page = doc.new_page()
        text = (
            f"Page {i + 1} of 5\n\n"
            f"This is content on page {i + 1}. It contains enough text "
            f"to be classified as a digital page by our detection logic. "
            f"The quick brown fox jumps over the lazy dog. "
            f"Pack my box with five dozen liquor jugs."
        )
        page.insert_text((72, 72), text, fontsize=11)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
