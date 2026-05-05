"""Tests for the `pdfmux audit` CLI command (1.6.4).

Behavior under test:
- Reads --against from CSV (filename,text) or JSON ({filename: text})
- Runs pdfmux on every overlapping PDF
- Computes per-document word-set Jaccard overlap
- Flags docs with overlap < threshold OR confidence < threshold
- Writes a CSV with the documented column set
- Exits 3 if anything was flagged, 0 if all clean, 2 on usage errors
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz
import pytest
from typer.testing import CliRunner

from pdfmux.cli import app

runner = CliRunner()


def _make_pdf(path: Path, body: str = "Hello world this is a test document.") -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), body, fontsize=11)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """Two simple digital PDFs."""
    d = tmp_path / "pdfs"
    d.mkdir()
    _make_pdf(d / "alpha.pdf", "Alpha document about chemistry safety.")
    _make_pdf(d / "beta.pdf", "Beta document about industrial regulation.")
    return d


@pytest.fixture
def matching_csv(tmp_path: Path) -> Path:
    """Other-extractor output that agrees with pdfmux — same word set."""
    csv_path = tmp_path / "other.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text"])
        # Word-for-word match with PDF bodies (built in _make_pdf via pdf_dir
        # fixture) so Jaccard = 1.0
        w.writerow(["alpha.pdf", "Alpha document about chemistry safety."])
        w.writerow(["beta.pdf", "Beta document about industrial regulation."])
    return csv_path


@pytest.fixture
def disagreeing_csv(tmp_path: Path) -> Path:
    """Other-extractor output that disagrees with pdfmux."""
    csv_path = tmp_path / "other.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text"])
        # Completely different words → low Jaccard → flag
        w.writerow(["alpha.pdf", "totally unrelated stub from a broken extractor"])
        w.writerow(["beta.pdf", "garbage output completely different content"])
    return csv_path


@pytest.fixture
def disagreeing_json(tmp_path: Path) -> Path:
    json_path = tmp_path / "other.json"
    json_path.write_text(
        json.dumps(
            {
                "alpha.pdf": "totally unrelated stub from a broken extractor",
                "beta.pdf": "garbage output completely different content",
            }
        ),
        encoding="utf-8",
    )
    return json_path


class TestAuditCSV:
    def test_csv_input_high_overlap_no_flag(
        self, pdf_dir: Path, matching_csv: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "audit.csv"
        result = runner.invoke(
            app,
            [
                "audit",
                "--against",
                str(matching_csv),
                "--on",
                str(pdf_dir),
                "--output",
                str(out),
                "--quality",
                "fast",
            ],
        )
        # Should NOT exit 3 — overlap is high enough to mark "ok"
        assert result.exit_code == 0, f"got {result.exit_code}; stdout={result.stdout}"
        assert out.exists()
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        for r in rows:
            assert r["recommendation"] == "ok", f"unexpected flag on {r}"

    def test_csv_input_low_overlap_flags_all(
        self, pdf_dir: Path, disagreeing_csv: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "audit.csv"
        result = runner.invoke(
            app,
            [
                "audit",
                "--against",
                str(disagreeing_csv),
                "--on",
                str(pdf_dir),
                "--output",
                str(out),
                "--quality",
                "fast",
            ],
        )
        # Should exit 3 — every document flagged
        assert result.exit_code == 3, f"got {result.exit_code}; stdout={result.stdout}"
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        for r in rows:
            assert r["recommendation"] == "review"
            assert float(r["jaccard_overlap"]) < 0.7

    def test_csv_columns_are_documented(
        self, pdf_dir: Path, matching_csv: Path, tmp_path: Path
    ) -> None:
        """The CSV column list is part of the public contract — pin it."""
        out = tmp_path / "audit.csv"
        runner.invoke(
            app,
            [
                "audit",
                "--against",
                str(matching_csv),
                "--on",
                str(pdf_dir),
                "--output",
                str(out),
                "--quality",
                "fast",
            ],
        )
        with out.open() as f:
            header = next(csv.reader(f))
        assert header == [
            "filename",
            "my_extractor_chars",
            "pdfmux_chars",
            "jaccard_overlap",
            "pdfmux_confidence",
            "recommendation",
            "error",
        ]


class TestAuditJSON:
    def test_json_input_works(self, pdf_dir: Path, disagreeing_json: Path, tmp_path: Path) -> None:
        out = tmp_path / "audit.csv"
        result = runner.invoke(
            app,
            [
                "audit",
                "--against",
                str(disagreeing_json),
                "--on",
                str(pdf_dir),
                "--output",
                str(out),
                "--quality",
                "fast",
            ],
        )
        assert result.exit_code == 3  # all flagged
        assert out.exists()


class TestAuditEdgeCases:
    def test_unsupported_extension_errors_cleanly(self, pdf_dir: Path, tmp_path: Path) -> None:
        bad = tmp_path / "other.txt"
        bad.write_text("filename,text\nalpha.pdf,foo\n")
        result = runner.invoke(
            app,
            ["audit", "--against", str(bad), "--on", str(pdf_dir), "--quality", "fast"],
        )
        assert result.exit_code == 2
        assert "must be .csv or .json" in result.stdout

    def test_no_filename_overlap_errors_cleanly(self, pdf_dir: Path, tmp_path: Path) -> None:
        """If --against names no PDF that's in --on, exit cleanly with a hint."""
        bad = tmp_path / "other.csv"
        with bad.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "text"])
            w.writerow(["nonexistent.pdf", "..."])
        result = runner.invoke(
            app,
            ["audit", "--against", str(bad), "--on", str(pdf_dir), "--quality", "fast"],
        )
        assert result.exit_code == 2
        assert "No filename overlap" in result.stdout

    def test_csv_with_alternate_column_names(self, pdf_dir: Path, tmp_path: Path) -> None:
        """Accept 'file' and 'content' as aliases for 'filename' and 'text'."""
        alt = tmp_path / "other.csv"
        with alt.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "content"])
            w.writerow(["alpha.pdf", "Alpha document about chemistry safety"])
            w.writerow(["beta.pdf", "Beta document about industrial regulation"])
        out = tmp_path / "audit.csv"
        result = runner.invoke(
            app,
            [
                "audit",
                "--against",
                str(alt),
                "--on",
                str(pdf_dir),
                "--output",
                str(out),
                "--quality",
                "fast",
            ],
        )
        # Either 0 or 3 is fine — what matters is it parsed.
        assert result.exit_code in (0, 3), f"got {result.exit_code}; stdout={result.stdout}"
        assert out.exists()

    def test_directory_with_no_pdfs_errors(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        bad = tmp_path / "other.csv"
        bad.write_text("filename,text\n")
        result = runner.invoke(
            app,
            ["audit", "--against", str(bad), "--on", str(empty), "--quality", "fast"],
        )
        assert result.exit_code == 2
