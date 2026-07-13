"""Tests for the Certify Anything verifier — audit any engine's extraction.

Covers the core scoring path (silent drop, clean pass, hallucination, coverage,
integrity), extraction parsing across JSON/markdown shapes, the unverifiable
(scanned-source) honesty guard, manifest signing/serialization, batch mode, and
a CLI smoke test.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from typer.testing import CliRunner

from pdfmux.cli import app
from pdfmux.verifier import (
    CertificationManifest,
    _alignment,
    _compute_signature,
    _hallucination_risk,
    _verify_page,
    parse_extraction,
    verify_batch,
    verify_extraction,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A body of real prose with enough tokens for alignment to be meaningful.
_P1 = (
    "# Quarterly Report\n\n"
    "The consolidated revenue for the fiscal quarter reached forty million "
    "dollars, an increase of twelve percent over the prior period. Operating "
    "margins expanded as the company reduced fulfilment costs across every "
    "regional warehouse. Management expects continued growth into the next "
    "half of the year, supported by new enterprise contracts."
)
_P2 = (
    "## Balance Sheet\n\n"
    "| Account | Amount |\n"
    "| --- | --- |\n"
    "| Cash and equivalents | 12,400,000 |\n"
    "| Accounts receivable | 8,900,000 |\n"
    "| Total assets | 61,300,000 |\n\n"
    "Liabilities remained stable while shareholder equity rose on retained "
    "earnings. The auditors issued an unqualified opinion for the period."
)
_P3 = (
    "## Outlook\n\n"
    "The board approved a capital expenditure programme covering automation "
    "of the primary distribution centre. Hiring will accelerate in the "
    "engineering and operations functions throughout the coming quarters. "
    "Currency headwinds remain the principal risk to the forecast."
)


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    """A 3-page digital PDF: prose, a table page, more prose."""
    pdf_path = tmp_path / "report.pdf"
    doc = fitz.open()
    for body in (_P1, _P2, _P3):
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def source_pages() -> dict[int, str]:
    """Pre-computed source page text so tests skip PDF re-extraction."""
    return {0: _P1, 1: _P2, 2: _P3}


# ---------------------------------------------------------------------------
# Extraction parsing
# ---------------------------------------------------------------------------


class TestParseExtraction:
    def test_pdfmux_style_json_pages_array(self) -> None:
        data = {"pages": [{"page": 1, "text": "hello"}, {"page": 2, "text": "world"}]}
        parsed = parse_extraction(data)
        assert parsed.page_aligned
        assert parsed.pages == {0: "hello", 1: "world"}

    def test_dict_pagenum_to_text(self) -> None:
        data = {"1": "first", "2": "second"}
        parsed = parse_extraction(data)
        assert parsed.pages == {0: "first", 1: "second"}

    def test_zero_indexed_json_preserved(self) -> None:
        data = [{"page": 0, "text": "a"}, {"page": 1, "text": "b"}]
        parsed = parse_extraction(data)
        assert parsed.pages == {0: "a", 1: "b"}

    def test_list_of_strings(self) -> None:
        parsed = parse_extraction(["page one", "page two", "page three"])
        assert parsed.pages == {0: "page one", 1: "page two", 2: "page three"}

    def test_reducto_style_chunks_with_content(self) -> None:
        data = {"result": {"chunks": [{"page": 1, "content": "x"}, {"page": 2, "content": "y"}]}}
        parsed = parse_extraction(data)
        assert parsed.pages == {0: "x", 1: "y"}

    def test_markdown_formfeed_split(self) -> None:
        parsed = parse_extraction("page one\f page two\f page three", fmt="markdown")
        assert parsed.page_aligned
        assert len(parsed.pages) == 3

    def test_flat_markdown_not_page_aligned(self) -> None:
        parsed = parse_extraction("just one big blob of text with no markers", fmt="markdown")
        assert not parsed.page_aligned
        assert set(parsed.pages) == {0}

    def test_json_file_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        p.write_text(json.dumps({"pages": [{"page": 1, "text": "hi"}]}), encoding="utf-8")
        parsed = parse_extraction(p)
        assert parsed.pages == {0: "hi"}
        assert parsed.raw_sha256.startswith("sha256:")

    def test_empty_json_yields_no_pages(self) -> None:
        parsed = parse_extraction({"unrelated": "structure"})
        assert parsed.pages == {}
        assert not parsed.page_aligned


# ---------------------------------------------------------------------------
# Token-level signals
# ---------------------------------------------------------------------------


class TestSignals:
    def test_alignment_perfect(self) -> None:
        assert _alignment("quick brown clever fox", "quick brown clever fox") == 1.0

    def test_alignment_partial(self) -> None:
        # 2 of 4 distinctive source tokens present (stopwords excluded).
        val = _alignment("quick brown clever fox", "quick brown")
        assert 0.4 <= val <= 0.6

    def test_hallucination_all_supported(self) -> None:
        assert _hallucination_risk("alpha beta gamma delta epsilon zeta", "alpha beta gamma") == 0.0

    def test_hallucination_invented_text(self) -> None:
        src = "alpha beta gamma delta"
        ext = "zulu yankee xray whiskey victor uniform tango sierra romeo quebec papa oscar"
        assert _hallucination_risk(src, ext) > 0.8

    def test_hallucination_ignores_tiny_fragments(self) -> None:
        # Below MIN_TOKENS_FOR_HALLUCINATION → 0.0 regardless.
        assert _hallucination_risk("alpha beta", "zzz") == 0.0


# ---------------------------------------------------------------------------
# Per-page verification
# ---------------------------------------------------------------------------


class TestVerifyPage:
    def test_clean_pass(self) -> None:
        pv = _verify_page(0, _P1, _P1)
        assert pv.verdict == "pass"
        assert not pv.silent_drop
        assert pv.alignment == 1.0

    def test_silent_drop(self) -> None:
        pv = _verify_page(0, _P1, "")
        assert pv.silent_drop
        assert pv.verdict == "fail"
        assert "silent_drop" in pv.flags

    def test_silent_drop_near_empty(self) -> None:
        pv = _verify_page(0, _P1, "x")  # under EMPTY_TEXT_THRESHOLD
        assert pv.silent_drop

    def test_blank_source_and_extraction_pass(self) -> None:
        pv = _verify_page(0, "", "")
        assert pv.verdict == "pass"
        assert "blank_page" in pv.flags

    def test_scanned_source_unverifiable(self) -> None:
        # Source has no text layer but engine produced text → we can't verify.
        pv = _verify_page(0, "", "Some OCR output that the engine claims to have read.")
        assert pv.verdict == "unverifiable"
        assert "source_no_text_layer" in pv.flags

    def test_table_dropped_flag(self) -> None:
        # Source page 2 has a markdown table; extraction flattens it to prose.
        flattened = "Balance Sheet Cash and equivalents 12,400,000 accounts receivable total assets"
        pv = _verify_page(1, _P2, flattened)
        assert not pv.table_integrity
        assert "table_dropped" in pv.flags
        assert pv.verdict == "review"

    def test_low_coverage_review(self) -> None:
        # Extraction keeps only a fraction of the source.
        pv = _verify_page(0, _P1, "The consolidated revenue reached forty million dollars.")
        assert pv.coverage < 0.6
        assert pv.verdict in {"review", "fail"}

    def test_hallucination_review(self) -> None:
        invented = (
            "The lunar colony reported record helium exports while the "
            "underwater railway connected three fictional continents overnight."
        )
        pv = _verify_page(0, _P1, invented)
        assert pv.hallucination_risk > 0.45
        assert "hallucination_risk" in pv.flags


# ---------------------------------------------------------------------------
# Full manifest
# ---------------------------------------------------------------------------


class TestVerifyExtraction:
    def test_faithful_extraction_passes(self, three_page_pdf: Path, source_pages) -> None:
        extraction = {"pages": [{"page": i + 1, "text": t} for i, t in source_pages.items()]}
        m = verify_extraction(three_page_pdf, extraction, engine="test", source_pages=source_pages)
        assert m.verdict == "PASS"
        assert m.silent_drops == ()
        assert m.confidence > 0.7

    def test_silent_drop_fails_overall(self, three_page_pdf: Path, source_pages) -> None:
        # Engine drops page 2 (returns empty) but claims success.
        extraction = {
            "pages": [
                {"page": 1, "text": _P1},
                {"page": 2, "text": ""},
                {"page": 3, "text": _P3},
            ]
        }
        m = verify_extraction(
            three_page_pdf, extraction, engine="dropper", source_pages=source_pages
        )
        assert m.verdict == "FAIL"
        assert 2 in m.silent_drops

    def test_missing_trailing_page_detected(self, three_page_pdf: Path, source_pages) -> None:
        # Engine only returns 2 of 3 pages → page 3 is a silent drop.
        extraction = {"pages": [{"page": 1, "text": _P1}, {"page": 2, "text": _P2}]}
        m = verify_extraction(
            three_page_pdf, extraction, engine="truncator", source_pages=source_pages
        )
        assert 3 in m.silent_drops
        assert m.verdict == "FAIL"

    def test_manifest_signature_is_deterministic_and_tamper_evident(
        self, three_page_pdf: Path, source_pages
    ) -> None:
        extraction = {"pages": [{"page": i + 1, "text": t} for i, t in source_pages.items()]}
        m = verify_extraction(three_page_pdf, extraction, engine="test", source_pages=source_pages)
        assert m.signature.startswith("sha256:")
        # Recompute over the body → same signature.
        assert _compute_signature(m._body()) == m.signature
        # Tamper with the body → signature no longer matches.
        body = m._body()
        body["verdict"] = "PASS_TAMPERED"
        assert _compute_signature(body) != m.signature

    def test_manifest_json_and_markdown_render(self, three_page_pdf: Path, source_pages) -> None:
        extraction = {"pages": [{"page": i + 1, "text": t} for i, t in source_pages.items()]}
        m = verify_extraction(three_page_pdf, extraction, engine="test", source_pages=source_pages)
        parsed = json.loads(m.to_json())
        assert parsed["schema_version"].startswith("certify-anything")
        assert "limitations" in parsed
        md = m.to_markdown()
        assert "Certification Manifest" in md
        assert "Provenance" in md

    def test_not_page_aligned_blob(self, three_page_pdf: Path, source_pages) -> None:
        # A flat markdown blob covering all pages, no page markers. All content
        # is present, just not segmented → must PASS, not fabricate drops.
        blob = _P1 + "\n\n" + _P2 + "\n\n" + _P3
        m = verify_extraction(
            three_page_pdf, blob, engine="blob", fmt="markdown", source_pages=source_pages
        )
        assert not m.page_aligned
        assert m.silent_drops == ()  # content is all there, no false drops
        assert m.verdict in {"PASS", "REVIEW"}
        assert all("unsegmented_extraction" in p.flags for p in m.pages)

    def test_unsegmented_blob_missing_page_is_caught(
        self, three_page_pdf: Path, source_pages
    ) -> None:
        # Blob that omits page 2 entirely → genuine silent drop, caught even
        # without pagination.
        blob = _P1 + "\n\n" + _P3
        m = verify_extraction(
            three_page_pdf, blob, engine="blob", fmt="markdown", source_pages=source_pages
        )
        assert not m.page_aligned
        assert 2 in m.silent_drops
        assert m.verdict == "FAIL"

    def test_single_blob_json_not_falsely_failed(self, three_page_pdf: Path, source_pages) -> None:
        # pdfmux's own JSON collapses all pages into one entry — must not be
        # read as "pages 2,3 silently dropped".
        blob_text = _P1 + "\n\n" + _P2 + "\n\n" + _P3
        extraction = {"page_count": 3, "pages": [{"page": 1, "text": blob_text}]}
        m = verify_extraction(
            three_page_pdf, extraction, engine="pdfmux", source_pages=source_pages
        )
        assert m.silent_drops == ()
        assert m.verdict in {"PASS", "REVIEW"}

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_extraction(tmp_path / "nope.pdf", {"pages": []})


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestVerifyBatch:
    def test_batch_aggregates(self, three_page_pdf: Path) -> None:
        good = {
            "pages": [{"page": 1, "text": _P1}, {"page": 2, "text": _P2}, {"page": 3, "text": _P3}]
        }
        bad = {
            "pages": [{"page": 1, "text": _P1}, {"page": 2, "text": ""}, {"page": 3, "text": _P3}]
        }
        good_path = three_page_pdf.parent / "good.json"
        bad_path = three_page_pdf.parent / "bad.json"
        good_path.write_text(json.dumps(good), encoding="utf-8")
        bad_path.write_text(json.dumps(bad), encoding="utf-8")

        batch = verify_batch(
            [(three_page_pdf, good_path), (three_page_pdf, bad_path)], engine="engineX"
        )
        assert batch.doc_count == 2
        assert batch.total_silent_drops == 1
        assert len(batch.docs_failed) == 1
        # Markdown report renders the killer headline.
        md = batch.to_markdown()
        assert "pages silently dropped" in md

    def test_batch_survives_bad_doc(self, three_page_pdf: Path, tmp_path: Path) -> None:
        good = {"pages": [{"page": 1, "text": _P1}]}
        good_path = tmp_path / "good.json"
        good_path.write_text(json.dumps(good), encoding="utf-8")
        batch = verify_batch(
            [(tmp_path / "missing.pdf", good_path), (three_page_pdf, good_path)],
            engine="engineX",
        )
        assert len(batch.errors) == 1
        assert len(batch.manifests) == 1


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestVerifyCLI:
    def test_cli_verify_extracted_json(self, three_page_pdf: Path, tmp_path: Path) -> None:
        extraction = {
            "pages": [{"page": 1, "text": _P1}, {"page": 2, "text": _P2}, {"page": 3, "text": _P3}]
        }
        ext_path = tmp_path / "ext.json"
        ext_path.write_text(json.dumps(extraction), encoding="utf-8")
        out_path = tmp_path / "manifest.json"
        result = runner.invoke(
            app,
            [
                "verify",
                "--source",
                str(three_page_pdf),
                "--extracted",
                str(ext_path),
                "--engine-name",
                "reducto",
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        manifest = json.loads(out_path.read_text())
        assert manifest["engine"] == "reducto"
        assert manifest["verdict"] in {"PASS", "REVIEW", "FAIL"}

    def test_cli_strict_gate_fails_on_silent_drop(
        self, three_page_pdf: Path, tmp_path: Path
    ) -> None:
        extraction = {
            "pages": [{"page": 1, "text": _P1}, {"page": 2, "text": ""}, {"page": 3, "text": _P3}]
        }
        ext_path = tmp_path / "ext.json"
        ext_path.write_text(json.dumps(extraction), encoding="utf-8")
        result = runner.invoke(
            app,
            ["verify", "--source", str(three_page_pdf), "--extracted", str(ext_path), "--strict"],
        )
        assert result.exit_code == 3, result.output

    def test_cli_rejects_both_engine_and_extracted(self, three_page_pdf: Path) -> None:
        result = runner.invoke(
            app,
            [
                "verify",
                "--source",
                str(three_page_pdf),
                "--extracted",
                "x.json",
                "--engine",
                "pdfmux",
            ],
        )
        assert result.exit_code == 2

    def test_cli_engine_stub_is_helpful(self, three_page_pdf: Path) -> None:
        result = runner.invoke(
            app, ["verify", "--source", str(three_page_pdf), "--engine", "reducto"]
        )
        assert result.exit_code == 2
        assert "reducto" in result.output.lower()
