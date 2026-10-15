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
    LOW_ALIGNMENT_THRESHOLD,
    LOW_COVERAGE_THRESHOLD,
    CertificationManifest,
    _alignment,
    _compute_signature,
    _hallucination_risk,
    _missing_numeric_literals,
    _normalize_numeric_literal,
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


class TestTableTruncationFlag:
    """The blind spot GT-0 measured: `_has_table` tests presence, not cardinality.

    A table cut from 40 rows to 3 still has >= 2 pipe rows, so `table_integrity`
    passes and the page reads clean. That is why 4 of 5 seeded truncations were
    missed. `table_truncated` is an ADDITIVE flag — no pre-registered threshold
    changes, so the published FP/FN figures stay valid for the signals they
    measured.
    """

    def test_truncated_table_is_flagged(self) -> None:
        from pdfmux.verifier import _table_truncated

        source = "\n".join(f"| r{i} | v{i} |" for i in range(40))
        truncated = "\n".join(f"| r{i} | v{i} |" for i in range(3))
        assert _table_truncated(source, truncated) is True

    def test_the_old_presence_check_still_passes_it(self) -> None:
        """Proves the gap is real and that the new flag is what closes it."""
        from pdfmux.verifier import _has_table, _table_truncated

        source = "\n".join(f"| r{i} | v{i} |" for i in range(40))
        truncated = "\n".join(f"| r{i} | v{i} |" for i in range(3))
        assert _has_table(truncated) is True, "presence check passes — the blind spot"
        assert _table_truncated(source, truncated) is True, "cardinality catches it"

    def test_intact_table_is_not_flagged(self) -> None:
        from pdfmux.verifier import _table_truncated

        source = "\n".join(f"| r{i} | v{i} |" for i in range(40))
        assert _table_truncated(source, source) is False

    def test_small_tables_cannot_trip_it(self) -> None:
        """A 3-row table losing one row must not fire — guards against new FPs."""
        from pdfmux.verifier import _table_truncated

        source = "| a | b |\n| - | - |\n| 1 | 2 |"
        assert _table_truncated(source, "| a | b |\n| - | - |") is False

    def test_no_table_in_source_never_flags(self) -> None:
        from pdfmux.verifier import _table_truncated

        assert _table_truncated("just prose, no table here", "just prose") is False

    def test_minor_row_loss_within_tolerance_is_not_flagged(self) -> None:
        """9 of 10 rows kept is above the 0.8 ratio — deliberately not a defect."""
        from pdfmux.verifier import _table_truncated

        source = "\n".join(f"| r{i} | v{i} |" for i in range(10))
        kept = "\n".join(f"| r{i} | v{i} |" for i in range(9))
        assert _table_truncated(source, kept) is False


# ---------------------------------------------------------------------------
# Numeric-literal review — opt-in `--check-numeric-literals` (NameetP/pdfmux#41)
#
# The reporter's repro (reportlab + pymupdf, in the issue body) showed
# `verify --strict` return PASS with no page flags after a 13-digit revenue
# figure was deleted from the extraction entirely — alignment 0.85, coverage
# 0.7864, both comfortably above the existing thresholds. Rebuilt here with
# pymupdf's own `page.insert_text` (this file's existing `three_page_pdf`
# fixture convention) instead of the reportlab generator, so no new test-only
# dependency is introduced; the page layout and the five financial figures are
# otherwise identical to the issue.
# ---------------------------------------------------------------------------

_FINANCIAL_ROWS = (
    ("Revenue", "1,348,243,150"),
    ("Costs", "872,000,100"),
    ("Profit", "476,243,050"),
    ("Research", "54,200,000"),
    ("Cash", "205,100,000"),
)

# The reporter's own larger variant: "a larger eight-row synthetic page where
# removing one amount or a whole row still returned PASS."
_FINANCIAL_ROWS_8 = _FINANCIAL_ROWS + (
    ("Tax", "128,400,000"),
    ("Dividends", "61,000,000"),
    ("Retained earnings", "286,843,050"),
)


def _build_financial_page(tmp_path: Path, rows: tuple[tuple[str, str], ...], name: str) -> Path:
    """One-page digital PDF of `label: value` rows, mirroring the issue's layout."""
    pdf_path = tmp_path / f"{name}.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i, (label, value) in enumerate(rows):
        y = 100 + 24 * i
        page.insert_text((60, y), label, fontsize=11)
        page.insert_text((300, y), value, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _page0_text(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as doc:
        return doc[0].get_text()


class TestNumericLiteralNormalization:
    """Format-normalizing comparison — the conservative handling the issue asks
    for, so `"1,348,243,150"` / `"1348243150"` / `"1,348,243,150.00"` never
    false-positive against each other."""

    @pytest.mark.parametrize(
        "token",
        ["1,348,243,150", "1348243150", "1,348,243,150.00", "1,348,243,150.0"],
    )
    def test_formatting_variants_normalize_identically(self, token: str) -> None:
        assert _normalize_numeric_literal(token) == "1348243150"

    def test_missing_literal_detected(self) -> None:
        findings = _missing_numeric_literals("Revenue 1,348,243,150", "Revenue missing here")
        assert len(findings) == 1
        assert findings[0].normalized == "1348243150"
        assert findings[0].literal == "1,348,243,150"
        assert findings[0].source_count == 1
        assert findings[0].extracted_count == 0

    @pytest.mark.parametrize(
        "extracted",
        ["Revenue 1348243150", "Revenue 1,348,243,150.00", "Revenue 1,348,243,150"],
    )
    def test_pure_formatting_difference_is_not_a_finding(self, extracted: str) -> None:
        findings = _missing_numeric_literals("Revenue 1,348,243,150", extracted)
        assert findings == ()

    def test_repeated_number_with_matching_counts_is_not_flagged(self) -> None:
        src = "Total 500,000 and again 500,000"
        ext = "Total 500,000 and again 500,000"
        assert _missing_numeric_literals(src, ext) == ()

    def test_small_bare_integers_are_not_tracked(self) -> None:
        # Bare (non-grouped, non-decimal) integers need >= 4 digits — keeps
        # page numbers / list indices / 2-digit years out of the signal.
        assert _missing_numeric_literals("page 12 of 45", "page of") == ()


class TestNumericLiteralReviewPerPage:
    """`_verify_page(..., check_numeric_literals=True)` — the per-page gate."""

    _SRC = (
        "Revenue reached 1,348,243,150 this quarter, a healthy increase over "
        "the prior comparable period for the whole division overall today."
    )
    _EXT_MISSING = (
        "Revenue reached this quarter, a healthy increase over the prior "
        "comparable period for the whole division overall today."
    )

    def test_flag_off_never_computes_findings(self) -> None:
        pv = _verify_page(0, self._SRC, self._EXT_MISSING)  # check_numeric_literals defaults False
        assert pv.numeric_literal_findings == ()
        assert "numeric_literal_review" not in pv.flags

    def test_intact_extraction_no_finding_flag_on(self) -> None:
        pv = _verify_page(0, self._SRC, self._SRC, check_numeric_literals=True)
        assert pv.numeric_literal_findings == ()
        assert pv.verdict == "pass"

    def test_missing_literal_is_review_not_fail(self) -> None:
        pv = _verify_page(0, self._SRC, self._EXT_MISSING, check_numeric_literals=True)
        assert pv.verdict == "review"
        assert pv.verdict != "fail"
        assert "numeric_literal_review" in pv.flags
        assert len(pv.numeric_literal_findings) == 1
        assert pv.numeric_literal_findings[0].normalized == "1348243150"

    def test_finding_never_escalates_coverage_fail(self) -> None:
        # A page easily clearing the existing coverage/alignment thresholds
        # still only reaches "review", never "fail", from this flag alone.
        pv = _verify_page(0, self._SRC, self._EXT_MISSING, check_numeric_literals=True)
        assert pv.coverage >= LOW_COVERAGE_THRESHOLD or pv.alignment >= LOW_ALIGNMENT_THRESHOLD
        assert pv.verdict == "review"


class TestNumericLiteralReviewRepro:
    """End-to-end against the reporter's own repro shape — a page-aligned JSON
    extraction of a one-page financial PDF, one figure deleted."""

    def test_intact_extraction_no_finding_either_way(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS, "intact_source")
        text = _page0_text(pdf_path)
        extraction = {"pages": [{"page": 1, "text": text}]}
        source_pages = {0: text}

        m_off = verify_extraction(
            pdf_path, extraction, check_numeric_literals=False, source_pages=source_pages
        )
        m_on = verify_extraction(
            pdf_path, extraction, check_numeric_literals=True, source_pages=source_pages
        )
        assert m_off.verdict == "PASS"
        assert m_on.verdict == "PASS"
        assert m_on.pages[0].numeric_literal_findings == ()

    def test_one_literal_deleted_review_only_when_flag_on(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS, "missing_source")
        text = _page0_text(pdf_path)
        assert text.count("1,348,243,150") == 1
        missing_text = text.replace("1,348,243,150\n", "", 1)
        extraction = {"pages": [{"page": 1, "text": missing_text}]}
        source_pages = {0: text}

        # Flag off: unchanged, backward-compatible behavior — PASS, exactly
        # what the reporter observed on 1.8.7.
        m_off = verify_extraction(
            pdf_path, extraction, check_numeric_literals=False, source_pages=source_pages
        )
        assert m_off.verdict == "PASS"
        assert m_off.pages[0].numeric_literal_findings == ()

        # Flag on: REVIEW (not FAIL), naming the exact dropped figure.
        m_on = verify_extraction(
            pdf_path, extraction, check_numeric_literals=True, source_pages=source_pages
        )
        assert m_on.verdict == "REVIEW"
        findings = m_on.pages[0].numeric_literal_findings
        assert len(findings) == 1
        assert findings[0].normalized == "1348243150"
        assert findings[0].literal == "1,348,243,150"
        assert findings[0].source_count == 1
        assert findings[0].extracted_count == 0

    def test_whole_row_deleted_flags_every_literal_not_just_the_first(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS_8, "eight_row_source")
        text = _page0_text(pdf_path)
        assert text.count("61,000,000") == 1
        row_deleted_text = text.replace("Dividends\n", "").replace("61,000,000\n", "")
        extraction = {"pages": [{"page": 1, "text": row_deleted_text}]}
        source_pages = {0: text}

        m = verify_extraction(
            pdf_path, extraction, check_numeric_literals=True, source_pages=source_pages
        )
        assert m.verdict == "REVIEW"
        findings = m.pages[0].numeric_literal_findings
        assert any(f.normalized == "61000000" for f in findings)
        # Every other figure on the page is still intact — exactly one finding.
        assert len(findings) == 1

    def test_deleting_two_rows_flags_both_literals(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS_8, "two_rows_source")
        text = _page0_text(pdf_path)
        two_rows_deleted = (
            text.replace("Dividends\n", "")
            .replace("61,000,000\n", "")
            .replace("Tax\n", "")
            .replace("128,400,000\n", "")
        )
        extraction = {"pages": [{"page": 1, "text": two_rows_deleted}]}
        source_pages = {0: text}

        m = verify_extraction(
            pdf_path, extraction, check_numeric_literals=True, source_pages=source_pages
        )
        findings = {f.normalized for f in m.pages[0].numeric_literal_findings}
        assert findings == {"61000000", "128400000"}

    def test_formatting_difference_not_flagged(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS, "reformatted_source")
        text = _page0_text(pdf_path)
        reformatted = text.replace("1,348,243,150", "1348243150")
        extraction = {"pages": [{"page": 1, "text": reformatted}]}
        source_pages = {0: text}

        m = verify_extraction(
            pdf_path, extraction, check_numeric_literals=True, source_pages=source_pages
        )
        assert m.verdict == "PASS"
        assert m.pages[0].numeric_literal_findings == ()

    def test_repeated_number_across_pages_only_flags_the_dropped_page(self, tmp_path: Path) -> None:
        """A running total shown on both pages; only page 2 drops it. Per-page
        counting must not punish page 1 for page 2's loss."""
        pdf_path = tmp_path / "two_page.pdf"
        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page()
            page.insert_text((60, 100), "Running total", fontsize=11)
            page.insert_text((300, 100), "999,000,000", fontsize=11)
            page.insert_text(
                (60, 140),
                "Narrative text to keep the page substantial enough to score reliably.",
                fontsize=10,
            )
        doc.save(str(pdf_path))
        doc.close()

        with fitz.open(pdf_path) as d:
            p1_text = d[0].get_text()
            p2_text = d[1].get_text()
        assert p2_text.count("999,000,000") == 1
        p2_dropped = p2_text.replace("999,000,000\n", "", 1)

        extraction = {"pages": [{"page": 1, "text": p1_text}, {"page": 2, "text": p2_dropped}]}
        source_pages = {0: p1_text, 1: p2_text}

        m = verify_extraction(
            pdf_path, extraction, check_numeric_literals=True, source_pages=source_pages
        )
        assert m.pages[0].numeric_literal_findings == ()
        assert m.pages[0].verdict == "pass"
        assert len(m.pages[1].numeric_literal_findings) == 1
        assert m.pages[1].numeric_literal_findings[0].normalized == "999000000"
        assert m.verdict == "REVIEW"


class TestNumericLiteralReviewCLI:
    def test_cli_strict_unaffected_by_default(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS, "cli_source_default")
        text = _page0_text(pdf_path)
        missing_text = text.replace("1,348,243,150\n", "", 1)
        ext_path = tmp_path / "missing.json"
        ext_path.write_text(
            json.dumps({"pages": [{"page": 1, "text": missing_text}]}), encoding="utf-8"
        )

        result = runner.invoke(
            app,
            [
                "verify",
                "--source",
                str(pdf_path),
                "--extracted",
                str(ext_path),
                "--strict",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0, result.output
        manifest = json.loads(result.stdout)
        assert manifest["verdict"] == "PASS"
        assert manifest["pages"][0]["numeric_literal_findings"] == []

    def test_cli_flag_surfaces_review_and_strict_gate(self, tmp_path: Path) -> None:
        pdf_path = _build_financial_page(tmp_path, _FINANCIAL_ROWS, "cli_source_flagged")
        text = _page0_text(pdf_path)
        missing_text = text.replace("1,348,243,150\n", "", 1)
        ext_path = tmp_path / "missing.json"
        ext_path.write_text(
            json.dumps({"pages": [{"page": 1, "text": missing_text}]}), encoding="utf-8"
        )

        result = runner.invoke(
            app,
            [
                "verify",
                "--source",
                str(pdf_path),
                "--extracted",
                str(ext_path),
                "--strict",
                "--check-numeric-literals",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 3, result.output
        manifest = json.loads(result.stdout)
        assert manifest["verdict"] == "REVIEW"
        findings = manifest["pages"][0]["numeric_literal_findings"]
        assert len(findings) == 1
        assert findings[0]["normalized"] == "1348243150"

    def test_cli_help_documents_the_flag(self) -> None:
        result = runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0
        assert "--check-numeric-literals" in result.output
