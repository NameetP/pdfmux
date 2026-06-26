"""Tests for the monotonic repair guard (§5.8) — the single accept/reject gate.

The guard decides whether a re-extraction candidate (region OCR, full-page OCR,
vision LLM, or agentic re-extraction) replaces or augments a page. Three
conjoined conditions: additive-patch-only for trusted native spans, hard-fail
signals, and a calibrated audit-delta gate. Every attempt — accepted AND
rejected — is recorded in the page's decision trace; retaining rejected patches
is the cleanest differentiator of the trace.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdfmux.audit import (
    NATIVE_TRUST_THRESHOLD,
    _repair_hard_fail,
    accept_repair,
    repair_score_delta,
    score_page,
)
from pdfmux.detect import classify
from pdfmux.pipeline import _multipass_extract
from pdfmux.types import WeakRegion

# A clean page of prose — scores ~1.0 (trusted native).
GOOD = (
    "Introduction. This document describes the quarterly results in clear prose. "
    "Revenue grew steadily across all regions and the team shipped on schedule. "
    "The following sections detail the methodology and the supporting evidence."
)
# Low-quality text: mojibake + low alphabetic ratio — scores low, not trusted.
GARBLED = "â€™â€ Ã© 12 34 $$ %% â€™â€ Ã¨ 99 00 ## â€™ Ã© â€ 11 22 33 $$ %%%"


class TestAcceptRepairFullReplacement:
    """Full (non-additive) replacement must strictly beat the original."""

    def test_garbled_to_good_accepts(self) -> None:
        accepted, sb, sa, reason = accept_repair(GARBLED, GOOD)
        assert accepted is True
        assert sa > sb
        assert "improved" in reason

    def test_trusted_native_blocks_full_replacement(self) -> None:
        """A high-scoring native page cannot be wholesale-replaced."""
        accepted, sb, _sa, reason = accept_repair(GOOD, GARBLED)
        assert accepted is False
        assert sb >= NATIVE_TRUST_THRESHOLD
        assert "full replacement barred" in reason

    def test_no_improvement_rejected(self) -> None:
        """Same-quality candidate on a non-trusted original is rejected."""
        # A short text scores below the trust bar, so we reach the delta gate.
        a = "Acct number twelve, balance due soon."
        accepted, sb, sa, _reason = accept_repair(a, a)
        assert sb < NATIVE_TRUST_THRESHOLD  # not trusted, so we reach the delta gate
        assert accepted is False
        assert sa == sb

    def test_empty_original_accepts_recovery(self) -> None:
        """Any usable text strictly beats an empty page."""
        accepted, sb, sa, _reason = accept_repair("", GOOD)
        assert accepted is True
        assert sb == 0.0
        assert sa > 0.0

    def test_empty_candidate_rejected(self) -> None:
        accepted, _sb, _sa, reason = accept_repair(GARBLED, "   ")
        assert accepted is False
        assert "no text" in reason

    def test_margin_demands_bigger_improvement(self) -> None:
        """A real improvement is rejected when the required margin exceeds it."""
        # short original (not trusted) -> longer candidate is a genuine improvement
        original = "Acct number twelve, balance due soon."
        better = (
            original + " Additional clean readable prose to extend the page well "
            "past the fifty character mark here."
        )
        sb, sa = repair_score_delta(original, better)
        assert sb < NATIVE_TRUST_THRESHOLD and sa > sb  # genuine, sub-trust improvement
        # With a margin larger than the delta, the same candidate is rejected.
        big_margin = (sa - sb) + 0.5
        accepted, _, _, reason = accept_repair(original, better, margin=big_margin)
        assert accepted is False
        assert "margin" in reason
        # And accepted at the default (zero) margin.
        assert accept_repair(original, better)[0] is True


class TestAcceptRepairAdditive:
    """Additive (region OCR) patches need only be non-decreasing in audit score."""

    def test_additive_clean_append_accepts_even_when_trusted(self) -> None:
        """Region OCR may augment a trusted page (it cannot overwrite it)."""
        candidate = GOOD + "\n\nRecovered caption text from an embedded chart image."
        accepted, _sb, _sa, reason = accept_repair(GOOD, candidate, additive=True)
        assert accepted is True
        assert "additive patch" in reason

    def test_additive_garbage_hard_fails(self) -> None:
        """Region OCR that injects mojibake is rejected (hard-fail)."""
        candidate = GOOD + "\n\nâ€™â€™ Ã© Ã¨ â€ â€™ garbage garbage â€™â€™â€™ Ã©Ã©Ã©"
        accepted, _sb, _sa, reason = accept_repair(GOOD, candidate, additive=True)
        assert accepted is False
        assert "mojibake" in reason


class TestRepairHardFail:
    """Hard-fail signals reject a degrading candidate regardless of score delta."""

    def test_introduces_mojibake(self) -> None:
        assert _repair_hard_fail("clean text here", "clean â€™â€™â€™ Ã© text") is not None

    def test_collapses_alpha_ratio(self) -> None:
        before = "This is a normal sentence with mostly letters in it."
        after = "12 34 56 78 90 !! @@ ## $$ %% ^^ && ** (( )) -- ++ == 11"
        reason = _repair_hard_fail(before, after)
        assert reason is not None and "alpha" in reason

    def test_suspicious_shortening(self) -> None:
        before = "x" * 400 + " of perfectly readable content that fills the page nicely"
        after = "tiny"
        reason = _repair_hard_fail(before, after)
        assert reason is not None and "%" in reason

    def test_loses_headings(self) -> None:
        before = "# Title\n\nSome body content under the heading goes right here."
        after = "Some body content under the heading goes right here, no heading."
        reason = _repair_hard_fail(before, after)
        assert reason is not None and "heading" in reason

    def test_clean_improvement_no_hard_fail(self) -> None:
        assert _repair_hard_fail(GARBLED, GOOD) is None


# ---------------------------------------------------------------------------
# Integration: the cascade records every attempt and retains rejected patches.
# Driven through _multipass_extract directly with a monkeypatched region-OCR so
# the test is deterministic whether or not an OCR engine is installed.
# ---------------------------------------------------------------------------


def _bad_page_pdf(tmp_path: Path) -> Path:
    """A PDF whose single page audits 'bad' — some text + an image without
    overlapping text — so it routes to region OCR."""
    pdf_path = tmp_path / "bad_page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Short caption near the top of the page.", fontsize=11)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 220, 220), 1)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(72, 400, 292, 620), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _disable_full_ocr_and_llm(monkeypatch) -> None:
    """Force full-page OCR and the LLM unavailable so the region decision is final."""
    monkeypatch.setattr(
        "pdfmux.extractors.rapid_ocr.RapidOCRExtractor.available", lambda self: False
    )
    monkeypatch.setattr("pdfmux.extractors.llm.LLMExtractor.available", lambda self: False)


def _find_attempt_page(pages):
    return next((p for p in pages if p.decision and p.decision.attempts), None)


class TestMonotonicGuardIntegration:
    def test_rejected_region_attempt_is_retained(self, tmp_path, monkeypatch) -> None:
        """A rejected region-OCR candidate is recorded; the page stays native."""
        pdf = _bad_page_pdf(tmp_path)
        _disable_full_ocr_and_llm(monkeypatch)

        def fake_region_ocr(fp, page_num, text):  # noqa: ARG001
            # Append mojibake garbage → additive hard-fail → rejected.
            garbage = text + "\n\nâ€™â€™â€™ Ã© Ã¨ â€ â€™â€™ Ã©Ã©Ã© garbage"
            return garbage, 1, [WeakRegion(page_num=page_num, bbox=(72, 400, 292, 620), reason="t")]

        monkeypatch.setattr("pdfmux.regions.region_ocr_page", fake_region_ocr)

        pages, _name, ocr_pages = _multipass_extract(pdf, classify(pdf))
        page = _find_attempt_page(pages)
        assert page is not None, "expected a page with a recorded repair attempt"

        rejected = [a for a in page.decision.attempts if a.stage == "region_ocr" and not a.accepted]
        assert rejected, "the rejected region-OCR attempt must be retained on the trace"
        assert "mojibake" in rejected[0].reason
        # The page kept its native text precisely because the candidate was rejected.
        assert page.decision.provenance_tier == "native"
        assert page.ocr_applied is False
        assert page.page_num not in ocr_pages

    def test_accepted_region_attempt_carries_provenance(self, tmp_path, monkeypatch) -> None:
        """A clean region recovery is accepted: region tier + bbox + accepted attempt."""
        pdf = _bad_page_pdf(tmp_path)
        _disable_full_ocr_and_llm(monkeypatch)
        bbox = (72.0, 400.0, 292.0, 620.0)

        def fake_region_ocr(fp, page_num, text):  # noqa: ARG001
            recovered = (
                text + "\n\nFigure 1. Annual revenue by region, recovered from the "
                "embedded chart image, with clean readable labels and a caption."
            )
            return recovered, 1, [WeakRegion(page_num=page_num, bbox=bbox, reason="t")]

        monkeypatch.setattr("pdfmux.regions.region_ocr_page", fake_region_ocr)

        pages, _name, ocr_pages = _multipass_extract(pdf, classify(pdf))
        page = _find_attempt_page(pages)
        assert page is not None

        accepted = [a for a in page.decision.attempts if a.stage == "region_ocr" and a.accepted]
        assert accepted, "the clean region recovery should be accepted and recorded"
        assert page.decision.provenance_tier == "region"
        assert bbox in page.decision.region_bboxes
        assert page.page_num in ocr_pages
