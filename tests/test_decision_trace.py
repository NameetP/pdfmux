"""Tests for the persisted per-page decision trace (JSON schema 1.2.0).

The decision trace records, for each page: the non-OCR audit class and score,
the recognition-budget verdict, the final extractor, and each repair attempt
(accepted or rejected). It is emitted under ``decision_trace`` in JSON output.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pdfmux
from pdfmux.types import PageDecision, RepairAttempt


class TestDecisionTraceTypes:
    """PageDecision / RepairAttempt are frozen and serialize cleanly."""

    def test_page_decision_serializes(self) -> None:
        d = PageDecision(
            page_num=0,
            audit_class="good",
            audit_score=0.95,
            budget_eligible=True,
            budget_reason="good page — no escalation needed",
            final_extractor="pymupdf4llm",
            ocr_applied=False,
        )
        out = asdict(d)
        assert out["audit_class"] == "good"
        assert out["audit_score"] == 0.95
        assert out["ocr_applied"] is False
        assert out["attempts"] == ()  # asdict preserves the empty tuple; JSON makes it []

    def test_rejected_and_accepted_attempts_are_retained(self) -> None:
        """A decision can carry both accepted and rejected repair attempts."""
        accepted = RepairAttempt(
            stage="ocr", extractor="rapidocr",
            score_before=0.0, score_after=0.9, accepted=True, reason="recovered text",
        )
        rejected = RepairAttempt(
            stage="vision", extractor="gemini",
            score_before=0.9, score_after=0.7, accepted=False,
            reason="candidate reduced audit score",
        )
        d = PageDecision(
            page_num=2, audit_class="empty", audit_score=0.0,
            budget_eligible=True, budget_reason="within OCR budget (30%)",
            final_extractor="rapidocr", ocr_applied=True,
            attempts=(accepted, rejected),
        )
        out = asdict(d)
        assert len(out["attempts"]) == 2
        assert out["attempts"][0]["accepted"] is True
        assert out["attempts"][1]["accepted"] is False
        assert out["attempts"][1]["reason"] == "candidate reduced audit score"


class TestDecisionTraceInJSON:
    """extract_json emits a per-page decision_trace under schema 1.2.0."""

    def test_schema_version_bumped(self, digital_pdf: Path) -> None:
        data = pdfmux.extract_json(digital_pdf)
        assert data["schema_version"] == "1.2.0"

    def test_json_includes_decision_trace(self, digital_pdf: Path) -> None:
        data = pdfmux.extract_json(digital_pdf)
        assert "decision_trace" in data, "standard-mode extraction should emit a trace"
        trace = data["decision_trace"]
        assert isinstance(trace, list) and len(trace) >= 1
        first = trace[0]
        for key in (
            "page_num", "audit_class", "audit_score", "budget_eligible",
            "budget_reason", "final_extractor", "ocr_applied", "attempts",
        ):
            assert key in first, f"decision record missing {key!r}"
        assert first["audit_class"] in ("good", "bad", "empty")
        assert 0.0 <= first["audit_score"] <= 1.0

    def test_one_decision_per_page(self, multi_page_pdf: Path) -> None:
        data = pdfmux.extract_json(multi_page_pdf)
        assert "decision_trace" in data
        assert len(data["decision_trace"]) == data["page_count"]

    def test_clean_digital_pages_not_escalated(self, digital_pdf: Path) -> None:
        """A clean digital PDF audits 'good': eligible, no OCR, no attempts."""
        data = pdfmux.extract_json(digital_pdf)
        for d in data["decision_trace"]:
            assert d["ocr_applied"] is False
            assert d["budget_eligible"] is True
            assert d["audit_class"] == "good"
            assert d["attempts"] == []
