"""Tests for provenance metadata on chunks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pdfmux
from pdfmux.chunking import chunk_by_sections
from pdfmux.formatters.json_fmt import format_json
from pdfmux.regions import region_ocr_page
from pdfmux.types import PageDecision, PageQuality, PageResult, WeakRegion


class TestChunkProvenance:
    """Chunks should carry extractor and ocr_applied metadata."""

    def test_chunk_has_extractor_field(self) -> None:
        """Chunk dataclass should have extractor field."""
        text = "# Heading\n\nSome content here that is long enough."
        chunks = chunk_by_sections(text, confidence=0.9, extractor="pymupdf4llm")
        assert len(chunks) > 0
        assert chunks[0].extractor == "pymupdf4llm"

    def test_chunk_has_ocr_applied_field(self) -> None:
        """Chunk dataclass should have ocr_applied field."""
        text = "# Heading\n\nSome OCR content here."
        chunks = chunk_by_sections(text, confidence=0.85, extractor="rapidocr", ocr_applied=True)
        assert len(chunks) > 0
        assert chunks[0].ocr_applied is True

    def test_chunk_defaults(self) -> None:
        """Default extractor and ocr_applied should be empty/False."""
        text = "# Heading\n\nContent."
        chunks = chunk_by_sections(text, confidence=0.9)
        assert len(chunks) > 0
        assert chunks[0].extractor == ""
        assert chunks[0].ocr_applied is False

    def test_llm_format_includes_provenance(self, digital_pdf: Path) -> None:
        """LLM format output should include extractor and ocr_applied per chunk."""
        chunks = pdfmux.load_llm_context(digital_pdf)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "extractor" in chunk
            assert "ocr_applied" in chunk
            assert isinstance(chunk["ocr_applied"], bool)


class TestTieredProvenance:
    """Build #3 — tiered provenance: native → exact, region → bbox, page/llm → page-level.

    The patent's tiered-provenance claim (family F) requires that the decision
    trace honestly records WHICH recognition tier produced each page's text and,
    for the region tier, the sub-page geometry. These tests exercise the data
    model + serialization deterministically (no OCR engine required).
    """

    def test_page_result_provenance_defaults(self) -> None:
        """A bare PageResult under-claims: page-level tier, no region geometry."""
        p = PageResult(
            page_num=0,
            text="hello",
            confidence=0.9,
            quality=PageQuality.GOOD,
            extractor="pymupdf4llm",
        )
        assert p.provenance_tier == "page"
        assert p.regions == ()

    def test_page_decision_provenance_defaults(self) -> None:
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
        assert out["provenance_tier"] == "page"
        assert out["region_bboxes"] == ()  # asdict keeps the tuple; JSON makes it []

    def test_region_tier_carries_bboxes_through_json(self) -> None:
        """A 'region'-tier decision serializes its sub-page geometry into JSON."""
        decision = PageDecision(
            page_num=0,
            audit_class="bad",
            audit_score=0.4,
            budget_eligible=True,
            budget_reason="bad page — within OCR budget (30%)",
            final_extractor="rapidocr",
            ocr_applied=True,
            provenance_tier="region",
            region_bboxes=((50.0, 100.0, 250.0, 300.0),),
        )
        out = json.loads(format_json("some text", decision_trace=[asdict(decision)]))
        assert out["schema_version"] == "1.4.0"
        entry = out["decision_trace"][0]
        assert entry["provenance_tier"] == "region"
        assert entry["region_bboxes"] == [[50.0, 100.0, 250.0, 300.0]]

    def test_region_ocr_page_returns_recovered_regions(self, digital_pdf: Path) -> None:
        """region_ocr_page returns a 3-tuple; the third element is a region list.

        On a page with no weak regions (clean digital text) it returns the page
        text unchanged, count 0, and an empty region list — without raising.
        """
        merged, n_regions, regions = region_ocr_page(digital_pdf, 0, "existing text")
        assert isinstance(merged, str)
        assert isinstance(n_regions, int)
        assert isinstance(regions, list)
        assert n_regions == len(regions)
        for r in regions:
            assert isinstance(r, WeakRegion)
            assert len(r.bbox) == 4

    def test_native_tier_has_no_region_geometry(self, multi_page_pdf: Path) -> None:
        """Every clean digital page is 'native' tier with no region bboxes."""
        data = pdfmux.extract_json(multi_page_pdf)
        tiers = {d["provenance_tier"] for d in data["decision_trace"]}
        assert tiers == {"native"}
        for d in data["decision_trace"]:
            assert d["region_bboxes"] == []
