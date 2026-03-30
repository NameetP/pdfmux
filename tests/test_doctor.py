"""Tests for the enhanced doctor command recommendations."""

from __future__ import annotations

from pdfmux.router.engine import QUALITY_ESTIMATES, RouterEngine


class TestRouterEngineAvailability:
    def test_pymupdf_always_available(self):
        engine = RouterEngine()
        available = engine._get_available_extractors()
        assert "pymupdf" in available

    def test_available_is_set(self):
        engine = RouterEngine()
        available = engine._get_available_extractors()
        assert isinstance(available, set)


class TestQualityEstimates:
    def test_all_page_types_have_pymupdf(self):
        """PyMuPDF should have estimates for common page types."""
        for pt in ("digital", "scanned", "tables", "mixed", "graphical"):
            assert ("pymupdf", pt) in QUALITY_ESTIMATES

    def test_digital_pymupdf_is_high(self):
        assert QUALITY_ESTIMATES[("pymupdf", "digital")] > 0.90

    def test_scanned_pymupdf_is_low(self):
        assert QUALITY_ESTIMATES[("pymupdf", "scanned")] < 0.30

    def test_llm_covers_all_types(self):
        """LLM should have estimates for all page types."""
        for pt in ("digital", "scanned", "tables", "mixed", "graphical", "handwritten", "forms"):
            assert ("llm", pt) in QUALITY_ESTIMATES

    def test_quality_values_in_range(self):
        for key, score in QUALITY_ESTIMATES.items():
            assert 0.0 <= score <= 1.0, f"Quality for {key} is {score}"


class TestCoverageComputation:
    def test_best_available_for_digital(self):
        """Digital text should have high coverage even with just pymupdf."""
        engine = RouterEngine()
        available = engine._get_available_extractors()

        best = 0.0
        for (ext, ptype), quality in QUALITY_ESTIMATES.items():
            if ptype == "digital" and ext in available:
                best = max(best, quality)

        assert best > 0.80  # pymupdf handles digital well

    def test_recommendations_for_scanned(self):
        """Should recommend an extractor for scanned docs when only pymupdf available."""
        best_avail = 0.0
        best_overall = 0.0
        best_ext = None

        for (ext, ptype), quality in QUALITY_ESTIMATES.items():
            if ptype == "scanned":
                if ext == "pymupdf":
                    best_avail = max(best_avail, quality)
                if quality > best_overall:
                    best_overall = quality
                    best_ext = ext

        # There should be a significant gap
        assert best_overall - best_avail > 0.30
        assert best_ext != "pymupdf"


class TestReadinessScore:
    def test_readiness_with_pymupdf_only(self):
        """Readiness with just pymupdf should be moderate."""
        page_types = ["digital", "scanned", "tables", "mixed", "graphical", "handwritten", "forms"]
        available = {"pymupdf"}

        scores = []
        for pt in page_types:
            best = 0.0
            for (ext, ptype), quality in QUALITY_ESTIMATES.items():
                if ptype == pt and ext in available:
                    best = max(best, quality)
            scores.append(best)

        readiness = sum(scores) / len(scores) * 100
        # With just pymupdf: good at digital (0.95), bad at everything else
        assert 20 < readiness < 60

    def test_readiness_improves_with_more_extractors(self):
        """Adding extractors should improve readiness."""
        page_types = ["digital", "scanned", "tables", "mixed", "graphical"]

        def compute_readiness(available: set) -> float:
            scores = []
            for pt in page_types:
                best = 0.0
                for (ext, ptype), quality in QUALITY_ESTIMATES.items():
                    if ptype == pt and ext in available:
                        best = max(best, quality)
                scores.append(best)
            return sum(scores) / len(scores)

        r_pymupdf = compute_readiness({"pymupdf"})
        r_with_ocr = compute_readiness({"pymupdf", "rapidocr"})
        r_with_all = compute_readiness({"pymupdf", "rapidocr", "docling", "llm", "opendataloader"})

        assert r_with_ocr > r_pymupdf
        assert r_with_all > r_with_ocr
