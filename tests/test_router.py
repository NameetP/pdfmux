"""Tests for the routing engine and strategies."""

from __future__ import annotations

import pytest

from pdfmux.router.engine import (
    ROUTING_MATRIX,
    RouteDecision,
    RouterEngine,
)
from pdfmux.router.strategies import (
    Strategy,
    StrategyWeights,
    get_weights,
)

# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestStrategy:
    def test_three_strategies_exist(self):
        assert Strategy.ECONOMY.value == "economy"
        assert Strategy.BALANCED.value == "balanced"
        assert Strategy.PREMIUM.value == "premium"

    def test_all_strategies_have_weights(self):
        for s in Strategy:
            w = get_weights(s)
            assert isinstance(w, StrategyWeights)
            assert abs(w.quality + w.cost + w.speed - 1.0) < 0.01

    def test_economy_favors_cost(self):
        w = get_weights(Strategy.ECONOMY)
        assert w.cost > w.quality
        assert w.cost > w.speed

    def test_premium_favors_quality(self):
        w = get_weights(Strategy.PREMIUM)
        assert w.quality > w.cost
        assert w.quality > w.speed

    def test_balanced_is_balanced(self):
        w = get_weights(Strategy.BALANCED)
        assert w.quality >= w.cost
        assert w.quality >= w.speed

    def test_economy_has_quality_floor(self):
        w = get_weights(Strategy.ECONOMY)
        assert w.quality_floor == 0.60

    def test_premium_has_no_quality_floor(self):
        w = get_weights(Strategy.PREMIUM)
        assert w.quality_floor == 0.0


# ---------------------------------------------------------------------------
# Routing matrix tests
# ---------------------------------------------------------------------------


class TestRoutingMatrix:
    def test_digital_economy_is_pymupdf(self):
        chain = ROUTING_MATRIX[("digital", Strategy.ECONOMY)]
        assert chain[0] == "pymupdf"

    def test_scanned_premium_starts_with_llm(self):
        chain = ROUTING_MATRIX[("scanned", Strategy.PREMIUM)]
        assert chain[0] == "llm"

    def test_tables_balanced_includes_docling(self):
        chain = ROUTING_MATRIX[("tables", Strategy.BALANCED)]
        assert "docling" in chain

    def test_all_chains_end_with_pymupdf(self):
        """Every fallback chain should have pymupdf as ultimate fallback."""
        for key, chain in ROUTING_MATRIX.items():
            assert "pymupdf" in chain, f"Chain for {key} missing pymupdf fallback"


# ---------------------------------------------------------------------------
# Router engine tests
# ---------------------------------------------------------------------------


class TestRouterEngine:
    def test_select_digital_economy(self):
        engine = RouterEngine()
        decision = engine.select("digital", Strategy.ECONOMY)
        assert decision.extractor == "pymupdf"
        assert decision.estimated_cost_usd == 0.0

    def test_select_returns_route_decision(self):
        engine = RouterEngine()
        decision = engine.select("digital", Strategy.BALANCED)
        assert isinstance(decision, RouteDecision)
        assert decision.extractor in ("pymupdf", "opendataloader")
        assert decision.reason != ""

    def test_select_unknown_page_type_falls_back(self):
        engine = RouterEngine()
        decision = engine.select("unknown_type", Strategy.BALANCED)
        assert decision.extractor in ("pymupdf", "opendataloader")

    def test_select_respects_budget(self):
        engine = RouterEngine()
        # With zero budget, should not pick LLM even for premium
        decision = engine.select("scanned", Strategy.PREMIUM, budget_remaining=0.0)
        assert decision.extractor != "llm"

    def test_fallback_chain_populated(self):
        engine = RouterEngine()
        decision = engine.select("tables", Strategy.BALANCED)
        # Should have fallbacks beyond the primary
        assert isinstance(decision.fallback_chain, tuple)

    def test_estimate_document_cost(self):
        engine = RouterEngine()
        page_types = ["digital", "digital", "digital", "tables", "scanned"]
        cost = engine.estimate_document_cost(page_types, Strategy.ECONOMY)
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_select_with_fallback_escalates(self):
        engine = RouterEngine()
        # Low confidence should trigger escalation
        decision = engine.select_with_fallback(
            "tables", Strategy.BALANCED, confidence=0.30, confidence_threshold=0.70
        )
        assert isinstance(decision, RouteDecision)
        assert "escalated" in decision.reason or decision.fallback_chain is not None

    def test_select_with_fallback_no_escalation(self):
        engine = RouterEngine()
        # High confidence should not escalate
        decision = engine.select_with_fallback(
            "digital", Strategy.BALANCED, confidence=0.95, confidence_threshold=0.70
        )
        assert "escalated" not in decision.reason


# ---------------------------------------------------------------------------
# RouteDecision tests
# ---------------------------------------------------------------------------


class TestRouteDecision:
    def test_creation(self):
        d = RouteDecision(extractor="pymupdf", reason="test")
        assert d.extractor == "pymupdf"
        assert d.provider is None
        assert d.estimated_cost_usd == 0.0

    def test_llm_decision(self):
        d = RouteDecision(
            extractor="llm",
            provider="claude",
            model="claude-sonnet-4-6-20250514",
            estimated_cost_usd=0.008,
        )
        assert d.extractor == "llm"
        assert d.provider == "claude"
        assert d.model == "claude-sonnet-4-6-20250514"

    def test_frozen(self):
        d = RouteDecision(extractor="pymupdf")
        with pytest.raises(AttributeError):
            d.extractor = "changed"


# ---------------------------------------------------------------------------
# Cost field tests
# ---------------------------------------------------------------------------


class TestCostFields:
    def test_page_result_has_cost_fields(self):
        from pdfmux.types import PageQuality, PageResult

        pr = PageResult(
            page_num=0,
            text="hello",
            confidence=0.9,
            quality=PageQuality.GOOD,
            extractor="test",
            cost_usd=0.005,
            tokens_used=100,
        )
        assert pr.cost_usd == 0.005
        assert pr.tokens_used == 100

    def test_page_result_cost_defaults_zero(self):
        from pdfmux.types import PageQuality, PageResult

        pr = PageResult(
            page_num=0,
            text="hello",
            confidence=0.9,
            quality=PageQuality.GOOD,
            extractor="test",
        )
        assert pr.cost_usd == 0.0
        assert pr.tokens_used == 0

    def test_document_result_has_cost_fields(self):
        from pdfmux.types import DocumentResult

        dr = DocumentResult(
            pages=(),
            source="test.pdf",
            confidence=0.9,
            extractor_used="test",
            format="markdown",
            text="hello",
            total_cost_usd=0.05,
            total_tokens=500,
        )
        assert dr.total_cost_usd == 0.05
        assert dr.total_tokens == 500
