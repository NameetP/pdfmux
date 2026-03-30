"""Router engine — picks the best extractor for each page type.

Uses a static routing matrix (routing_defaults) that maps
(page_type, strategy) → ordered fallback chain of extractors.
When benchmark data exists, it overrides the static defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pdfmux.router.strategies import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteDecision:
    """Result of a routing decision."""

    extractor: str  # "pymupdf", "opendataloader", "docling", "rapidocr", "llm"
    provider: str | None = None  # LLM provider name if extractor=="llm"
    model: str | None = None  # LLM model if extractor=="llm"
    fallback_chain: tuple[str, ...] = ()  # remaining fallbacks
    estimated_cost_usd: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Static routing matrix — the cold-start defaults
# ---------------------------------------------------------------------------

# Fallback chains per (page_type, strategy).
# First entry is preferred. If it's not available, try next.
# "llm" means use the best available LLM provider.

ROUTING_MATRIX: dict[tuple[str, Strategy], tuple[str, ...]] = {
    # Digital text — no LLM needed, fast extractors dominate
    ("digital", Strategy.ECONOMY): ("pymupdf",),
    ("digital", Strategy.BALANCED): ("opendataloader", "pymupdf"),
    ("digital", Strategy.PREMIUM): ("opendataloader", "pymupdf"),
    # Scanned documents — OCR or LLM
    ("scanned", Strategy.ECONOMY): ("rapidocr", "pymupdf"),
    ("scanned", Strategy.BALANCED): ("rapidocr", "llm", "pymupdf"),
    ("scanned", Strategy.PREMIUM): ("llm", "rapidocr", "pymupdf"),
    # Tables — Docling excels, LLM for complex cases
    ("tables", Strategy.ECONOMY): ("docling", "opendataloader", "pymupdf"),
    ("tables", Strategy.BALANCED): ("docling", "opendataloader", "llm", "pymupdf"),
    ("tables", Strategy.PREMIUM): ("llm", "docling", "opendataloader", "pymupdf"),
    # Mixed documents — multi-pass approach
    ("mixed", Strategy.ECONOMY): ("opendataloader", "pymupdf"),
    ("mixed", Strategy.BALANCED): ("opendataloader", "pymupdf"),
    ("mixed", Strategy.PREMIUM): ("llm", "opendataloader", "pymupdf"),
    # Graphical / image-heavy — needs OCR or LLM
    ("graphical", Strategy.ECONOMY): ("rapidocr", "pymupdf"),
    ("graphical", Strategy.BALANCED): ("rapidocr", "llm", "pymupdf"),
    ("graphical", Strategy.PREMIUM): ("llm", "rapidocr", "pymupdf"),
    # Handwriting — LLM is best, OCR as fallback
    ("handwritten", Strategy.ECONOMY): ("rapidocr", "pymupdf"),
    ("handwritten", Strategy.BALANCED): ("llm", "rapidocr", "pymupdf"),
    ("handwritten", Strategy.PREMIUM): ("llm", "rapidocr", "pymupdf"),
    # Forms — structured extraction, LLM excels
    ("forms", Strategy.ECONOMY): ("opendataloader", "docling", "pymupdf"),
    ("forms", Strategy.BALANCED): ("opendataloader", "docling", "llm", "pymupdf"),
    ("forms", Strategy.PREMIUM): ("llm", "docling", "opendataloader", "pymupdf"),
}

# Default fallback for unknown page types
DEFAULT_CHAIN: dict[Strategy, tuple[str, ...]] = {
    Strategy.ECONOMY: ("pymupdf",),
    Strategy.BALANCED: ("opendataloader", "pymupdf"),
    Strategy.PREMIUM: ("llm", "opendataloader", "pymupdf"),
}

# Extractor name → pip install hint
INSTALL_HINTS: dict[str, str] = {
    "opendataloader": "pip install pdfmux[opendataloader]",
    "docling": "pip install pdfmux[tables]",
    "rapidocr": "pip install pdfmux[ocr]",
    "llm": "pip install pdfmux[llm]",
}

# Approximate cost per page (USD) for cost scoring
COST_PER_PAGE: dict[str, float] = {
    "pymupdf": 0.0,
    "opendataloader": 0.0,
    "docling": 0.0,
    "rapidocr": 0.0,
    "surya": 0.0,
    "llm": 0.01,  # varies by provider, this is average
}

# Approximate quality score per (extractor, page_type) — 0.0 to 1.0
# These are overridden by benchmark data when available
QUALITY_ESTIMATES: dict[tuple[str, str], float] = {
    ("pymupdf", "digital"): 0.95,
    ("pymupdf", "scanned"): 0.10,
    ("pymupdf", "tables"): 0.60,
    ("pymupdf", "mixed"): 0.70,
    ("pymupdf", "graphical"): 0.15,
    ("opendataloader", "digital"): 0.97,
    ("opendataloader", "scanned"): 0.40,
    ("opendataloader", "tables"): 0.85,
    ("opendataloader", "mixed"): 0.85,
    ("opendataloader", "graphical"): 0.30,
    ("docling", "digital"): 0.90,
    ("docling", "tables"): 0.92,
    ("docling", "mixed"): 0.80,
    ("rapidocr", "scanned"): 0.75,
    ("rapidocr", "graphical"): 0.65,
    ("rapidocr", "mixed"): 0.60,
    ("llm", "digital"): 0.90,
    ("llm", "scanned"): 0.88,
    ("llm", "tables"): 0.85,
    ("llm", "mixed"): 0.85,
    ("llm", "graphical"): 0.82,
    ("llm", "handwritten"): 0.80,
    ("llm", "forms"): 0.85,
}

# Approximate speed (pages per second)
SPEED_ESTIMATES: dict[str, float] = {
    "pymupdf": 100.0,
    "opendataloader": 20.0,
    "docling": 1.0,
    "rapidocr": 2.0,
    "surya": 1.0,
    "llm": 0.5,
}


class RouterEngine:
    """Intelligent extraction routing engine.

    Selects the best extractor for a given page type and strategy,
    considering available backends and cost budget.
    """

    def __init__(self) -> None:
        self._available_extractors: set[str] | None = None
        self._benchmark_data: dict | None = None
        self._load_benchmarks()

    def _load_benchmarks(self) -> None:
        """Load benchmark results if available."""
        bench_path = Path.home() / ".config" / "pdfmux" / "eval_results.json"
        if bench_path.is_file():
            try:
                import json

                with open(bench_path) as f:
                    self._benchmark_data = json.load(f)
                logger.debug("Loaded benchmark data from %s", bench_path)
            except Exception as e:
                logger.debug("Failed to load benchmarks: %s", e)

    def _get_available_extractors(self) -> set[str]:
        """Discover which extractors are actually installed."""
        if self._available_extractors is not None:
            return self._available_extractors

        available = {"pymupdf"}  # always available (required dep)

        try:
            from pdfmux.extractors import available_extractors

            for name, _ in available_extractors():
                available.add(name)
        except Exception:
            pass

        # Also check LLM providers
        try:
            from pdfmux.providers import available_providers

            if available_providers():
                available.add("llm")
        except Exception:
            pass

        self._available_extractors = available
        return available

    def select(
        self,
        page_type: str,
        strategy: Strategy = Strategy.BALANCED,
        budget_remaining: float | None = None,
    ) -> RouteDecision:
        """Pick the best extractor for this page type and strategy.

        Args:
            page_type: "digital", "scanned", "tables", "mixed", "graphical", etc.
            strategy: ECONOMY, BALANCED, or PREMIUM
            budget_remaining: Optional cost cap in USD. If set and exceeded,
                             forces free extractors only.

        Returns:
            RouteDecision with the selected extractor and fallback chain.
        """
        available = self._get_available_extractors()

        # If budget exhausted, force economy
        if budget_remaining is not None and budget_remaining <= 0:
            strategy = Strategy.ECONOMY

        # Get fallback chain for this (page_type, strategy)
        chain = ROUTING_MATRIX.get(
            (page_type, strategy),
            DEFAULT_CHAIN.get(strategy, ("pymupdf",)),
        )

        # Filter to available extractors
        filtered = [e for e in chain if e in available]
        if not filtered:
            filtered = ["pymupdf"]  # ultimate fallback

        # Budget check: skip LLM if it would exceed budget
        if budget_remaining is not None:
            llm_cost = COST_PER_PAGE.get("llm", 0.01)
            if llm_cost > budget_remaining:
                filtered = [e for e in filtered if e != "llm"]
                if not filtered:
                    filtered = ["pymupdf"]

        selected = filtered[0]
        remaining = tuple(filtered[1:])

        # Resolve LLM provider details
        provider = None
        model = None
        if selected == "llm":
            try:
                from pdfmux.providers import resolve_provider

                p = resolve_provider()
                provider = p.name
                model = os.environ.get("PDFMUX_LLM_MODEL") or p.default_model
            except Exception:
                # LLM unavailable, skip to next
                if remaining:
                    selected = remaining[0]
                    remaining = remaining[1:]
                else:
                    selected = "pymupdf"
                    remaining = ()

        estimated_cost = COST_PER_PAGE.get(selected, 0.0)
        quality_est = QUALITY_ESTIMATES.get((selected, page_type), 0.5)

        reason = (
            f"strategy={strategy.value}, page_type={page_type}, "
            f"quality_est={quality_est:.2f}, cost=${estimated_cost:.4f}"
        )

        return RouteDecision(
            extractor=selected,
            provider=provider,
            model=model,
            fallback_chain=remaining,
            estimated_cost_usd=estimated_cost,
            reason=reason,
        )

    def estimate_document_cost(
        self,
        page_types: list[str],
        strategy: Strategy = Strategy.BALANCED,
    ) -> float:
        """Estimate total cost for a document given its page types."""
        total = 0.0
        for pt in page_types:
            decision = self.select(pt, strategy)
            total += decision.estimated_cost_usd
        return total

    def select_with_fallback(
        self,
        page_type: str,
        strategy: Strategy,
        confidence: float,
        confidence_threshold: float = 0.70,
        budget_remaining: float | None = None,
    ) -> RouteDecision:
        """Select extractor, considering previous extraction confidence.

        If confidence is below threshold, escalate to a better extractor
        (skip the first in the chain, go to the next).
        """
        decision = self.select(page_type, strategy, budget_remaining)

        if confidence >= confidence_threshold:
            return decision

        # Confidence too low — try next in fallback chain
        if decision.fallback_chain:
            next_extractor = decision.fallback_chain[0]
            remaining = decision.fallback_chain[1:]

            provider = None
            model = None
            if next_extractor == "llm":
                try:
                    from pdfmux.providers import resolve_provider

                    p = resolve_provider()
                    provider = p.name
                    model = os.environ.get("PDFMUX_LLM_MODEL") or p.default_model
                except Exception:
                    pass

            return RouteDecision(
                extractor=next_extractor,
                provider=provider,
                model=model,
                fallback_chain=remaining,
                estimated_cost_usd=COST_PER_PAGE.get(next_extractor, 0.0),
                reason=f"escalated: confidence={confidence:.2f} < {confidence_threshold:.2f}",
            )

        return decision  # no fallback available
