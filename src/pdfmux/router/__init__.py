"""Router — intelligent per-page extraction routing.

Routes each page to the best available extractor based on page type,
user strategy (economy/balanced/premium), and available providers.

Usage:
    from pdfmux.router import RouterEngine, Strategy

    engine = RouterEngine()
    decision = engine.select("tables", Strategy.BALANCED)
"""

from pdfmux.router.engine import RouteDecision, RouterEngine
from pdfmux.router.strategies import Strategy

__all__ = ["RouterEngine", "RouteDecision", "Strategy"]
