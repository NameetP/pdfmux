"""Routing strategies — economy, balanced, premium.

Each strategy defines weight profiles for scoring candidate extractors
across quality, cost, and speed dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Strategy(Enum):
    """Extraction strategy — controls quality/cost/speed tradeoff."""

    ECONOMY = "economy"  # cheapest above quality floor
    BALANCED = "balanced"  # best quality/cost ratio (default)
    PREMIUM = "premium"  # best quality regardless of cost


@dataclass(frozen=True)
class StrategyWeights:
    """Scoring weights for a strategy."""

    quality: float
    cost: float
    speed: float
    quality_floor: float  # minimum acceptable quality score


# Weight profiles per strategy
STRATEGY_WEIGHTS: dict[Strategy, StrategyWeights] = {
    Strategy.ECONOMY: StrategyWeights(quality=0.2, cost=0.6, speed=0.2, quality_floor=0.60),
    Strategy.BALANCED: StrategyWeights(quality=0.4, cost=0.3, speed=0.3, quality_floor=0.70),
    Strategy.PREMIUM: StrategyWeights(quality=0.7, cost=0.1, speed=0.2, quality_floor=0.0),
}


def get_weights(strategy: Strategy) -> StrategyWeights:
    """Get scoring weights for a strategy."""
    return STRATEGY_WEIGHTS[strategy]
