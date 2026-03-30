"""Data flywheel — local telemetry + routing re-ranking.

Records extraction outcomes locally (JSONL). After enough data,
re-ranks the routing matrix based on real-world performance.

All data stays local. Opt-in via PDFMUX_TELEMETRY=local.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

TELEMETRY_DIR = Path.home() / ".config" / "pdfmux"
TELEMETRY_FILE = TELEMETRY_DIR / "telemetry.jsonl"
RANKINGS_FILE = TELEMETRY_DIR / "learned_rankings.json"

# Minimum events before we start re-ranking
MIN_EVENTS_FOR_RANKING = 50

# Exponential decay half-life in days
DECAY_HALF_LIFE_DAYS = 14.0


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled (opt-in only)."""
    return os.environ.get("PDFMUX_TELEMETRY", "").lower() in ("local", "true", "1")


@dataclass
class ExtractionEvent:
    """One extraction outcome — recorded per page."""

    timestamp: str
    page_type: str  # digital, scanned, tables, mixed, graphical, etc.
    extractor: str  # pymupdf, docling, llm, etc.
    provider: str | None  # LLM provider name (if extractor=llm)
    model: str | None  # LLM model (if extractor=llm)
    confidence: float  # 0.0–1.0
    latency_ms: int
    cost_usd: float
    success: bool  # confidence >= threshold


class TelemetryCollector:
    """Records extraction events to local JSONL file."""

    def __init__(self, telemetry_path: Path | None = None):
        self.path = telemetry_path or TELEMETRY_FILE

    def record(self, event: ExtractionEvent) -> None:
        """Append an extraction event to the telemetry file."""
        if not is_telemetry_enabled():
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception as e:
            logger.debug("Failed to record telemetry: %s", e)

    def record_extraction(
        self,
        page_type: str,
        extractor: str,
        confidence: float,
        latency_ms: int,
        cost_usd: float = 0.0,
        provider: str | None = None,
        model: str | None = None,
        confidence_threshold: float = 0.70,
    ) -> None:
        """Convenience method to record an extraction."""
        event = ExtractionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            page_type=page_type,
            extractor=extractor,
            provider=provider,
            model=model,
            confidence=confidence,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=confidence >= confidence_threshold,
        )
        self.record(event)

    def load_events(self) -> list[ExtractionEvent]:
        """Load all recorded events."""
        events = []
        if not self.path.is_file():
            return events

        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        events.append(ExtractionEvent(**data))
        except Exception as e:
            logger.warning("Failed to load telemetry: %s", e)

        return events

    def event_count(self) -> int:
        """Count events without loading all data."""
        if not self.path.is_file():
            return 0
        try:
            with open(self.path) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all telemetry data."""
        if self.path.is_file():
            self.path.unlink()
        rankings_path = self.path.parent / "learned_rankings.json"
        if rankings_path.is_file():
            rankings_path.unlink()


class ReRanker:
    """Re-ranks routing based on accumulated telemetry.

    Uses exponential decay to weight recent events more heavily.
    Produces a score per (extractor, page_type) that the router
    can use to override static defaults.
    """

    def __init__(self, collector: TelemetryCollector | None = None):
        self.collector = collector or TelemetryCollector()

    def compute_rankings(self) -> dict[str, dict[str, float]]:
        """Compute efficiency scores per (page_type, extractor).

        Returns:
            {page_type: {extractor: score}} where score is 0.0–1.0.
        """
        events = self.collector.load_events()
        if len(events) < MIN_EVENTS_FOR_RANKING:
            return {}

        now = datetime.now(UTC)

        # Group by (page_type, extractor), apply decay
        scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for event in events:
            try:
                event_time = datetime.fromisoformat(event.timestamp)
                age_days = (now - event_time).total_seconds() / 86400
            except Exception:
                age_days = 0

            # Exponential decay weight
            decay = math.exp(-0.693 * age_days / DECAY_HALF_LIFE_DAYS)

            # Efficiency score: quality adjusted by cost
            # Free extractors get full confidence as score
            # Paid extractors get quality/cost ratio (capped at 1.0)
            if event.cost_usd > 0:
                efficiency = min(1.0, event.confidence / (event.cost_usd * 100 + 0.01))
            else:
                efficiency = event.confidence

            weighted_score = efficiency * decay
            scores[event.page_type][event.extractor].append(weighted_score)

        # Average scores
        result: dict[str, dict[str, float]] = {}
        for page_type, extractors in scores.items():
            result[page_type] = {}
            for ext, weighted_scores in extractors.items():
                if weighted_scores:
                    result[page_type][ext] = sum(weighted_scores) / len(weighted_scores)

        return result

    def save_rankings(self, rankings: dict | None = None) -> Path:
        """Compute and save rankings to disk."""
        if rankings is None:
            rankings = self.compute_rankings()

        output_path = RANKINGS_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "min_events": MIN_EVENTS_FOR_RANKING,
                    "rankings": rankings,
                },
                f,
                indent=2,
            )

        return output_path

    def load_rankings(self) -> dict[str, dict[str, float]]:
        """Load saved rankings from disk."""
        if not RANKINGS_FILE.is_file():
            return {}

        try:
            with open(RANKINGS_FILE) as f:
                data = json.load(f)
            return data.get("rankings", {})
        except Exception:
            return {}
