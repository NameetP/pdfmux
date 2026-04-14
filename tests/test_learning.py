"""Tests for the data flywheel — telemetry + re-ranking."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

from pdfmux.router.learning import (
    ExtractionEvent,
    ReRanker,
    TelemetryCollector,
    is_telemetry_enabled,
)


class TestTelemetryEnabled:
    def test_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert is_telemetry_enabled() is False

    def test_enabled_with_local(self):
        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            assert is_telemetry_enabled() is True

    def test_enabled_with_true(self):
        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "true"}):
            assert is_telemetry_enabled() is True

    def test_disabled_with_random_value(self):
        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "off"}):
            assert is_telemetry_enabled() is False


class TestTelemetryCollector:
    def test_record_when_disabled(self, tmp_path):
        collector = TelemetryCollector(tmp_path / "telemetry.jsonl")
        event = ExtractionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            page_type="digital",
            extractor="pymupdf",
            provider=None,
            model=None,
            confidence=0.95,
            latency_ms=10,
            cost_usd=0.0,
            success=True,
        )
        with patch.dict("os.environ", {}, clear=True):
            collector.record(event)

        # File should NOT be created when telemetry is disabled
        assert not (tmp_path / "telemetry.jsonl").is_file()

    def test_record_when_enabled(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)
        event = ExtractionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            page_type="digital",
            extractor="pymupdf",
            provider=None,
            model=None,
            confidence=0.95,
            latency_ms=10,
            cost_usd=0.0,
            success=True,
        )
        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            collector.record(event)

        assert path.is_file()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["page_type"] == "digital"
        assert data["confidence"] == 0.95

    def test_record_multiple_events(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)

        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            for i in range(5):
                collector.record_extraction(
                    page_type="tables",
                    extractor="docling",
                    confidence=0.85 + i * 0.02,
                    latency_ms=1000 + i * 100,
                )

        assert collector.event_count() == 5

    def test_load_events(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)

        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            collector.record_extraction(
                page_type="scanned",
                extractor="rapidocr",
                confidence=0.75,
                latency_ms=500,
            )
            collector.record_extraction(
                page_type="tables",
                extractor="llm",
                confidence=0.90,
                latency_ms=2000,
                cost_usd=0.01,
                provider="claude",
                model="claude-sonnet",
            )

        events = collector.load_events()
        assert len(events) == 2
        assert events[0].page_type == "scanned"
        assert events[1].provider == "claude"

    def test_clear(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)

        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            collector.record_extraction(
                page_type="digital",
                extractor="pymupdf",
                confidence=0.95,
                latency_ms=10,
            )

        assert path.is_file()
        collector.clear()
        assert not path.is_file()

    def test_event_count_empty(self, tmp_path):
        collector = TelemetryCollector(tmp_path / "telemetry.jsonl")
        assert collector.event_count() == 0


class TestReRanker:
    def test_no_rankings_with_few_events(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)

        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            for i in range(10):  # below MIN_EVENTS_FOR_RANKING (50)
                collector.record_extraction(
                    page_type="digital",
                    extractor="pymupdf",
                    confidence=0.95,
                    latency_ms=10,
                )

        ranker = ReRanker(collector)
        rankings = ranker.compute_rankings()
        assert rankings == {}

    def test_rankings_with_enough_events(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)

        with patch.dict("os.environ", {"PDFMUX_TELEMETRY": "local"}):
            for i in range(60):
                collector.record_extraction(
                    page_type="tables",
                    extractor="docling" if i % 2 == 0 else "pymupdf",
                    confidence=0.90 if i % 2 == 0 else 0.60,
                    latency_ms=1000 if i % 2 == 0 else 10,
                )

        ranker = ReRanker(collector)
        rankings = ranker.compute_rankings()
        assert "tables" in rankings
        assert "docling" in rankings["tables"]
        assert "pymupdf" in rankings["tables"]
        # Docling should rank higher (higher confidence)
        assert rankings["tables"]["docling"] > rankings["tables"]["pymupdf"]

    def test_save_and_load_rankings(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        collector = TelemetryCollector(path)
        ranker = ReRanker(collector)

        test_rankings = {"digital": {"pymupdf": 0.95, "llm": 0.80}}

        with patch("pdfmux.router.learning.RANKINGS_FILE", tmp_path / "rankings.json"):
            saved_path = ranker.save_rankings(test_rankings)
            assert saved_path.is_file()

            loaded = ranker.load_rankings()
            assert loaded["digital"]["pymupdf"] == 0.95

    def test_load_rankings_missing_file(self, tmp_path):
        with patch("pdfmux.router.learning.RANKINGS_FILE", tmp_path / "missing.json"):
            ranker = ReRanker()
            rankings = ranker.load_rankings()
            assert rankings == {}


class TestExtractionEvent:
    def test_creation(self):
        event = ExtractionEvent(
            timestamp="2026-03-30T00:00:00Z",
            page_type="scanned",
            extractor="llm",
            provider="gemini",
            model="gemini-2.5-flash",
            confidence=0.88,
            latency_ms=2500,
            cost_usd=0.005,
            success=True,
        )
        assert event.page_type == "scanned"
        assert event.success is True

    def test_success_flag(self):
        event = ExtractionEvent(
            timestamp="2026-03-30T00:00:00Z",
            page_type="tables",
            extractor="pymupdf",
            provider=None,
            model=None,
            confidence=0.55,
            latency_ms=10,
            cost_usd=0.0,
            success=False,
        )
        assert event.success is False
