"""Tests for the streaming extraction generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pdfmux.cli import app
from pdfmux.streaming import StreamEvent, process_streaming


def _collect(events):
    return list(events)


def test_yields_event_objects(digital_pdf: Path) -> None:
    events = _collect(process_streaming(digital_pdf, quality="fast"))
    assert events
    for ev in events:
        assert isinstance(ev, StreamEvent)


def test_first_event_is_classified(digital_pdf: Path) -> None:
    events = _collect(process_streaming(digital_pdf, quality="fast"))
    assert events[0].type == "classified"
    payload = events[0].data
    assert payload["page_count"] >= 1
    assert isinstance(payload["page_types"], list)
    assert len(payload["page_types"]) == payload["page_count"]


def test_last_event_is_complete(digital_pdf: Path) -> None:
    events = _collect(process_streaming(digital_pdf, quality="fast"))
    assert events[-1].type == "complete"
    payload = events[-1].data
    assert "total_confidence" in payload
    assert isinstance(payload["ocr_pages"], list)
    assert "extractor_used" in payload
    assert "warnings" in payload
    assert payload["page_count"] == events[0].data["page_count"]


def test_one_page_event_per_page(multi_page_pdf: Path) -> None:
    events = _collect(process_streaming(multi_page_pdf, quality="fast"))
    page_events = [e for e in events if e.type == "page"]
    page_count = events[0].data["page_count"]
    assert len(page_events) == page_count

    # Page events arrive in document order.
    assert [e.data["page_num"] for e in page_events] == list(range(page_count))


def test_page_event_shape(digital_pdf: Path) -> None:
    events = _collect(process_streaming(digital_pdf, quality="fast"))
    page_events = [e for e in events if e.type == "page"]
    assert page_events
    pe = page_events[0]
    for key in ("page_num", "text", "confidence", "quality", "extractor", "ocr_applied"):
        assert key in pe.data
    assert isinstance(pe.data["confidence"], float)
    assert pe.data["quality"] in {"good", "bad", "empty"}


def test_event_order_classified_pages_complete(digital_pdf: Path) -> None:
    events = _collect(process_streaming(digital_pdf, quality="fast"))
    types = [e.type for e in events]

    # 'classified' precedes the first 'page'; 'complete' follows the last.
    classified_idx = types.index("classified")
    complete_idx = len(types) - 1 - list(reversed(types)).index("complete")
    page_indices = [i for i, t in enumerate(types) if t == "page"]
    assert page_indices, "expected at least one page event"
    assert classified_idx < page_indices[0]
    assert page_indices[-1] < complete_idx
    assert types[-1] == "complete"
    assert types[0] == "classified"


def test_to_dict_is_json_serialisable(digital_pdf: Path) -> None:
    events = _collect(process_streaming(digital_pdf, quality="fast"))
    for ev in events:
        d = ev.to_dict()
        assert "type" in d
        assert "data" in d
        # Round-trip through JSON to verify all values serialise.
        json.dumps(d, ensure_ascii=False)


def test_missing_file_raises(tmp_path: Path) -> None:
    from pdfmux.errors import FileError

    with pytest.raises(FileError):
        list(process_streaming(tmp_path / "nope.pdf", quality="fast"))


def test_cli_stream_command_outputs_ndjson(tmp_path: Path, digital_pdf: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["stream", str(digital_pdf), "--quality", "fast"])
    assert result.exit_code == 0, result.output

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, "expected NDJSON output"
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["type"] == "classified"
    assert parsed[-1]["type"] == "complete"
    assert any(p["type"] == "page" for p in parsed)


def test_cli_stream_writes_to_file(tmp_path: Path, digital_pdf: Path) -> None:
    out_path = tmp_path / "stream.ndjson"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["stream", str(digital_pdf), "--quality", "fast", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()

    lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["type"] == "classified"
    assert parsed[-1]["type"] == "complete"
