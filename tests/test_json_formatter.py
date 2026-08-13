"""Tests for the JSON formatter's per-page structure.

The ``pages`` array is the JSON format's reason to exist — it carries per-page
text and the ocr flag. These tests pin that it comes from authoritative
per-page data when the pipeline supplies it, and never by re-slicing the joined
``content`` blob (which collapses multi-page documents to one page).
"""

from __future__ import annotations

import json

from pdfmux.formatters.json_fmt import format_json


def test_page_entries_are_authoritative() -> None:
    entries = [
        {"page": 1, "text": "Alpha", "ocr": False},
        {"page": 2, "text": "", "ocr": False},  # empty page kept as its own entry
        {"page": 3, "text": "Gamma", "ocr": True},
    ]
    out = json.loads(format_json("Alpha\n\nGamma", page_count=3, page_entries=entries))
    assert out["page_count"] == 3
    assert len(out["pages"]) == 3
    assert [p["page"] for p in out["pages"]] == [1, 2, 3]
    assert out["pages"][1]["text"] == ""
    assert out["pages"][2]["ocr"] is True


def test_page_entries_not_reconstructed_from_content() -> None:
    # content joined with "\n\n" (as the pipeline does) must NOT be re-split.
    entries = [{"page": i + 1, "text": f"P{i + 1}", "ocr": False} for i in range(4)]
    out = json.loads(format_json("P1\n\nP2\n\nP3\n\nP4", page_count=4, page_entries=entries))
    assert len(out["pages"]) == 4  # would be 1 if it split on "\n\n---\n\n"


def test_fallback_without_page_entries_splits_on_separator() -> None:
    text = "One\n\n---\n\nTwo\n\n---\n\nThree"
    out = json.loads(format_json(text, page_count=3))
    assert len(out["pages"]) == 3
    assert out["pages"][0]["text"] == "One"


def test_fallback_single_blob_when_no_separator() -> None:
    out = json.loads(format_json("no separators here", page_count=1))
    assert len(out["pages"]) == 1
    assert out["pages"][0]["page"] == 1


def test_control_chars_stripped_in_page_text() -> None:
    entries = [{"page": 1, "text": "a\x00b", "ocr": False}]
    out = json.loads(format_json("a\x00b", page_count=1, page_entries=entries))
    assert out["pages"][0]["text"] == "ab"
