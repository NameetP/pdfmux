"""Tests for the smart result cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pdfmux.detect import PDFClassification
from pdfmux.pipeline import ConversionResult
from pdfmux.result_cache import ResultCache, file_hash, reset_default_cache


def _make_result(text: str = "hello world", pages: int = 2) -> ConversionResult:
    return ConversionResult(
        text=text,
        format="markdown",
        confidence=0.91,
        extractor_used="pymupdf4llm",
        page_count=pages,
        warnings=["one warning"],
        classification=PDFClassification(
            page_count=pages,
            is_digital=True,
            confidence=0.95,
        ),
        ocr_pages=[1],
    )


@pytest.fixture(autouse=True)
def _reset_default():
    reset_default_cache()
    yield
    reset_default_cache()


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def cache(cache_dir: Path) -> ResultCache:
    # Force-enable: bypass any PDFMUX_NO_CACHE env var that the suite
    # may set globally to keep state out of the user's home cache.
    return ResultCache(
        cache_dir, ttl_seconds=3600, max_bytes=10 * 1024 * 1024, enabled=True
    )


def test_file_hash_changes_when_contents_change(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    h1 = file_hash(p)
    p.write_bytes(b"abcd")
    h2 = file_hash(p)
    assert h1 != h2
    assert len(h1) == 64  # sha256 hex


def test_get_put_roundtrip(cache: ResultCache, digital_pdf: Path) -> None:
    assert cache.get(digital_pdf, "standard", "markdown") is None

    result = _make_result()
    cache.put(digital_pdf, "standard", "markdown", None, result)

    fetched = cache.get(digital_pdf, "standard", "markdown")
    assert fetched is not None
    assert fetched.text == result.text
    assert fetched.confidence == pytest.approx(result.confidence)
    assert fetched.page_count == result.page_count
    assert fetched.extractor_used == result.extractor_used
    assert fetched.warnings == result.warnings
    assert fetched.ocr_pages == result.ocr_pages
    assert fetched.classification.page_count == result.classification.page_count


def test_key_separates_quality_format_schema(cache: ResultCache, digital_pdf: Path) -> None:
    a = _make_result("A")
    b = _make_result("B")
    c = _make_result("C")
    d = _make_result("D")

    cache.put(digital_pdf, "standard", "markdown", None, a)
    cache.put(digital_pdf, "fast", "markdown", None, b)
    cache.put(digital_pdf, "standard", "json", None, c)
    cache.put(digital_pdf, "standard", "markdown", "preset_x", d)

    assert cache.get(digital_pdf, "standard", "markdown").text == "A"
    assert cache.get(digital_pdf, "fast", "markdown").text == "B"
    assert cache.get(digital_pdf, "standard", "json").text == "C"
    assert cache.get(digital_pdf, "standard", "markdown", "preset_x").text == "D"


def test_cache_miss_when_file_changes(cache: ResultCache, tmp_path: Path) -> None:
    import fitz

    pdf_path = tmp_path / "shifty.pdf"

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "first version", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    cache.put(pdf_path, "standard", "markdown", None, _make_result("v1"))
    assert cache.get(pdf_path, "standard", "markdown").text == "v1"

    # Replace the file content; ensure different bytes/mtime.
    time.sleep(0.01)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "second very different version", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    # New hash → miss.
    assert cache.get(pdf_path, "standard", "markdown") is None


def test_ttl_expiration(cache_dir: Path, digital_pdf: Path) -> None:
    cache = ResultCache(cache_dir, ttl_seconds=1, max_bytes=10 * 1024 * 1024, enabled=True)
    cache.put(digital_pdf, "standard", "markdown", None, _make_result())
    assert cache.get(digital_pdf, "standard", "markdown") is not None

    # Force expiry by rewriting the index with an old created_at.
    raw = json.loads((cache_dir / "index.json").read_text())
    for entry in raw["entries"].values():
        entry["created_at"] = time.time() - 3600
    (cache_dir / "index.json").write_text(json.dumps(raw))

    # Reload index.
    cache2 = ResultCache(cache_dir, ttl_seconds=1, max_bytes=10 * 1024 * 1024, enabled=True)
    assert cache2.get(digital_pdf, "standard", "markdown") is None


def test_size_eviction_lru(cache_dir: Path, digital_pdf: Path) -> None:
    # Tiny size limit forces eviction after the second insert.
    cache = ResultCache(cache_dir, ttl_seconds=3600, max_bytes=1, enabled=True)

    cache.put(digital_pdf, "standard", "markdown", None, _make_result("AAA"))
    time.sleep(0.01)
    cache.put(digital_pdf, "fast", "markdown", None, _make_result("BBB"))

    stats = cache.stats()
    # Over-cap entries should be evicted.
    assert stats["entries"] <= 1
    # The most recently inserted entry should still be available.
    assert cache.get(digital_pdf, "fast", "markdown") is not None


def test_clear_wipes_all(cache: ResultCache, digital_pdf: Path) -> None:
    cache.put(digital_pdf, "standard", "markdown", None, _make_result("A"))
    cache.put(digital_pdf, "fast", "markdown", None, _make_result("B"))

    assert cache.stats()["entries"] == 2
    removed = cache.clear()
    assert removed >= 2
    assert cache.stats()["entries"] == 0
    assert cache.get(digital_pdf, "standard", "markdown") is None


def test_stats_shape(cache: ResultCache, digital_pdf: Path) -> None:
    cache.put(digital_pdf, "standard", "markdown", None, _make_result("hello"))
    s = cache.stats()
    assert s["enabled"] is True
    assert s["entries"] == 1
    assert s["total_bytes"] > 0
    assert "cache_dir" in s
    assert "ttl_seconds" in s


def test_disabled_cache(cache_dir: Path, digital_pdf: Path) -> None:
    cache = ResultCache(cache_dir, enabled=False)
    cache.put(digital_pdf, "standard", "markdown", None, _make_result())
    assert cache.get(digital_pdf, "standard", "markdown") is None


def test_pipeline_uses_cache(monkeypatch, tmp_path, digital_pdf: Path) -> None:
    """Pipeline should hit the cache on the second call."""
    import pdfmux.pipeline as pipeline_module
    import pdfmux.result_cache as result_cache_module

    monkeypatch.setenv("PDFMUX_CACHE_DIR", str(tmp_path / "pdfmux_cache"))
    monkeypatch.delenv("PDFMUX_NO_CACHE", raising=False)
    result_cache_module.reset_default_cache()

    # Force a fresh, enabled cache regardless of env state.
    fresh = result_cache_module.ResultCache(tmp_path / "pdfmux_cache", enabled=True)
    monkeypatch.setattr(result_cache_module, "_default_cache", fresh)

    calls: list[int] = []

    real_classify = pipeline_module.classify

    def _spy_classify(path):
        calls.append(1)
        return real_classify(path)

    monkeypatch.setattr(pipeline_module, "classify", _spy_classify)

    r1 = pipeline_module.process(digital_pdf, output_format="markdown", quality="fast")
    r2 = pipeline_module.process(digital_pdf, output_format="markdown", quality="fast")

    assert r1.text == r2.text
    # First call extracts; second call hits cache and skips classify.
    assert len(calls) == 1


def test_pipeline_use_cache_false(monkeypatch, tmp_path, digital_pdf: Path) -> None:
    """`use_cache=False` bypasses the cache on both read and write."""
    import pdfmux.pipeline as pipeline_module
    import pdfmux.result_cache as result_cache_module

    monkeypatch.setenv("PDFMUX_CACHE_DIR", str(tmp_path / "pdfmux_cache"))
    monkeypatch.delenv("PDFMUX_NO_CACHE", raising=False)
    result_cache_module.reset_default_cache()

    pipeline_module.process(
        digital_pdf, output_format="markdown", quality="fast", use_cache=False
    )

    cache = result_cache_module.get_default_cache()
    assert cache.get(digital_pdf, "fast", "markdown") is None
