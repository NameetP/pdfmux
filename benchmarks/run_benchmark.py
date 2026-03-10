#!/usr/bin/env python3
"""Benchmark: pdfmux vs PyMuPDF alone vs pymupdf4llm alone.

Measures: time, output size, page count, and empty-page detection.
Uses fast quality to avoid OCR model timeouts on large files.
"""
import json
import time
from pathlib import Path

from pdfmux.pipeline import process
from pdfmux.detect import classify

import pymupdf
import pymupdf4llm

BENCHMARK_DIR = Path(__file__).parent
PDF_FILES = sorted(BENCHMARK_DIR.glob("*.pdf"))

results = []

for pdf in PDF_FILES:
    print(f"\n{'='*60}")
    print(f"  {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)")
    print(f"{'='*60}", flush=True)

    # --- Classify ---
    classification = classify(pdf)
    page_count = classification.page_count
    has_tables = classification.has_tables
    is_scanned = classification.is_scanned
    graphical_count = len(classification.graphical_pages)

    # --- 1. PyMuPDF raw text ---
    t0 = time.time()
    doc = pymupdf.open(str(pdf))
    pymupdf_text = ""
    for page in doc:
        pymupdf_text += page.get_text()
    doc.close()
    pymupdf_time = time.time() - t0
    pymupdf_chars = len(pymupdf_text)

    doc2 = pymupdf.open(str(pdf))
    pymupdf_empty = sum(1 for p in doc2 if not p.get_text().strip())
    doc2.close()

    print(f"  PyMuPDF raw:     {pymupdf_time:.2f}s | {pymupdf_chars:,} chars | {pymupdf_empty} empty pages", flush=True)

    # --- 2. pymupdf4llm (markdown) ---
    t0 = time.time()
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf))
        p4l_time = time.time() - t0
        p4l_chars = len(md_text)
    except Exception as e:
        p4l_time = time.time() - t0
        p4l_chars = 0
        print(f"  pymupdf4llm err: {e}")

    print(f"  pymupdf4llm:     {p4l_time:.2f}s | {p4l_chars:,} chars", flush=True)

    # --- 3. pdfmux fast ---
    t0 = time.time()
    try:
        result = process(
            file_path=pdf,
            output_format="markdown",
            quality="fast",
        )
        pdfmux_time = time.time() - t0
        pdfmux_text = result.text
        pdfmux_chars = len(pdfmux_text)
        pdfmux_confidence = result.confidence
        pdfmux_extractor = result.extractor_used
        pdfmux_ocr_pages = len(result.ocr_pages)
    except Exception as e:
        pdfmux_time = time.time() - t0
        pdfmux_chars = 0
        pdfmux_confidence = 0
        pdfmux_extractor = f"error: {e}"
        pdfmux_ocr_pages = 0
        print(f"  pdfmux err:      {e}")

    print(f"  pdfmux fast:     {pdfmux_time:.2f}s | {pdfmux_chars:,} chars | {pdfmux_confidence*100:.0f}% conf | {pdfmux_extractor} | {pdfmux_ocr_pages} OCR'd", flush=True)

    # --- Quality delta ---
    if p4l_chars > 0 and pdfmux_chars > 0:
        delta = ((pdfmux_chars - p4l_chars) / p4l_chars) * 100
        print(f"  Delta vs p4llm:  {delta:+.1f}% chars")

    results.append({
        "file": pdf.name,
        "pages": page_count,
        "size_kb": round(pdf.stat().st_size / 1024),
        "has_tables": has_tables,
        "is_scanned": is_scanned,
        "graphical_pages": graphical_count,
        "pymupdf_raw": {
            "time_s": round(pymupdf_time, 2),
            "chars": pymupdf_chars,
            "empty_pages": pymupdf_empty,
        },
        "pymupdf4llm": {
            "time_s": round(p4l_time, 2),
            "chars": p4l_chars,
        },
        "pdfmux": {
            "time_s": round(pdfmux_time, 2),
            "chars": pdfmux_chars,
            "confidence": round(pdfmux_confidence * 100),
            "extractor": pdfmux_extractor,
            "ocr_pages": pdfmux_ocr_pages,
        },
    })

# Save results
output_path = BENCHMARK_DIR / "output" / "benchmark-results.json"
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*60}")
print(f"  BENCHMARK SUMMARY")
print(f"{'='*60}")
print(f"\n{'File':<35} {'Pages':>5} {'PyMuPDF':>8} {'p4llm':>8} {'pdfmux':>8} {'Conf':>5} {'Empty':>5}")
print("-" * 82)
for r in results:
    print(f"{r['file']:<35} {r['pages']:>5} {r['pymupdf_raw']['time_s']:>7.2f}s {r['pymupdf4llm']['time_s']:>7.2f}s {r['pdfmux']['time_s']:>7.2f}s {r['pdfmux']['confidence']:>4}% {r['pymupdf_raw']['empty_pages']:>5}")

# Totals
total_pages = sum(r['pages'] for r in results)
total_pymupdf_time = sum(r['pymupdf_raw']['time_s'] for r in results)
total_p4l_time = sum(r['pymupdf4llm']['time_s'] for r in results)
total_pdfmux_time = sum(r['pdfmux']['time_s'] for r in results)
total_empty = sum(r['pymupdf_raw']['empty_pages'] for r in results)
avg_conf = sum(r['pdfmux']['confidence'] for r in results) / len(results)
print("-" * 82)
print(f"{'TOTAL':<35} {total_pages:>5} {total_pymupdf_time:>7.2f}s {total_p4l_time:>7.2f}s {total_pdfmux_time:>7.2f}s {avg_conf:>4.0f}% {total_empty:>5}")

print(f"\nResults saved to {output_path}")
