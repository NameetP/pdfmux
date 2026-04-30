#!/usr/bin/env python3
"""Benchmark Gemma 4 as a local LLM backend for pdfmux.

Tests all available Gemma 4 model sizes on opendataloader-bench (200 PDFs).
Renders each page as an image, sends to Gemma 4 via Ollama, gets markdown.

Usage:
    python scripts/benchmark_gemma4.py                    # test all sizes
    python scripts/benchmark_gemma4.py --model gemma4:4b  # test specific size
    python scripts/benchmark_gemma4.py --fast             # test only 6 broken docs
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from statistics import fmean

import fitz  # PyMuPDF

BENCH_ROOT = Path.home() / "Projects/opendataloader-bench"
GT_DIR = BENCH_ROOT / "ground-truth/markdown"
PDF_DIR = BENCH_ROOT / "pdfs"
RESULTS_DIR = Path(__file__).parent.parent / "benchmarks/gemma4"

# Known broken docs — highest leverage targets
FAST_SUBSET = [
    "01030000000110", "01030000000122",  # TEDS=0
    "01030000000033", "01030000000107", "01030000000141", "01030000000198",  # MHS=0
]

GEMMA4_MODELS = [
    "gemma4:2b",
    "gemma4:4b",
    "gemma4:27b",  # MoE
    "gemma4:31b",  # Dense (if available)
]

PROMPT = """Convert this PDF page to clean Markdown. Rules:
- Preserve document structure: headings (# ## ###), paragraphs, lists
- Convert tables to markdown pipe tables (| col1 | col2 |)
- Preserve reading order (left-to-right, top-to-bottom)
- For multi-column layouts, read column by column (left column first, then right)
- Include all text content, don't skip anything
- Don't add commentary or metadata — just the content as markdown"""


def render_page_to_image(pdf_path: Path, page_num: int = 0, dpi: int = 200) -> bytes:
    """Render a PDF page to PNG bytes."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def query_ollama(model: str, image_bytes: bytes, prompt: str, timeout: int = 120) -> tuple[str, float]:
    """Send image to Ollama and get markdown response. Returns (text, latency_seconds)."""
    import urllib.request
    import urllib.error

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            latency = time.time() - start
            return data.get("response", ""), latency
    except (urllib.error.URLError, TimeoutError) as e:
        latency = time.time() - start
        return f"ERROR: {e}", latency


def evaluate_doc(gt_text: str, pred_text: str) -> dict:
    """Evaluate a single document. Returns scores dict."""
    sys.path.insert(0, str(BENCH_ROOT / "src"))
    from evaluator_reading_order import evaluate_reading_order
    from evaluator_table import evaluate_table
    from evaluator_heading_level import evaluate_heading_level

    nid, _ = evaluate_reading_order(gt_text, pred_text)
    teds, _ = evaluate_table(gt_text, pred_text)
    mhs, _ = evaluate_heading_level(gt_text, pred_text)

    components = [v for v in [nid, teds, mhs] if v is not None]
    overall = fmean(components) if components else None

    return {"overall": overall, "nid": nid, "teds": teds, "mhs": mhs}


def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        return True
    except Exception:
        return False


def list_available_models() -> list[str]:
    """List models available in Ollama."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def run_benchmark(model: str, doc_ids: list[str] | None = None):
    """Run the full benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model}")
    print(f"{'='*60}\n")

    if doc_ids:
        gt_files = [GT_DIR / f"{did}.md" for did in doc_ids]
        gt_files = [f for f in gt_files if f.exists()]
    else:
        gt_files = sorted(GT_DIR.glob("*.md"))

    results = []
    total_latency = 0
    success = 0
    failed = 0

    for i, gt_path in enumerate(gt_files, 1):
        doc_id = gt_path.stem
        pdf_path = PDF_DIR / f"{doc_id}.pdf"

        if not pdf_path.exists():
            print(f"  [{i}/{len(gt_files)}] {doc_id} — PDF not found, skipping")
            continue

        gt_text = gt_path.read_text(encoding="utf-8")

        # Render page to image
        try:
            img_bytes = render_page_to_image(pdf_path)
        except Exception as e:
            print(f"  [{i}/{len(gt_files)}] {doc_id} — render failed: {e}")
            failed += 1
            continue

        # Query Ollama
        pred_text, latency = query_ollama(model, img_bytes, PROMPT)
        total_latency += latency

        if pred_text.startswith("ERROR:"):
            print(f"  [{i}/{len(gt_files)}] {doc_id} — {pred_text} ({latency:.1f}s)")
            failed += 1
            continue

        # Evaluate
        scores = evaluate_doc(gt_text, pred_text)
        results.append({
            "document_id": doc_id,
            "scores": scores,
            "latency_s": round(latency, 2),
            "output_chars": len(pred_text),
        })
        success += 1

        overall_s = f"{scores['overall']:.3f}" if scores['overall'] is not None else "N/A"
        print(f"  [{i}/{len(gt_files)}] {doc_id} — overall={overall_s} ({latency:.1f}s)")

    # Aggregate
    if results:
        overall_vals = [r["scores"]["overall"] for r in results if r["scores"]["overall"] is not None]
        nid_vals = [r["scores"]["nid"] for r in results if r["scores"]["nid"] is not None]
        teds_vals = [r["scores"]["teds"] for r in results if r["scores"]["teds"] is not None]
        mhs_vals = [r["scores"]["mhs"] for r in results if r["scores"]["mhs"] is not None]
        latencies = [r["latency_s"] for r in results]

        summary = {
            "model": model,
            "docs_tested": len(gt_files),
            "success": success,
            "failed": failed,
            "overall_mean": fmean(overall_vals) if overall_vals else 0,
            "nid_mean": fmean(nid_vals) if nid_vals else 0,
            "teds_mean": fmean(teds_vals) if teds_vals else 0,
            "mhs_mean": fmean(mhs_vals) if mhs_vals else 0,
            "latency_mean_s": fmean(latencies),
            "latency_p95_s": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "total_time_s": round(total_latency, 1),
        }

        print(f"\n{'='*60}")
        print(f"RESULTS: {model}")
        print(f"{'='*60}")
        print(f"  Overall: {summary['overall_mean']:.4f}")
        print(f"  NID:     {summary['nid_mean']:.4f}")
        print(f"  TEDS:    {summary['teds_mean']:.4f}")
        print(f"  MHS:     {summary['mhs_mean']:.4f}")
        print(f"  Latency: {summary['latency_mean_s']:.1f}s/page (p95: {summary['latency_p95_s']:.1f}s)")
        print(f"  Total:   {summary['total_time_s']:.0f}s for {success} docs")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        model_slug = model.replace(":", "-").replace("/", "-")
        results_path = RESULTS_DIR / f"{model_slug}.json"
        results_path.write_text(json.dumps({
            "summary": summary,
            "documents": results,
        }, indent=2))
        print(f"  Saved to: {results_path}")

        return summary
    else:
        print("No results to report.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 for pdfmux")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--fast", action="store_true", help="Only test 6 broken docs")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if not check_ollama_running():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    available = list_available_models()

    if args.list:
        print("Available models:")
        for m in available:
            print(f"  {m}")
        return

    doc_ids = FAST_SUBSET if args.fast else None

    if args.model:
        models = [args.model]
    else:
        # Test all available Gemma 4 models
        models = [m for m in available if "gemma" in m.lower()]
        if not models:
            print("No Gemma models found. Pull one with: ollama pull gemma4:4b")
            print(f"Available models: {available}")
            sys.exit(1)

    print(f"Models to test: {models}")
    print(f"Docs: {'6 broken docs (fast mode)' if args.fast else 'all 200'}")
    print()

    all_summaries = []
    for model in models:
        summary = run_benchmark(model, doc_ids)
        if summary:
            all_summaries.append(summary)

    # Comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Overall':>8} {'NID':>8} {'TEDS':>8} {'MHS':>8} {'Lat(s)':>8}")
        print("-" * 60)
        for s in all_summaries:
            print(f"{s['model']:<20} {s['overall_mean']:>8.4f} {s['nid_mean']:>8.4f} {s['teds_mean']:>8.4f} {s['mhs_mean']:>8.4f} {s['latency_mean_s']:>8.1f}")

        # Add pdfmux baseline for comparison
        print(f"{'pdfmux (baseline)':<20} {'0.8671':>8} {'0.9102':>8} {'0.8841':>8} {'0.7385':>8} {'0.3':>8}")
        print(f"{'Gemini Flash':<20} {'0.9092':>8} {'0.9353':>8} {'0.9276':>8} {'0.8277':>8} {'2.0':>8}")


if __name__ == "__main__":
    main()
