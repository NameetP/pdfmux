"""Benchmark pypdf vs pdfmux on the eval set's `bad` and `image_only` fixtures.

Question we want to answer for the blog post comparison section:
  On the failure modes that bit our v1 batch (image-only PDFs, truncated
  PDFs, blank PDFs), what does pypdf do that pdfmux does not?

Output: eval/outputs/pypdf_vs_pdfmux.csv with columns:
  fixture, label, category, pypdf_chars, pypdf_error, pdfmux_chars, pdfmux_conf
"""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stderr
from pathlib import Path

import pypdf

import pdfmux

EVAL_DIR = Path(__file__).parent
LABELS_PATH = EVAL_DIR / "labels.csv"
OUT_PATH = EVAL_DIR / "outputs" / "pypdf_vs_pdfmux.csv"


def _pypdf_extract(path: Path) -> tuple[int, str]:
    """Return (char_count, error_string)."""
    try:
        # Suppress pypdf's voluminous stderr complaints during the run.
        with redirect_stderr(io.StringIO()):
            reader = pypdf.PdfReader(str(path), strict=False)
            pages_text = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                pages_text.append(t)
        text = "\n\n".join(pages_text).strip()
        return len(text), ""
    except Exception as e:
        return 0, type(e).__name__ + ": " + str(e)[:200]


def _pdfmux_extract(path: Path) -> tuple[int, float, str]:
    """Return (char_count, confidence, error_string)."""
    try:
        data = pdfmux.extract_json(str(path), quality="fast")
        text = (data.get("content") or "").strip()
        return len(text), float(data.get("confidence", 0.0)), ""
    except Exception as e:
        return 0, 0.0, type(e).__name__ + ": " + str(e)[:200]


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixtures_dir = EVAL_DIR / "fixtures"

    # We only run the comparison on labels that match the blog's failure narrative:
    # image-only-no-OCR (the silent-fail mode), truncated, blank, html-as-pdf, 0-byte.
    relevant_categories = {
        "image_only",
        "truncated_heavy",
        "truncated_light",
        "blank_page",
        "html_as_pdf",
        "zero_byte",
    }

    out_rows = []
    for r in rows:
        if r["category"] not in relevant_categories:
            continue
        path = fixtures_dir / r["fixture"]

        py_chars, py_err = _pypdf_extract(path)
        mx_chars, mx_conf, mx_err = _pdfmux_extract(path)

        out_rows.append({
            "fixture": r["fixture"],
            "label": r["label"],
            "category": r["category"],
            "pypdf_chars": py_chars,
            "pypdf_error": py_err[:120],
            "pdfmux_chars": mx_chars,
            "pdfmux_conf": f"{mx_conf:.4f}" if not mx_err else "",
            "pdfmux_error": mx_err[:120],
        })

    fields = ["fixture", "label", "category", "pypdf_chars", "pypdf_error",
              "pdfmux_chars", "pdfmux_conf", "pdfmux_error"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # ----- print a readable summary for the blog comparison section -----
    by_cat: dict[str, list[dict]] = {}
    for r in out_rows:
        by_cat.setdefault(r["category"], []).append(r)

    print(f"Wrote {len(out_rows)} rows to {OUT_PATH}")
    print()
    print(f"{'category':<18} {'n':>3}  {'pypdf_recovered':>16}  {'pdfmux_recovered':>17}")
    print(f"{'-'*18:<18} {'-'*3:>3}  {'-'*16:>16}  {'-'*17:>17}")
    for cat, cat_rows in sorted(by_cat.items()):
        n = len(cat_rows)
        pypdf_ok = sum(1 for r in cat_rows if r["pypdf_chars"] and int(r["pypdf_chars"]) > 50)
        mx_ok = sum(1 for r in cat_rows if r["pdfmux_chars"] and int(r["pdfmux_chars"]) > 50)
        print(f"{cat:<18} {n:>3}  {pypdf_ok:>10} of {n:<3}  {mx_ok:>11} of {n:<3}")
    print()
    print("Detail (first 20 rows):")
    for r in out_rows[:20]:
        print(f"  {r['fixture']:35s} {r['category']:<18} "
              f"pypdf={r['pypdf_chars']:>4} pdfmux={r['pdfmux_chars']:>4} "
              f"conf={r['pdfmux_conf']:<6} {(r['pypdf_error'] or r['pdfmux_error'])[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
