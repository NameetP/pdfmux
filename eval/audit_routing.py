"""Audit scanned-vs-text routing against a real corpus — no labels required.

The problem with real documents is that they carry no ground truth. This works
around that: extract every page twice — once the way the router decided, once
with OCR forced — and let the disagreement label itself.

    under-route  router said "digital", but forced OCR finds much more text
                 -> the page was never OCR'd and extraction silently returned
                    a fraction of the document. This is the failure mode that
                    installing pdfmux[ocr] does NOT fix, because the page never
                    reaches the OCR stage.

    over-route   router said "scanned", but OCR finds no more than the fast
                 path already had -> OCR ran for nothing. Costs latency, not
                 correctness.

Also dumps the two geometry features the captioned-scan rule keys on
(image-area coverage, text-area coverage) per page, so the thresholds in
`detect.py` can be re-tuned against a real distribution instead of synthetic
fixtures.

SENSITIVITY: the under-route test needs OCR to recover materially more text
than the fast path, so it is least sensitive when the caption is a large
fraction of the document's real content. Validated against the synthetic
`edge-captioned-scan-*` fixtures on an unpatched `detect.py`: it caught the
60- and 120-character captions but not the 300/600/1200 ones, because those
fixtures OCR to only ~386 characters of body — a 300-char caption swamps the
ratio. Real datasheets carry thousands of characters of body text, so the test
is far sharper there. Treat the under-route count as a floor, not a total,
and read the CSV directly when a document looks suspicious.

PRIVACY: this script records only *measurements* — character counts, ratios,
page numbers, file names. It never writes, prints or stores document text. Safe
to run on confidential corpora and to share the CSV it produces.

CPU-only. No LLM calls, no network, no API keys, no spend.

Usage:
    python eval/audit_routing.py /path/to/pdfs --out audit.csv
    python eval/audit_routing.py /path/to/pdfs --max-pages 5 --limit 100

Requires OCR: pip install "pdfmux[ocr]"
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from pdfmux.detect import classify

# A page is flagged as under-routed when forced OCR yields at least this many
# times the characters the fast path produced, and at least MIN_GAIN more
# characters in absolute terms (so a 30-char page going to 90 isn't "3x worse").
UNDER_ROUTE_RATIO = 2.0
UNDER_ROUTE_MIN_GAIN = 200
# A scanned-routed page is flagged over-routed when OCR added essentially
# nothing over what the fast path already had.
OVER_ROUTE_RATIO = 1.1


@dataclass
class PageRow:
    doc: str
    page: int
    routed: str  # "digital" | "scanned" | "empty"
    fast_chars: int
    ocr_chars: int
    image_cov: float
    text_cov: float
    verdict: str  # "ok" | "under-route" | "over-route" | "ocr-failed"


@dataclass
class Totals:
    docs: int = 0
    pages: int = 0
    under: int = 0
    over: int = 0
    ocr_failed: int = 0
    doc_errors: list[str] = field(default_factory=list)
    under_docs: set[str] = field(default_factory=set)


def _coverage(page: fitz.Page, rects: list[fitz.Rect]) -> float:
    page_area = abs(page.rect)
    if not page_area:
        return 0.0
    covered = 0.0
    for r in rects:
        clipped = r & page.rect
        if not clipped.is_empty:
            covered += abs(clipped)
    return min(covered / page_area, 1.0)


def _features(page: fitz.Page) -> tuple[float, float]:
    """(image-area coverage, text-area coverage) — the two detect.py keys on."""
    img = _coverage(page, [fitz.Rect(i["bbox"]) for i in page.get_image_info()])
    txt = _coverage(page, [fitz.Rect(b[:4]) for b in page.get_text("blocks") if b[4].strip()])
    return img, txt


def audit_doc(path: Path, ocr, max_pages: int | None) -> list[PageRow]:
    rows: list[PageRow] = []
    cls = classify(path)
    scanned = set(cls.scanned_pages)
    empty = set(cls.empty_pages)

    doc = fitz.open(path)
    n = len(doc) if max_pages is None else min(len(doc), max_pages)
    for i in range(n):
        page = doc[i]
        fast = len(page.get_text("text").strip())
        img_cov, txt_cov = _features(page)
        routed = "scanned" if i in scanned else ("empty" if i in empty else "digital")

        try:
            ocr_text = ocr.extract_page(path, i)
            ocr_chars = len(ocr_text.strip())
            failed = False
        except Exception:
            ocr_chars, failed = 0, True

        if failed:
            verdict = "ocr-failed"
        elif routed == "digital" and ocr_chars >= max(
            fast * UNDER_ROUTE_RATIO, fast + UNDER_ROUTE_MIN_GAIN
        ):
            verdict = "under-route"
        elif routed == "scanned" and fast > 0 and ocr_chars <= fast * OVER_ROUTE_RATIO:
            verdict = "over-route"
        else:
            verdict = "ok"

        rows.append(
            PageRow(
                path.name, i, routed, fast, ocr_chars, round(img_cov, 4), round(txt_cov, 4), verdict
            )
        )
    doc.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus", type=Path, help="directory of PDFs (searched recursively)")
    ap.add_argument("--out", type=Path, default=Path("routing-audit.csv"))
    ap.add_argument(
        "--max-pages", type=int, default=None, help="cap pages audited per document (default: all)"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="cap number of documents (default: all)"
    )
    args = ap.parse_args()

    try:
        from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

        ocr = RapidOCRExtractor()
    except Exception as e:
        print(f"OCR unavailable ({e}). Install with: pip install 'pdfmux[ocr]'", file=sys.stderr)
        return 2

    pdfs = sorted(p for p in args.corpus.rglob("*.pdf") if p.is_file())
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no PDFs under {args.corpus}", file=sys.stderr)
        return 2

    print(f"auditing {len(pdfs)} documents (OCR forced on every page — this is slow)\n")
    t0 = time.perf_counter()
    tot = Totals()
    all_rows: list[PageRow] = []

    for idx, pdf in enumerate(pdfs, 1):
        try:
            rows = audit_doc(pdf, ocr, args.max_pages)
        except Exception as e:
            tot.doc_errors.append(f"{pdf.name}: {type(e).__name__}: {e}")
            continue
        all_rows.extend(rows)
        tot.docs += 1
        tot.pages += len(rows)
        for r in rows:
            if r.verdict == "under-route":
                tot.under += 1
                tot.under_docs.add(r.doc)
            elif r.verdict == "over-route":
                tot.over += 1
            elif r.verdict == "ocr-failed":
                tot.ocr_failed += 1
        if idx % 25 == 0 or idx == len(pdfs):
            print(f"  {idx}/{len(pdfs)} docs · {tot.under} under-routed pages so far")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["doc", "page", "routed", "fast_chars", "ocr_chars", "image_cov", "text_cov", "verdict"]
        )
        for r in all_rows:
            w.writerow(
                [
                    r.doc,
                    r.page,
                    r.routed,
                    r.fast_chars,
                    r.ocr_chars,
                    r.image_cov,
                    r.text_cov,
                    r.verdict,
                ]
            )

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 68}\nROUTING AUDIT\n{'=' * 68}")
    print(f"documents      {tot.docs} ({len(tot.doc_errors)} failed to open)")
    print(f"pages          {tot.pages}")
    print(
        f"under-routed   {tot.under} pages across {len(tot.under_docs)} documents"
        f"  <-- silent extraction failures"
    )
    print(f"over-routed    {tot.over} pages   (wasted OCR, cost not correctness)")
    print(f"ocr-failed     {tot.ocr_failed} pages")
    print(f"elapsed        {elapsed:.1f}s")

    if tot.under:
        worst = sorted(
            (r for r in all_rows if r.verdict == "under-route"),
            key=lambda r: r.ocr_chars - r.fast_chars,
            reverse=True,
        )[:15]
        print(
            f"\nworst under-routes (chars missed — these pages returned a fraction "
            f"of the document):\n{'-' * 68}"
        )
        print(f"{'doc':<40}{'pg':<5}{'fast':<8}{'ocr':<8}{'img_cov':<9}{'txt_cov'}")
        for r in worst:
            print(
                f"{r.doc[:38]:<40}{r.page:<5}{r.fast_chars:<8}{r.ocr_chars:<8}"
                f"{r.image_cov:<9.3f}{r.text_cov:.3f}"
            )
        print(
            "\nCheck img_cov/txt_cov against detect.py's _SCANNED_IMAGE_COVERAGE "
            "(0.75) and\n_SCANNED_MAX_TEXT_COVERAGE (0.10) — if real misses sit "
            "outside those bounds,\nthe thresholds need retuning against this "
            "distribution."
        )

    if tot.doc_errors:
        print(f"\ndocuments that failed to open ({len(tot.doc_errors)}):")
        for e in tot.doc_errors[:10]:
            print(f"  {e}")

    print(f"\nper-page detail written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
