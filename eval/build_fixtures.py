"""Generate the labeled fixture set for confidence calibration.

Writes 50+ PDFs to eval/fixtures/ and a labels.csv with binary ground truth
(`good` or `bad`). Each fixture's label is set by *construction*, not by
running pdfmux on it — so the eval set is independent of the system under
test.

Label semantics:
  good — a reasonable extractor (PyMuPDF + RapidOCR) should produce
         coherent text from this document.
  bad  — no extractor can produce useful text. The correct extractor
         output is "I extracted nothing usable, confidence near zero."

Run with:
    python eval/build_fixtures.py

Idempotent — overwrites existing fixtures + labels. Deterministic via
the FIXED_SEED below.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import fitz

FIXED_SEED = 20260502
EVAL_DIR = Path(__file__).parent
FIXTURES_DIR = EVAL_DIR / "fixtures"
LABELS_PATH = EVAL_DIR / "labels.csv"

# Reference paragraphs we'll use as PDF body text. Multiple short paragraphs
# matter for testing the audit's character-density and word-structure checks.
LOREM_PARAS = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. How vexingly quick daft zebras jump.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "PDF extraction quality depends on the source document's structure: "
    "digital text exports cleanly, scanned images need OCR, and tables "
    "require dedicated extractors for fidelity.",
    "Proper extraction returns markdown with headings preserved, tables "
    "rendered as pipe-tables, and reading order matching visual layout.",
    "When extraction fails silently, downstream RAG pipelines retrieve "
    "empty chunks and confidently hallucinate answers from missing data.",
]

ARABIC_TEXT = "النص العربي في هذا الملف بسيط لكنه كافٍ لاختبار معالجة النصوص"


def _make_digital(path: Path, paragraphs: list[str], pages: int = 1) -> None:
    """Build a clean digital PDF with the given paragraphs across N pages."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        body = "\n\n".join(paragraphs[i % len(paragraphs)] for _ in range(2))
        page.insert_text((72, 72), body, fontsize=11)
    doc.save(str(path))
    doc.close()


def _make_truncated(path: Path, base_paragraphs: list[str], keep_fraction: float) -> None:
    """Build a PDF then truncate it to keep_fraction of its bytes."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "\n\n".join(base_paragraphs), fontsize=11)
    full = doc.tobytes()
    doc.close()
    cutoff = max(64, int(len(full) * keep_fraction))
    path.write_bytes(full[:cutoff])


def _make_image_only(path: Path) -> None:
    """A PDF whose only content is a rendered raster — no text layer."""
    # Build a digital PDF, render its first page as PNG, then construct a new
    # PDF whose page is just the image. PyMuPDF without OCR returns no text.
    src = fitz.open()
    page = src.new_page()
    page.insert_text(
        (72, 72),
        "\n\n".join(LOREM_PARAS[:3]),
        fontsize=11,
    )
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    src.close()

    out = fitz.open()
    out_page = out.new_page(width=pix.width, height=pix.height)
    out_page.insert_image(out_page.rect, stream=img_bytes)
    out.save(str(path))
    out.close()


def _make_micro_text(path: Path, content: str) -> None:
    """A PDF with a tiny scrap of text — below pdfmux's 'empty' threshold."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), content, fontsize=11)
    doc.save(str(path))
    doc.close()


def _make_html_as_pdf(path: Path) -> None:
    path.write_text(
        "<html><head><title>not a pdf</title></head>"
        "<body><h1>HTML pretending</h1><p>This is not a PDF.</p></body></html>",
        encoding="utf-8",
    )


def _make_arabic(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    try:
        page.insert_text((72, 72), ARABIC_TEXT, fontsize=14)
    except Exception:
        # PyMuPDF default font may reject some glyphs; fall back to ASCII.
        page.insert_text((72, 72), "fallback content for arabic fixture", fontsize=11)
    doc.save(str(path))
    doc.close()


def _make_table_heavy(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    rows = ["| Col A | Col B | Col C |", "|-------|-------|-------|"]
    for i in range(8):
        rows.append(f"| row {i} | {i * 3} | data {i} |")
    body = "Title\n\nSummary paragraph for the table-heavy fixture.\n\n" + "\n".join(rows)
    page.insert_text((72, 72), body, fontsize=10)
    doc.save(str(path))
    doc.close()


def _label_row(stem: str, label: str, category: str, note: str = "") -> dict[str, str]:
    return {
        "fixture": f"{stem}.pdf",
        "label": label,
        "category": category,
        "note": note,
    }


def main() -> None:
    random.seed(FIXED_SEED)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Clear stale fixtures so re-runs are clean.
    for old in FIXTURES_DIR.glob("*.pdf"):
        old.unlink()
    for old in FIXTURES_DIR.glob("*.html"):
        old.unlink()

    rows: list[dict[str, str]] = []

    # --- clearly good (18) -----------------------------------------------
    for i in range(6):
        stem = f"good-digital-{i:02d}"
        _make_digital(FIXTURES_DIR / f"{stem}.pdf", LOREM_PARAS, pages=1)
        rows.append(_label_row(stem, "good", "digital_single", "1 page, clean digital text"))

    for i in range(6):
        stem = f"good-multipage-{i:02d}"
        _make_digital(FIXTURES_DIR / f"{stem}.pdf", LOREM_PARAS, pages=3)
        rows.append(_label_row(stem, "good", "digital_multi", "3 pages, clean digital text"))

    for i in range(3):
        stem = f"good-table-{i:02d}"
        _make_table_heavy(FIXTURES_DIR / f"{stem}.pdf")
        rows.append(_label_row(stem, "good", "digital_table", "table-heavy markdown body"))

    for i in range(3):
        stem = f"good-arabic-{i:02d}"
        _make_arabic(FIXTURES_DIR / f"{stem}.pdf")
        rows.append(_label_row(stem, "good", "arabic", "Arabic-only content"))

    # --- clearly bad (18) ------------------------------------------------
    for i in range(5):
        stem = f"bad-zero-byte-{i:02d}"
        (FIXTURES_DIR / f"{stem}.pdf").write_bytes(b"")
        rows.append(_label_row(stem, "bad", "zero_byte", "0-byte file"))

    for i in range(5):
        stem = f"bad-html-as-pdf-{i:02d}"
        _make_html_as_pdf(FIXTURES_DIR / f"{stem}.pdf")
        rows.append(_label_row(stem, "bad", "html_as_pdf", "HTML body, .pdf extension"))

    for i in range(4):
        # Heavily truncated — keep only ~15% of bytes. Header probably gone.
        stem = f"bad-trunc-15pct-{i:02d}"
        _make_truncated(FIXTURES_DIR / f"{stem}.pdf", LOREM_PARAS[:2], keep_fraction=0.15)
        rows.append(_label_row(stem, "bad", "truncated_heavy", "15% bytes kept"))

    for i in range(4):
        stem = f"bad-micro-text-{i:02d}"
        _make_micro_text(FIXTURES_DIR / f"{stem}.pdf", "x")
        rows.append(_label_row(stem, "bad", "micro_text", "single character body"))

    # --- edge / depends-on-stack (14) ------------------------------------
    # PyMuPDF's xref repair handles ~70% truncation cleanly — these should
    # extract most of the original text. Label as `good`.
    for i in range(5):
        stem = f"edge-trunc-70pct-{i:02d}"
        _make_truncated(FIXTURES_DIR / f"{stem}.pdf", LOREM_PARAS[:3], keep_fraction=0.70)
        rows.append(_label_row(stem, "good", "truncated_light", "70% bytes kept — PyMuPDF repair"))

    # Image-only: extraction works only if RapidOCR is installed. Label as
    # `good` because the eval is run with pdfmux[ocr] in the standard config.
    for i in range(5):
        stem = f"edge-image-only-{i:02d}"
        _make_image_only(FIXTURES_DIR / f"{stem}.pdf")
        rows.append(_label_row(stem, "good", "image_only", "rendered raster, needs OCR"))

    # Empty PDF (one blank page, no content). Label `bad` — there's nothing
    # to extract, but the file is structurally valid.
    for i in range(4):
        stem = f"bad-blank-page-{i:02d}"
        d = fitz.open()
        d.new_page()
        d.save(str(FIXTURES_DIR / f"{stem}.pdf"))
        d.close()
        rows.append(_label_row(stem, "bad", "blank_page", "structurally valid, zero text"))

    # Write labels.csv
    fields = ["fixture", "label", "category", "note"]
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    good = sum(1 for r in rows if r["label"] == "good")
    bad = sum(1 for r in rows if r["label"] == "bad")
    print(f"Wrote {len(rows)} fixtures to {FIXTURES_DIR}")
    print(f"  good: {good}")
    print(f"  bad:  {bad}")
    print(f"Labels written to {LABELS_PATH}")


if __name__ == "__main__":
    main()
