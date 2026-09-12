"""Generate supplemental corpus fixtures for the pdf-inspector eval.

The repo's eval/fixtures set (51 docs, labels.csv) covers TextBased and
image-only/Scanned cells plus robustness probes, but has no Mixed,
ImageBased (image-heavy deck), searchable-scan, 500+ page, or encrypted
docs. This script generates those cells deterministically with PyMuPDF so
every ground-truth label is set BY CONSTRUCTION, independent of both
systems under test (same principle as eval/build_fixtures.py).

Output: ./corpus-extra/*.pdf + ./manifest.csv covering BOTH the repo
fixtures (referenced in place) and the generated ones.

Idempotent, deterministic (fixed seed, fixed creation metadata).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import fitz  # PyMuPDF

SEED = 20260902
HERE = Path(__file__).parent
EXTRA = HERE / "corpus-extra"
FIXTURES = HERE.parent / "fixtures"
REPO_LABELS = HERE.parent / "labels.csv"
MANIFEST = HERE / "manifest.csv"

PARAS = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. How vexingly quick daft zebras jump.",
    "Quarterly logistics throughput rose 14 percent as the Jebel Ali "
    "corridor cleared its customs backlog ahead of schedule.",
    "PDF extraction quality depends on the source document structure: "
    "digital text exports cleanly, scanned images need OCR, and tables "
    "require dedicated extractors for fidelity.",
    "When extraction fails silently, downstream RAG pipelines retrieve "
    "empty chunks and confidently hallucinate answers from missing data.",
    "Verification is the differentiator: a classifier only routes, but a "
    "verifier certifies that what came out matches what went in.",
]


def _text_doc(pages: int, offset: int = 0) -> fitz.Document:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        body = "\n\n".join(PARAS[(i + offset + j) % len(PARAS)] for j in range(3))
        page.insert_text((72, 72), f"Section {i + 1}\n\n{body}", fontsize=11)
    return doc


def _page_png(offset: int, dpi: int = 110) -> bytes:
    """Render one deterministic text page to PNG bytes (a 'scan')."""
    src = _text_doc(1, offset=offset)
    pix = src[0].get_pixmap(dpi=dpi)
    data = pix.tobytes("png")
    src.close()
    return data


def _small_png(color: tuple[float, float, float]) -> bytes:
    """A small colored-figure PNG for slide-deck style pages."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(10, 10, 290, 190), color=color, fill=color)
    page.draw_circle(fitz.Point(150, 100), 60, color=(0, 0, 0), width=3)
    pix = page.get_pixmap(dpi=96)
    data = pix.tobytes("png")
    doc.close()
    return data


def make_scanned(path: Path, pages: int, offset: int) -> None:
    """Full-page raster pages, no text layer — a classic scan."""
    doc = fitz.open()
    for i in range(pages):
        png = _page_png(offset + i)
        page = doc.new_page()
        page.insert_image(page.rect, stream=png)
    doc.save(str(path))
    doc.close()


def make_mixed(path: Path, offset: int) -> None:
    """4 pages: digital text on 0 and 2, full-page raster on 1 and 3."""
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        if i % 2 == 0:
            body = "\n\n".join(PARAS[(offset + i + j) % len(PARAS)] for j in range(3))
            page.insert_text((72, 72), f"Chapter {i + 1}\n\n{body}", fontsize=11)
        else:
            page.insert_image(page.rect, stream=_page_png(offset + i))
    doc.save(str(path))
    doc.close()


def make_imagedeck(path: Path, offset: int) -> None:
    """4 slide-style pages: 3 figure images each + a short caption (<100 chars)."""
    colors = [(0.9, 0.3, 0.2), (0.2, 0.5, 0.9), (0.3, 0.8, 0.4)]
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        for k, c in enumerate(colors):
            r = fitz.Rect(60 + k * 170, 120, 200 + k * 170, 260)
            page.insert_image(r, stream=_small_png(c))
        page.insert_text((72, 80), f"Figure deck slide {i + 1} — chart {offset}", fontsize=14)
    doc.save(str(path))
    doc.close()


def make_searchable_scan(path: Path, offset: int) -> None:
    """Full-page raster + invisible text layer (render_mode=3) — a searchable scan."""
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_image(page.rect, stream=_page_png(offset + i))
        body = "\n\n".join(PARAS[(offset + i + j) % len(PARAS)] for j in range(3))
        page.insert_text(
            (72, 72), f"Section {i + 1}\n\n{body}", fontsize=11, render_mode=3
        )
    doc.save(str(path))
    doc.close()


def make_huge(path: Path, pages: int = 500) -> None:
    doc = _text_doc(pages)
    doc.save(str(path))
    doc.close()


def make_encrypted(path: Path) -> None:
    doc = _text_doc(2)
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# Ground-truth mapping for repo fixtures (labels.csv category -> 4-class axis or probe).
# Set by construction semantics documented in eval/build_fixtures.py.
REPO_GT = {
    "digital_single": ("TextBased", "classifiable"),
    "digital_multi": ("TextBased", "classifiable"),
    "digital_table": ("TextBased", "classifiable"),
    "arabic": ("TextBased", "classifiable"),  # digital Arabic text; sub-type has no 4-class counterpart
    "truncated_light": ("TextBased", "classifiable"),  # repairable digital text
    "image_only": ("Scanned", "classifiable"),  # full-page raster, needs OCR
    "zero_byte": ("invalid", "probe"),
    "html_as_pdf": ("invalid", "probe"),
    "truncated_heavy": ("invalid", "probe"),
    "micro_text": ("degenerate", "probe"),  # valid PDF, 1-char body — GT contestable
    "blank_page": ("degenerate", "probe"),  # valid PDF, zero content — no 4-class GT
}


def main() -> None:
    EXTRA.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []

    # 1. Repo fixtures, referenced in place.
    with open(REPO_LABELS, newline="") as f:
        for r in csv.DictReader(f):
            gt, tier = REPO_GT[r["category"]]
            p = FIXTURES / r["fixture"]
            rows.append(
                dict(
                    file=str(p.relative_to(HERE.parent.parent)),
                    source="repo-fixture (eval/fixtures, label by construction)",
                    category=r["category"],
                    gt_4class=gt,
                    tier=tier,
                    sha256=sha256(p),
                    note=r["note"],
                )
            )

    # 2. Generated fixtures.
    gen: list[tuple[str, str, str, str, str]] = []
    for i in range(5):
        p = EXTRA / f"gen-scanned-{i:02d}.pdf"
        make_scanned(p, pages=3, offset=i)
        gen.append((p.name, "gen_scanned", "Scanned", "classifiable", "3 full-page raster pages, no text layer"))
    for i in range(5):
        p = EXTRA / f"gen-mixed-{i:02d}.pdf"
        make_mixed(p, offset=i)
        gen.append((p.name, "gen_mixed", "Mixed", "classifiable", "4 pages: text on 0/2, raster on 1/3"))
    for i in range(5):
        p = EXTRA / f"gen-imagedeck-{i:02d}.pdf"
        make_imagedeck(p, offset=i)
        gen.append((p.name, "gen_imagedeck", "ImageBased", "classifiable", "4 slide pages, 3 figures + <100 char caption each"))
    for i in range(3):
        p = EXTRA / f"gen-searchable-{i:02d}.pdf"
        make_searchable_scan(p, offset=i)
        gen.append((p.name, "gen_searchable_scan", "TextBased", "classifiable", "raster + invisible text layer (render_mode=3); complete text extractable"))
    p = EXTRA / "gen-huge-500p.pdf"
    make_huge(p)
    gen.append((p.name, "gen_huge", "TextBased", "classifiable", "500 digital text pages — sampling/speed probe"))
    p = EXTRA / "gen-encrypted.pdf"
    make_encrypted(p)
    gen.append((p.name, "gen_encrypted", "encrypted", "probe", "AES-256, user password set — error-behavior probe"))

    for name, cat, gt, tier, note in gen:
        p = EXTRA / name
        rows.append(
            dict(
                file=str(p.relative_to(HERE.parent.parent)),
                source="generated (build_corpus_extra.py, label by construction)",
                category=cat,
                gt_4class=gt,
                tier=tier,
                sha256=sha256(p),
                note=note,
            )
        )

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_class = sum(1 for r in rows if r["tier"] == "classifiable")
    n_probe = sum(1 for r in rows if r["tier"] == "probe")
    print(f"manifest: {len(rows)} docs ({n_class} classifiable, {n_probe} probes)")
    from collections import Counter

    print(Counter(r["gt_4class"] for r in rows if r["tier"] == "classifiable"))


if __name__ == "__main__":
    main()
