"""Build a 20-document, image-only ground-truth set for the model A/B.

Why image-only: pdfmux's LLM path (the thing a Sonnet-5-vs-GLM-vs-Kimi A/B is
supposed to tune) only fires on pages with no extractable native text — scans.
A digital PDF is read by PyMuPDF for free, no model involved, so it can't
discriminate between models. Each doc here is rendered as a raster image and
saved as a PDF with no text layer, which forces the OCR/vision model to do the
work. The source markdown IS the ground truth (independent of any model).

Deterministic: same text, same layout, same bytes on every run.

    python eval/build_ab_dataset.py        # writes eval/ab_datasets/*.pdf + *.gt.md

Output layout matches pdfmux.eval.runner's discover_datasets convention
(<name>.pdf + <name>.gt.md side by side), but this harness scores via
eval/ab_models.py, which drives the full extraction pipeline per model.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent / "ab_datasets"

# A4-ish canvas at ~150 DPI. Large enough that OCR/vision reads it cleanly.
PAGE_W, PAGE_H = 1240, 1754
MARGIN = 80
FONT_SIZE = 30
LINE_H = 44

# 20 documents spanning the failure modes a real scanned corpus hits:
# key-value blocks, tables, headings+prose, lists, and accented text.
SOURCE_DOCS: dict[str, str] = {
    "invoice-01": (
        "INVOICE #A-10432\n"
        "Date: 2026-03-14\n"
        "Bill To: Northgate Logistics\n"
        "Subtotal: 4,200.00 USD\n"
        "Tax (5%): 210.00 USD\n"
        "Total Due: 4,410.00 USD"
    ),
    "invoice-02": (
        "RECEIPT 8871\n"
        "Vendor: Blue Harbor Supplies\n"
        "Payment: Visa ending 4021\n"
        "Amount: 1,875.50 USD\n"
        "Status: PAID"
    ),
    "invoice-03": (
        "PURCHASE ORDER PO-55120\n"
        "Supplier: Cedar Mill Fabrication\n"
        "Ship Date: 2026-04-02\n"
        "Terms: Net 30\n"
        "Line Total: 12,640.00 USD"
    ),
    "kv-04": (
        "PATIENT INTAKE\n"
        "Name: Marcus Ellery\n"
        "DOB: 1988-11-02\n"
        "Policy: HLT-77410-B\n"
        "Provider: Riverside Clinic"
    ),
    "table-05": (
        "Q1 SALES BY REGION\n"
        "| Region | Units | Revenue |\n"
        "| North  | 1240  | 62000  |\n"
        "| South  | 980   | 49000  |\n"
        "| West   | 1510  | 75500  |"
    ),
    "table-06": (
        "INVENTORY SNAPSHOT\n"
        "| SKU    | On Hand | Reorder |\n"
        "| BX-100 | 45      | 20      |\n"
        "| BX-220 | 8       | 25      |\n"
        "| BX-330 | 130     | 40      |"
    ),
    "table-07": (
        "SHIPMENT MANIFEST\n"
        "| Crate | Weight | Dest |\n"
        "| C-1   | 220 kg | Lyon |\n"
        "| C-2   | 185 kg | Turin |\n"
        "| C-3   | 310 kg | Basel |"
    ),
    "table-08": (
        "STAFF ROSTER\n"
        "| Name    | Shift | Role |\n"
        "| Alvarez | AM    | Lead |\n"
        "| Bianchi | PM    | Tech |\n"
        "| Cheng   | AM    | Tech |"
    ),
    "prose-09": (
        "# Field Report: Bridge Inspection\n"
        "The north expansion joint shows minor\n"
        "corrosion but remains within tolerance.\n"
        "Deck drainage is functioning. Recommend\n"
        "reinspection in twelve months."
    ),
    "prose-10": (
        "# Meeting Minutes\n"
        "The committee approved the revised budget\n"
        "by a vote of five to two. Procurement will\n"
        "issue the tender next week. Action items\n"
        "were assigned to two subgroups."
    ),
    "prose-11": (
        "# Policy Note\n"
        "Employees may carry over up to ten unused\n"
        "leave days into the following year. Days\n"
        "beyond that limit are forfeited unless a\n"
        "written exception is filed with HR."
    ),
    "prose-12": (
        "# Abstract\n"
        "We present a method for detecting drift in\n"
        "streaming sensor data. The approach uses a\n"
        "sliding window and a lightweight statistic\n"
        "that flags anomalies without retraining."
    ),
    "list-13": (
        "SAFETY CHECKLIST\n"
        "1. Confirm power is isolated.\n"
        "2. Verify pressure gauge reads zero.\n"
        "3. Attach the grounding clamp.\n"
        "4. Log the start time."
    ),
    "list-14": (
        "PACKING LIST\n"
        "- Two coax cables\n"
        "- One power adapter\n"
        "- Mounting bracket kit\n"
        "- Printed quick-start guide"
    ),
    "list-15": (
        "ONBOARDING STEPS\n"
        "1. Sign the equipment agreement.\n"
        "2. Collect the access badge.\n"
        "3. Complete the security module.\n"
        "4. Meet the team lead."
    ),
    "mixed-16": (
        "SERVICE TICKET T-9043\n"
        "Priority: High\n"
        "| Item  | Qty | Cost |\n"
        "| Valve | 2   | 340  |\n"
        "| Seal  | 6   | 90   |\n"
        "Resolved: Yes"
    ),
    "mixed-17": (
        "# Expense Summary\n"
        "Traveler: Dana Whitfield\n"
        "| Day | Category | USD |\n"
        "| Mon | Airfare  | 410 |\n"
        "| Tue | Hotel    | 190 |\n"
        "Approved by: Finance"
    ),
    "mixed-18": (
        "WARRANTY CLAIM\n"
        "Product: Model 7 Pump\n"
        "Serial: PW-3391-KX\n"
        "Issue: Intermittent stall under load.\n"
        "Requested Action: Replace controller board."
    ),
    "intl-19": (
        "FACTURE 2026-118\n"
        "Client: Societe General du Nord\n"
        "Montant HT: 3 250,00 EUR\n"
        "TVA (20%): 650,00 EUR\n"
        "Total TTC: 3 900,00 EUR"
    ),
    "intl-20": (
        "RESUMEN DE PEDIDO\n"
        "Proveedor: Almacen Andino\n"
        "Articulo: Cafe en grano, 12 kg\n"
        "Precio unitario: 8,40 USD\n"
        "Importe total: 100,80 USD"
    ),
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render_image_pdf(text: str, out_path: Path, font) -> None:
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for line in text.split("\n"):
        draw.text((MARGIN, y), line, fill="black", font=font)
        y += LINE_H
    # Saving a PIL image as PDF produces an image-only page — no text layer.
    img.save(out_path, "PDF", resolution=150.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _load_font(FONT_SIZE)
    for name, text in SOURCE_DOCS.items():
        pdf_path = OUT_DIR / f"{name}.pdf"
        gt_path = OUT_DIR / f"{name}.gt.md"
        _render_image_pdf(text, pdf_path, font)
        gt_path.write_text(text + "\n", encoding="utf-8")
        print(f"  wrote {pdf_path.name}  ({pdf_path.stat().st_size} bytes)  + {gt_path.name}")
    print(f"\n{len(SOURCE_DOCS)} image-only docs + ground truth in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
