"""OCR-based table extraction from image regions in PDFs.

For pages where a table is embedded as an image (not as text),
this module renders the image region, OCRs it, and reconstructs
a markdown pipe table from the OCR results using spatial clustering.

This is ONLY used as a last-resort fallback when no other extraction
method finds tables on a page that has substantial images.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _cluster_values(values: list[float], gap: float = 30.0) -> list[list[int]]:
    """Cluster sorted values by gap threshold. Returns groups of indices."""
    if not values:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    groups: list[list[int]] = [[indexed[0][0]]]

    for i in range(1, len(indexed)):
        if indexed[i][1] - indexed[i - 1][1] > gap:
            groups.append([])
        groups[-1].append(indexed[i][0])

    return groups


def ocr_image_to_table(
    file_path: str | Path,
    page_num: int,
    image_bbox: tuple[float, float, float, float],
    dpi: int = 300,
) -> str | None:
    """Extract a table from an image region via OCR.

    Returns a markdown pipe table string, or None if the OCR output
    doesn't look like a structured table (< 10 data rows, inconsistent
    columns, etc.).
    """
    try:
        import fitz
        from rapidocr import RapidOCR
    except ImportError:
        return None

    try:
        doc = fitz.open(str(file_path))
        if page_num >= len(doc):
            return None

        page = doc[page_num]
        clip = fitz.Rect(*image_bbox)

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_bytes = pix.tobytes("png")

        engine = RapidOCR()
        result = engine(img_bytes)

        if result.boxes is None or result.txts is None or len(result.txts) < 8:
            doc.close()
            return None

        # Reconstruct table from OCR boxes
        cells = []
        for box, txt in zip(result.boxes, result.txts):
            x_center = sum(p[0] for p in box) / 4
            y_center = sum(p[1] for p in box) / 4
            cells.append({"x": x_center, "y": y_center, "text": txt.strip()})

        # Cluster by y (rows) and x (columns)
        y_values = [c["y"] for c in cells]
        x_values = [c["x"] for c in cells]

        row_groups = _cluster_values(y_values, gap=25.0)
        col_groups = _cluster_values(x_values, gap=80.0)

        n_cols = len(col_groups)
        n_rows = len(row_groups)

        # Strict validation: must look like a real data table
        if n_cols < 2 or n_rows < 4:
            doc.close()
            return None

        # Build column centers
        col_centers = sorted(
            sum(x_values[i] for i in group) / len(group) for group in col_groups
        )

        # Assign cells to (row, col)
        rows: dict[int, dict[int, str]] = {}
        for cell in cells:
            col_idx = min(range(n_cols), key=lambda c: abs(cell["x"] - col_centers[c]))
            row_idx = min(
                range(n_rows),
                key=lambda r: abs(
                    cell["y"]
                    - sum(y_values[i] for i in row_groups[r]) / len(row_groups[r])
                ),
            )
            rows.setdefault(row_idx, {})[col_idx] = cell["text"]

        if len(rows) < 4:
            doc.close()
            return None

        # Build markdown table
        sorted_row_keys = sorted(rows.keys())
        lines = []
        for i, rk in enumerate(sorted_row_keys):
            row_data = rows[rk]
            cells_str = [row_data.get(c, "") for c in range(n_cols)]
            lines.append("| " + " | ".join(cells_str) + " |")
            if i == 0:
                lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")

        doc.close()

        # Final validation: consistent pipe count
        pipe_counts = [l.count("|") for l in lines if "---" not in l]
        if len(set(pipe_counts)) > 2:
            return None

        # Validate: data rows must have substantial numeric content
        # (real data tables have numbers; chart OCR has labels/text)
        import re
        total_data_cells = 0
        numeric_data_cells = 0
        for l in lines[2:]:  # skip header + separator
            for cell in l.split("|"):
                cell = cell.strip()
                if not cell:
                    continue
                total_data_cells += 1
                if re.match(r"^[\d.E\-+,]+$", cell.replace(" ", "")):
                    numeric_data_cells += 1
        if total_data_cells > 0:
            numeric_pct = numeric_data_cells / total_data_cells
            if numeric_pct < 0.9:  # < 90% numeric = probably not a data table
                return None

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"Image table OCR failed: {e}")
        return None
