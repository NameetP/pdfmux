"""PDF type detection — classify a PDF to route it to the best extractor.

Opens the PDF with PyMuPDF. Inspects every page for:
- Text content (character count)
- Embedded images (count + coverage area)
- Line patterns (table detection)
- Text alignment patterns (table detection)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from pdfmux.errors import FileError


@dataclass
class PDFClassification:
    """Result of classifying a PDF document."""

    is_digital: bool = False
    is_scanned: bool = False
    is_mixed: bool = False
    is_graphical: bool = False  # Image-heavy — text in images that fast extraction misses
    has_tables: bool = False
    page_count: int = 0
    languages: list[str] = field(default_factory=list)
    confidence: float = 0.0
    digital_pages: list[int] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    graphical_pages: list[int] = field(default_factory=list)


def classify(file_path: str | Path) -> PDFClassification:
    """Classify a PDF to determine the best extraction strategy.

    Args:
        file_path: Path to the PDF file.

    Returns:
        PDFClassification with detection results.

    Raises:
        FileError: If the file doesn't exist or isn't a PDF.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileError(f"PDF not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise FileError(f"Not a PDF file: {file_path}")

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise FileError(f"Cannot open PDF: {file_path} — {e}") from e

    result = PDFClassification(page_count=len(doc))

    digital_pages = []
    scanned_pages = []
    graphical_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        text_len = len(text)
        image_count = len(images)

        # Classify into digital / scanned
        if text_len > 50:
            digital_pages.append(page_num)
        elif images:
            scanned_pages.append(page_num)
        else:
            digital_pages.append(page_num)

        # Detect graphical pages (pitch decks, infographics, slides)
        if (image_count >= 2 and text_len < 500) or (image_count >= 1 and text_len < 100):
            graphical_pages.append(page_num)

    result.digital_pages = digital_pages
    result.scanned_pages = scanned_pages
    result.graphical_pages = graphical_pages

    total = len(doc)
    if total == 0:
        result.confidence = 0.0
        doc.close()
        return result

    digital_ratio = len(digital_pages) / total

    if digital_ratio >= 0.8:
        result.is_digital = True
        result.confidence = min(0.95, digital_ratio)
    elif digital_ratio <= 0.2:
        result.is_scanned = True
        result.confidence = min(0.95, 1 - digital_ratio)
    else:
        result.is_mixed = True
        result.confidence = 0.7

    graphical_ratio = len(graphical_pages) / total
    if graphical_ratio > 0.25:
        result.is_graphical = True

    result.has_tables = _detect_tables(doc)

    doc.close()
    return result


def _detect_tables(doc: fitz.Document) -> bool:
    """Heuristic table detection using line analysis."""
    for page_num in range(min(len(doc), 5)):
        page = doc[page_num]
        drawings = page.get_drawings()

        horizontal_lines = 0
        vertical_lines = 0

        for drawing in drawings:
            for item in drawing.get("items", []):
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 50:
                        horizontal_lines += 1
                    elif abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) > 20:
                        vertical_lines += 1

        if horizontal_lines >= 3 and vertical_lines >= 2:
            return True

    for page_num in range(min(len(doc), 5)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split("\n")
        tab_lines = sum(1 for line in lines if "\t" in line or line.count("  ") >= 3)
        if tab_lines >= 3:
            return True

    return False
