"""Post-processing — clean extracted text and score confidence."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ProcessedResult:
    """Result of post-processing extracted text."""

    text: str
    confidence: float  # 0.0 - 1.0
    page_count: int
    warnings: list[str]


def clean_and_score(
    raw_text: str,
    page_count: int,
    *,
    extraction_limited: bool = False,
    graphical_page_count: int = 0,
    ocr_page_count: int = 0,
) -> ProcessedResult:
    """Clean extracted text and compute a confidence score.

    Cleaning steps:
    1. Normalize whitespace (collapse multiple blank lines)
    2. Fix common extraction artifacts
    3. Remove control characters
    4. Score confidence based on output quality signals

    Args:
        raw_text: Raw text from an extractor.
        page_count: Number of pages in the source PDF.
        extraction_limited: True when fast extraction was used on a graphical PDF
                            (known to miss image-embedded text).
        graphical_page_count: Number of pages detected as graphical/image-heavy.
        ocr_page_count: Number of pages re-extracted with OCR via multi-pass.
                        When > 0, the pipeline already recovered bad pages.

    Returns:
        ProcessedResult with cleaned text and confidence score.
    """
    warnings: list[str] = []

    text = raw_text

    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Fix broken words (hyphenation at line breaks)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Fix spaced-out text (common PDF artifact: "W i t h  o v e r" → "With over")
    text = _fix_spaced_text(text)

    # Remove trailing whitespace from lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Strip leading/trailing whitespace from the whole document
    text = text.strip()

    # Compute confidence score
    confidence = _compute_confidence(
        text,
        page_count,
        warnings,
        extraction_limited=extraction_limited,
        graphical_page_count=graphical_page_count,
        ocr_page_count=ocr_page_count,
    )

    return ProcessedResult(
        text=text,
        confidence=confidence,
        page_count=page_count,
        warnings=warnings,
    )


def _fix_spaced_text(text: str) -> str:
    """Fix spaced-out text — a common PDF extraction artifact.

    Some PDFs render text with individual character placement, producing
    output like "W i t h  o v e r  1 7  y e a r s" instead of "With over 17 years".

    Detection: a line where >50% of "words" are single characters.
    Fix: collapse single-char sequences into words.
    """
    lines = text.split("\n")
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue

        words = stripped.split()
        if len(words) < 5:
            fixed_lines.append(line)
            continue

        single_char_count = sum(1 for w in words if len(w) == 1)
        single_char_ratio = single_char_count / len(words)

        if single_char_ratio > 0.5:
            # This line is likely spaced-out text — collapse it
            # Strategy: join chars that are separated by single spaces
            # "W i t h  o v e r" → split on double-space → rejoin each group
            groups = re.split(r"  +", stripped)
            fixed_groups = []
            for group in groups:
                # If this group is single-spaced single chars, collapse
                parts = group.split(" ")
                if all(len(p) <= 1 for p in parts) and len(parts) >= 2:
                    fixed_groups.append("".join(parts))
                else:
                    fixed_groups.append(group)
            fixed_line = " ".join(fixed_groups)
            # Preserve leading whitespace
            leading = len(line) - len(line.lstrip())
            fixed_lines.append(" " * leading + fixed_line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def _compute_confidence(
    text: str,
    page_count: int,
    warnings: list[str],
    *,
    extraction_limited: bool = False,
    graphical_page_count: int = 0,
    ocr_page_count: int = 0,
) -> float:
    """Compute a confidence score for the extracted text.

    Checks:
    - Text completeness (chars per page)
    - Sparse page detection (pages with very little text)
    - Extraction limitation flag (graphical PDF + fast extractor)
    - Multi-pass OCR recovery (bonus when bad pages were re-extracted)
    - Encoding quality (mojibake detection)
    - Structure preservation (headings, lists)
    - Whitespace sanity
    """
    if not text or page_count == 0:
        warnings.append("Empty output — extraction may have failed")
        return 0.0

    score = 1.0

    # --- Known extraction limitation: graphical PDF with fast extractor ---
    # This is the strongest signal. We KNOW image content was missed.
    if extraction_limited and graphical_page_count > 0:
        # Penalty scales with how many pages are graphical
        graphical_ratio = graphical_page_count / page_count
        penalty = min(0.5, graphical_ratio * 0.6)
        score -= penalty
        warnings.append(
            f"{graphical_page_count} of {page_count} pages contain images with "
            f"text that could not be extracted. "
            f"Install pdfmux[ocr] or pdfmux[llm] for better results."
        )

    # --- Multi-pass OCR recovery ---
    # When multi-pass re-extracted bad pages, we get a confidence boost.
    # OCR text is noisier than digital extraction, so small penalty per OCR page,
    # but much better than having no text at all.
    if ocr_page_count > 0:
        ocr_ratio = ocr_page_count / page_count
        # Small penalty for OCR noise (0.02 per 10% of pages OCR'd)
        ocr_penalty = min(0.15, ocr_ratio * 0.2)
        score -= ocr_penalty

    # --- Text completeness: chars per page ---
    chars_per_page = len(text) / page_count
    if chars_per_page < 50:
        score -= 0.35
        warnings.append(f"Very little text extracted ({chars_per_page:.0f} chars/page)")
    elif chars_per_page < 200:
        score -= 0.2
        warnings.append(f"Low text density ({chars_per_page:.0f} chars/page)")
    elif chars_per_page < 500:
        score -= 0.05  # Slightly below average but could be normal (slides, short pages)

    # --- Sparse page detection ---
    # Estimate per-page density by splitting on common page separators
    # pymupdf4llm uses "-----" as page breaks
    page_chunks = re.split(r"\n-{3,}\n|\n#{1,2}\s+Page\s+\d+", text)
    if len(page_chunks) > 1:
        sparse_pages = sum(1 for chunk in page_chunks if len(chunk.strip()) < 100)
        empty_pages = sum(1 for chunk in page_chunks if len(chunk.strip()) < 20)
        sparse_ratio = sparse_pages / len(page_chunks)
        empty_ratio = empty_pages / len(page_chunks)

        if empty_ratio > 0.15:
            score -= 0.2
            warnings.append(f"{empty_pages} pages appear to have no extractable text")
        elif sparse_ratio > 0.25:
            score -= 0.1
            warnings.append(f"{sparse_pages} pages have very little text")

    # --- Encoding quality: mojibake detection ---
    mojibake_patterns = [r"â€", r"Ã©", r"Ã¨", r"â€™", r"ï¿½"]
    mojibake_count = sum(len(re.findall(p, text)) for p in mojibake_patterns)
    if mojibake_count > 10:
        score -= 0.3
        warnings.append(f"Possible encoding issues detected ({mojibake_count} patterns)")
    elif mojibake_count > 0:
        score -= 0.1

    # --- Structure: presence of Markdown headings is a good sign ---
    headings = len(re.findall(r"^#+\s", text, re.MULTILINE))
    if headings > 0:
        score += 0.05  # Bonus for preserved structure

    # --- Whitespace sanity: too many consecutive spaces = layout issues ---
    wide_spaces = len(re.findall(r"  {5,}", text))
    if wide_spaces > 20:
        score -= 0.1
        warnings.append("Excessive whitespace — layout extraction may be imperfect")

    return max(0.0, min(1.0, score))
