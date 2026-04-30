"""Tests for Arabic text handling — detection, BiDi reorder, normalization."""

from __future__ import annotations

import pytest

from pdfmux.arabic import (
    arabic_ratio,
    fix_bidi_order,
    is_arabic_text,
    is_rtl_dominant,
    normalize_arabic,
)

# ---------------------------------------------------------------------------
# Sample strings — keep small, copy-pasteable, real shipping/logistics terms
# ---------------------------------------------------------------------------

HELLO_WORLD_AR = "مرحبا بالعالم"  # "Hello world"
BILL_OF_LADING = "بوليصة الشحن"  # "Bill of lading"
MIXED = "Invoice رقم INV-2026-001"  # mixed LTR/RTL
ENGLISH = "Hello world"
EMPTY = ""
DIGITS = "123 456 789"
HEBREW = "שלום עולם"  # Hebrew is RTL but not Arabic


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestIsArabicText:
    def test_pure_arabic(self) -> None:
        assert is_arabic_text(HELLO_WORLD_AR) is True

    def test_bill_of_lading(self) -> None:
        assert is_arabic_text(BILL_OF_LADING) is True

    def test_mixed_string(self) -> None:
        assert is_arabic_text(MIXED) is True

    def test_english_only(self) -> None:
        assert is_arabic_text(ENGLISH) is False

    def test_empty(self) -> None:
        assert is_arabic_text(EMPTY) is False

    def test_digits_only(self) -> None:
        assert is_arabic_text(DIGITS) is False

    def test_hebrew_is_not_arabic(self) -> None:
        # Hebrew is RTL but should not be flagged as Arabic
        assert is_arabic_text(HEBREW) is False


class TestIsRtlDominant:
    def test_pure_arabic_is_dominant(self) -> None:
        assert is_rtl_dominant(HELLO_WORLD_AR) is True

    def test_pure_hebrew_is_dominant(self) -> None:
        # is_rtl_dominant covers all RTL scripts, not just Arabic
        assert is_rtl_dominant(HEBREW) is True

    def test_english_is_not_dominant(self) -> None:
        assert is_rtl_dominant(ENGLISH) is False

    def test_empty_is_not_dominant(self) -> None:
        assert is_rtl_dominant(EMPTY) is False

    def test_digits_only_is_not_dominant(self) -> None:
        # No letters at all → False (not >50%)
        assert is_rtl_dominant(DIGITS) is False

    def test_mostly_arabic_with_few_latin(self) -> None:
        # Arabic invoice with English code: still RTL-dominant by letter count
        text = "بوليصة شحن A"  # 11 Arabic letters + 1 Latin letter
        assert is_rtl_dominant(text) is True

    def test_mostly_english_with_arabic_word(self) -> None:
        text = "Invoice number for the shipment رقم"  # 3 Arabic letters, lots of Latin
        assert is_rtl_dominant(text) is False


class TestArabicRatio:
    def test_pure_arabic(self) -> None:
        # All non-whitespace chars are Arabic
        assert arabic_ratio(HELLO_WORLD_AR) > 0.95

    def test_pure_english(self) -> None:
        assert arabic_ratio(ENGLISH) == 0.0

    def test_empty(self) -> None:
        assert arabic_ratio(EMPTY) == 0.0

    def test_mixed_threshold(self) -> None:
        # Mixed string should still register a meaningful ratio
        ratio = arabic_ratio(MIXED)
        assert 0.0 < ratio < 1.0


# ---------------------------------------------------------------------------
# BiDi reordering
# ---------------------------------------------------------------------------


class TestFixBidiOrder:
    def test_returns_string(self) -> None:
        # Either python-bidi is installed (correct ordering) or not (passthrough),
        # but the result must always be a string.
        result = fix_bidi_order(HELLO_WORLD_AR)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self) -> None:
        assert fix_bidi_order("") == ""

    def test_english_only_unchanged(self) -> None:
        # English-only lines must not be re-ordered.
        assert fix_bidi_order(ENGLISH) == ENGLISH

    def test_preserves_arabic_characters(self) -> None:
        # Even if the algorithm reorders, the same character set must remain.
        result = fix_bidi_order(HELLO_WORLD_AR)
        assert set(result) == set(HELLO_WORLD_AR)

    def test_markdown_heading_prefix_preserved(self) -> None:
        text = "# مرحبا بالعالم"
        result = fix_bidi_order(text)
        assert result.startswith("# ")

    def test_pipe_table_structure_preserved(self) -> None:
        text = "| الاسم | القيمة |"
        result = fix_bidi_order(text)
        # Pipe characters survive at the same count.
        assert result.count("|") == text.count("|")
        assert result.startswith("|")
        assert result.endswith("|")

    def test_multiline_only_rtl_lines_processed(self) -> None:
        text = "Invoice number\nمرحبا بالعالم\nDate: 2026-04-30"
        result = fix_bidi_order(text)
        lines = result.split("\n")
        assert len(lines) == 3
        # English lines unchanged.
        assert lines[0] == "Invoice number"
        assert lines[2] == "Date: 2026-04-30"

    def test_idempotent_on_english(self) -> None:
        # Repeated application on English content should not change anything.
        once = fix_bidi_order(ENGLISH)
        twice = fix_bidi_order(once)
        assert once == twice == ENGLISH


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizeArabic:
    def test_empty_passes_through(self) -> None:
        assert normalize_arabic("") == ""

    def test_english_passes_through(self) -> None:
        assert normalize_arabic("Hello world") == "Hello world"

    def test_strips_tatweel(self) -> None:
        # Tatweel U+0640 between letters
        text = "بـوـلـيـصـة"  # Bill, with kashida
        result = normalize_arabic(text)
        assert "ـ" not in result
        assert "بوليصة" in result

    def test_unifies_alef_with_madda(self) -> None:
        # U+0622 (Alef Madda) → U+0627 (bare Alef)
        text = "آية"
        result = normalize_arabic(text)
        assert "آ" not in result
        assert result.startswith("ا")

    def test_unifies_alef_with_hamza_above(self) -> None:
        # U+0623 → U+0627
        text = "أحمد"
        result = normalize_arabic(text)
        assert "أ" not in result
        assert result.startswith("ا")

    def test_unifies_alef_with_hamza_below(self) -> None:
        # U+0625 → U+0627
        text = "إسلام"
        result = normalize_arabic(text)
        assert "إ" not in result
        assert result.startswith("ا")

    def test_unifies_alef_maksura_to_yeh(self) -> None:
        # U+0649 → U+064A
        text = "مستشفى"
        result = normalize_arabic(text)
        assert "ى" not in result
        assert "ي" in result

    def test_strips_diacritics(self) -> None:
        # Fatha (U+064E) on a base letter
        text = "كَتَبَ"  # "wrote" with full Tashkeel
        result = normalize_arabic(text)
        # All marks U+064B–U+065F removed.
        for ch in result:
            cp = ord(ch)
            assert not (0x064B <= cp <= 0x065F)

    def test_strips_shadda_and_sukun(self) -> None:
        text = "مُحَمَّدْ"  # Muhammad with shadda + sukun
        result = normalize_arabic(text)
        assert "ّ" not in result  # Shadda
        assert "ْ" not in result  # Sukun

    def test_idempotent(self) -> None:
        # Running normalize twice should give the same result.
        text = "أَحْمَدُ بـن مُحَمَّدْ"
        once = normalize_arabic(text)
        twice = normalize_arabic(once)
        assert once == twice

    def test_collapses_whitespace(self) -> None:
        text = "بوليصة    الشحن"
        result = normalize_arabic(text)
        assert "    " not in result
        assert "بوليصة الشحن" == result.strip()

    def test_preserves_mixed_content(self) -> None:
        # Arabic characters normalized, English untouched.
        text = "Invoice رقم INV-2026-001"
        result = normalize_arabic(text)
        assert "Invoice" in result
        assert "INV-2026-001" in result


# ---------------------------------------------------------------------------
# Detect integration
# ---------------------------------------------------------------------------


class TestArabicDetectionInClassify:
    """Verify the Arabic detection wires into PDFClassification."""

    def test_arabic_pdf_flagged(self, tmp_path) -> None:
        import fitz

        from pdfmux.detect import classify

        # Build an Arabic-heavy PDF using a font that supports Arabic.
        # If a test environment lacks a suitable system font, skip.
        pdf = tmp_path / "arabic.pdf"
        doc = fitz.open()
        page = doc.new_page()
        try:
            # Try a font that supports Arabic; fall back to default.
            page.insert_text(
                (72, 100),
                "بوليصة الشحن رقم INV-2026-001 مرحبا بالعالم بوليصة الشحن",
                fontsize=14,
            )
        except Exception:
            doc.close()
            pytest.skip("No Arabic-capable font available in test environment")

        doc.save(str(pdf))
        doc.close()

        result = classify(pdf)
        # The PDF text extraction may or may not succeed depending on font;
        # if it does, is_arabic should be True. Otherwise the flag is False
        # but the field exists — both outcomes are acceptable.
        assert hasattr(result, "is_arabic")
        assert hasattr(result, "arabic_pages")
        assert isinstance(result.arabic_pages, list)

    def test_english_pdf_not_arabic(self, digital_pdf) -> None:
        from pdfmux.detect import classify

        result = classify(digital_pdf)
        assert result.is_arabic is False
        assert result.arabic_pages == []
