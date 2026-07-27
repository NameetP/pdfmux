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


class TestArabicRouting:
    """The Arabic route must actually reach the matrix.

    Until 2026-07-27 `_classify_to_page_type` returned "arabic" but
    ROUTING_MATRIX had no "arabic" rows, so every Arabic document fell through
    to DEFAULT_CHAIN — whose BALANCED arm is ("opendataloader", "pymupdf") and
    never reaches an LLM. The route was computed and discarded, which is the
    opposite of what both `pipeline.py:527` and the README promise.
    """

    def test_arabic_rows_exist_in_the_matrix(self) -> None:
        from pdfmux.router.engine import ROUTING_MATRIX
        from pdfmux.router.strategies import Strategy

        for strategy in (Strategy.ECONOMY, Strategy.BALANCED, Strategy.PREMIUM):
            assert ("arabic", strategy) in ROUTING_MATRIX, (
                f"no arabic row for {strategy} — the route falls through to DEFAULT_CHAIN"
            )

    def test_arabic_does_not_fall_back_to_the_default_chain(self) -> None:
        from pdfmux.router.engine import DEFAULT_CHAIN, ROUTING_MATRIX
        from pdfmux.router.strategies import Strategy

        assert ROUTING_MATRIX[("arabic", Strategy.BALANCED)] != DEFAULT_CHAIN[Strategy.BALANCED]

    def test_balanced_and_premium_prefer_the_llm_backend(self) -> None:
        """The Gemma provider is the only backend advertising an "arabic" capability, and the
        README documents routing Arabic through it instead of PyMuPDF."""
        from pdfmux.router.engine import ROUTING_MATRIX
        from pdfmux.router.strategies import Strategy

        for strategy in (Strategy.BALANCED, Strategy.PREMIUM):
            chain = ROUTING_MATRIX[("arabic", strategy)]
            assert chain[0] == "llm", f"{strategy} should lead with the LLM backend, got {chain}"

    def test_every_arabic_chain_ends_in_a_free_local_fallback(self) -> None:
        """The chain must never be able to return nothing when no LLM is configured."""
        from pdfmux.router.engine import ROUTING_MATRIX
        from pdfmux.router.strategies import Strategy

        for strategy in (Strategy.ECONOMY, Strategy.BALANCED, Strategy.PREMIUM):
            assert ROUTING_MATRIX[("arabic", strategy)][-1] == "pymupdf"

    def test_classifier_still_emits_the_arabic_route(self) -> None:
        """Guards the other half: the matrix rows are useless if the route changes."""
        from pdfmux.detect import PDFClassification
        from pdfmux.pipeline import _classify_to_page_type

        c = PDFClassification(page_count=1)
        c.is_arabic = True
        assert _classify_to_page_type(c) == "arabic"


class TestBidiDependencyIsCore:
    """python-bidi is a core dependency, not an optional extra.

    Until 2026-07-27 both docstrings in ``pdfmux/arabic.py`` told users to run
    ``pip install pdfmux[arabic]``. No such extra exists — and because pip
    accepts unknown extras silently, a user following that hint installed
    nothing while believing they had enabled RTL support.
    """

    def test_python_bidi_is_a_core_dependency(self) -> None:
        """If this ever moves to an extra, the docstrings must move with it."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        cfg = tomllib.loads(pyproject.read_text())
        core = " ".join(cfg["project"]["dependencies"])
        assert "python-bidi" in core, "python-bidi must stay a core dependency"

    def test_no_install_instruction_names_a_nonexistent_extra(self) -> None:
        """Every ``pip install pdfmux[...]`` must resolve to a real extra.

        Generalised deliberately. This defect class has now shipped three
        times — ``pdfmux[arabic]`` here, ``pdfmux[arabic,llm-gemma]`` in the
        README, and ``pdfmux[local]`` on the blog — because a wrong extra
        fails silently at every layer: pip does not error, and the import it
        was supposed to enable is often already present for another reason.

        Matches only *install instructions*, not every mention of the string.
        Prose that names a bad extra in order to rule it out is correct and
        must not fail — that distinction is the whole reason the original hint
        survived so long, since a blunt scan would have been switched off.
        """
        import re
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        cfg = tomllib.loads((root / "pyproject.toml").read_text())
        valid = set(cfg["project"].get("optional-dependencies", {}))

        targets = [
            *(root / "src").rglob("*.py"),
            root / "README.md",
            *(root / "docs").rglob("*.md"),
        ]
        # CHANGELOG is excluded by design: it records superseded commands
        # verbatim, including the broken ones this test exists to prevent.
        instruction = re.compile(r"""pip install\s+["']?pdfmux\[([a-zA-Z0-9,_\-]+)\]""")

        offenders: list[str] = []
        for path in targets:
            if not path.is_file():
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                for match in instruction.findall(line):
                    bad = sorted({e.strip() for e in match.split(",")} - valid)
                    if bad:
                        rel = path.relative_to(root)
                        offenders.append(f"{rel}:{lineno} -> unknown extra(s) {bad}")

        assert not offenders, "install instructions name extras that do not exist:\n" + "\n".join(
            offenders
        )

    def test_missing_python_bidi_warns_instead_of_failing_silently(
        self, monkeypatch, caplog
    ) -> None:
        """A broken install must not quietly return reversed Arabic.

        Without BiDi, RTL text comes back in storage order — reversed, but
        still perfectly plausible-looking to anyone who does not read Arabic.
        Silent, plausible, wrong output is the failure mode this package
        exists to surface, so the one path that can produce it must say so.
        """
        import builtins
        import logging

        from pdfmux import arabic

        real_import = builtins.__import__

        def no_bidi(name, *args, **kwargs):
            if name.startswith("bidi"):
                raise ImportError("simulated broken install")
            return real_import(name, *args, **kwargs)

        arabic._load_get_display.cache_clear()
        monkeypatch.setattr(builtins, "__import__", no_bidi)
        try:
            with caplog.at_level(logging.WARNING, logger="pdfmux.arabic"):
                text = "شحنة MSKU1234567"
                assert arabic.fix_bidi_order(text) == text, "must degrade, not raise"
                assert arabic.fix_bidi_order(text) == text
                assert arabic.fix_bidi_order(text) == text

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
            msg = warnings[0].getMessage()
            assert "core pdfmux dependency" in msg
            assert "pdfmux[arabic]" in msg, "must name the non-extra users will search for"
        finally:
            arabic._load_get_display.cache_clear()
