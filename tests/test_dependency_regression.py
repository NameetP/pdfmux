"""Guards against dependency upgrades that silently gut extraction.

pdfmux's whole claim is that it tells you when an extraction lost content. The
one failure mode that claim cannot survive is pdfmux's *own* extraction losing
content without anyone noticing — which is exactly what happened on 2026-07-20:
pymupdf4llm renumbered 0.3.4 -> 1.28.0, its ``to_markdown()`` started returning
headings only, and ``audit_document`` began returning 0 characters for plain
digital PDFs. Nothing failed loudly; one unrelated batch-aggregate assertion in
test_verifier.py went red and the real cause sat unread for four days.

These tests are deliberately crude and dependency-facing: if a future resolve
changes what the PDF stack gives back, they fail here with a message that names
the dependency, instead of surfacing as a confusing verdict assertion elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdfmux.audit import audit_document

# Plain, unambiguous body prose — no tables, no columns, no OCR. Any extractor
# worth shipping returns essentially all of this.
_BODY = (
    "The consolidated revenue for the fiscal quarter reached forty two million "
    "dirhams, an increase of eleven percent against the prior period. Operating "
    "margin held steady at nineteen percent despite higher input costs."
)

# audit_document normalizes whitespace and adds markdown syntax, so it will
# never match raw character count exactly. It must not lose the body, though —
# anything below this ratio means prose is being dropped, not reformatted.
MIN_RECOVERY_RATIO = 0.7


@pytest.fixture
def digital_pdf(tmp_path: Path) -> Path:
    """A 2-page digital PDF with a heading and body prose on each page."""
    pdf_path = tmp_path / "digital.pdf"
    doc = fitz.open()
    for heading in ("# Quarterly Report", "## Outlook"):
        page = doc.new_page()
        page.insert_text((72, 72), f"{heading}\n{_BODY}", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_audit_recovers_body_prose_not_just_headings(digital_pdf: Path) -> None:
    """audit_document must return the body text, not only the headings.

    Fails on pymupdf4llm >= 1.0 (see the upper bound in pyproject.toml).
    """
    raw_chars = sum(len(page.get_text()) for page in fitz.open(str(digital_pdf)))
    assert raw_chars > 0, "fixture produced no text layer — fixture bug, not a dep bug"

    audit = audit_document(digital_pdf)
    audit_chars = sum(len(page.text or "") for page in audit.pages)
    ratio = audit_chars / raw_chars

    assert ratio >= MIN_RECOVERY_RATIO, (
        f"audit_document recovered only {audit_chars}/{raw_chars} chars "
        f"({ratio:.0%}) from a plain digital PDF. The PDF text stack is dropping "
        f"body prose — check the installed pymupdf4llm / pymupdf versions against "
        f"the upper bounds in pyproject.toml before touching pdfmux's own code."
    )


def test_audit_returns_body_text_verbatim(digital_pdf: Path) -> None:
    """A distinctive body phrase must survive extraction on every page."""
    audit = audit_document(digital_pdf)
    for page in audit.pages:
        assert "consolidated revenue" in (page.text or ""), (
            f"page {page.page_num} lost its body prose entirely — heading-only "
            f"extraction. Got: {(page.text or '')[:120]!r}"
        )
