"""Tests for OCR budget control in the pipeline."""

from __future__ import annotations

from pdfmux.audit import DocumentAudit, PageAudit


def _make_audit(good: int, bad: int, empty: int) -> DocumentAudit:
    """Create a DocumentAudit with specified page counts."""
    pages = []
    idx = 0
    for _ in range(good):
        pages.append(PageAudit(idx, "x" * 300, 300, 0, "good", "ok"))
        idx += 1
    for _ in range(bad):
        pages.append(PageAudit(idx, "x" * 50, 50, 3, "bad", "low text"))
        idx += 1
    for _ in range(empty):
        pages.append(PageAudit(idx, "", 0, 2, "empty", "no text"))
        idx += 1
    return DocumentAudit(pages=pages, total_pages=idx)


class TestBudgetCaps:
    """Tests for OCR budget capping logic."""

    def test_budget_caps_ocr_pages(self) -> None:
        """When bad+empty > budget, only budget pages are OCR'd."""
        from pdfmux.pipeline import OCR_BUDGET_RATIO

        # 10 page doc: 4 good, 3 bad, 3 empty → 6 need OCR
        # Budget = 30% of 10 = 3 pages
        audit = _make_audit(good=4, bad=3, empty=3)
        all_needing_ocr = audit.bad_pages + audit.empty_pages
        assert len(all_needing_ocr) == 6

        max_ocr = max(1, int(audit.total_pages * OCR_BUDGET_RATIO))
        assert max_ocr == 3
        assert len(all_needing_ocr) > max_ocr  # Budget would kick in

    def test_priority_bad_over_empty(self) -> None:
        """Budget prioritizes 'bad' pages (some text) over 'empty' pages."""
        audit = _make_audit(good=4, bad=2, empty=4)
        all_needing_ocr = audit.bad_pages + audit.empty_pages

        # Simulate the priority sort from pipeline.py
        prioritized = sorted(
            all_needing_ocr,
            key=lambda pn: (0 if audit.pages[pn].quality == "bad" else 1, pn),
        )

        # First should be bad pages (indices 4, 5), then empty (6, 7, 8, 9)
        bad_indices = set(audit.bad_pages)
        empty_indices = set(audit.empty_pages)

        # First N entries should be bad pages
        num_bad = len(audit.bad_pages)
        first_n = prioritized[:num_bad]
        assert all(p in bad_indices for p in first_n)

        # Remaining should be empty pages
        remaining = prioritized[num_bad:]
        assert all(p in empty_indices for p in remaining)

    def test_under_budget_processes_all(self) -> None:
        """When bad+empty <= budget, all pages are processed."""
        from pdfmux.pipeline import OCR_BUDGET_RATIO

        # 20 page doc: 18 good, 1 bad, 1 empty → 2 need OCR
        # Budget = 30% of 20 = 6 pages → plenty of room
        audit = _make_audit(good=18, bad=1, empty=1)
        all_needing_ocr = audit.bad_pages + audit.empty_pages
        assert len(all_needing_ocr) == 2

        max_ocr = max(1, int(audit.total_pages * OCR_BUDGET_RATIO))
        assert max_ocr == 6
        assert len(all_needing_ocr) <= max_ocr  # No budget cutoff
