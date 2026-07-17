"""Tests for validate_verifier's defect seeding — must be deterministic.

Two runs with the same doc id must produce byte-identical defects, and the seed must be a
pure function of the doc id (no unseeded randomness anywhere in the seeding path).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pdfmux-bench/

import validate_verifier as vv  # noqa: E402

SINGLE = {0: "# Title\n\nAlpha beta gamma delta epsilon.\n\nSecond paragraph with more words here."}
MULTI = {
    0: "# Page one\n\n" + " ".join(f"word{i}" for i in range(40)),
    1: "# Page two\n\n" + " ".join(f"term{i}" for i in range(40)),
    2: "# Page three\n\n" + " ".join(f"item{i}" for i in range(40)),
}
TABLE = {
    0: "Intro text before the table.\n\n"
    "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n"
    "| 4 | 5 | 6 |\n| 7 | 8 | 9 |\n| 10 | 11 | 12 |\n\n"
    "Trailing text after the table."
}


def test_seed_is_pure_function_of_id():
    assert vv.seed_for("abc") == vv.seed_for("abc")
    assert vv.seed_for("abc") != vv.seed_for("abd")
    assert isinstance(vv.seed_for("x"), int)


def test_each_defect_is_deterministic():
    for doc_id in ("doc-alpha", "irs-fw9", "arxiv-1706.03762"):
        seed = vv.seed_for(doc_id)
        for pages in (SINGLE, MULTI, TABLE):
            for name, (_expected, fn) in vv.DEFECTS.items():
                r1 = fn(dict(pages), seed)
                r2 = fn(dict(pages), seed)
                assert r1 == r2, f"{name} non-deterministic for {doc_id}"


def test_defects_do_not_mutate_input():
    seed = vv.seed_for("doc")
    for pages in (SINGLE, MULTI, TABLE):
        original = {k: v for k, v in pages.items()}
        for _name, (_expected, fn) in vv.DEFECTS.items():
            fn(pages, seed)
        assert pages == original


def test_drop_page_blanks_exactly_one_page():
    seed = vv.seed_for("doc")
    new, meta = vv.defect_drop_page(MULTI, seed)
    blanked = [k for k in new if new[k] == "" and MULTI[k] != ""]
    assert len(blanked) == 1
    assert meta["dropped_page"] == blanked[0] + 1
    # every other page is untouched
    for k in MULTI:
        if k != blanked[0]:
            assert new[k] == MULTI[k]


def test_truncate_table_removes_rows_or_is_na():
    seed = vv.seed_for("doc")
    out = vv.defect_truncate_table(TABLE, seed)
    assert out is not None
    new, meta = out
    assert meta["rows_removed"] >= 1
    assert new[0].count("|") < TABLE[0].count("|")
    # a doc with no table -> not applicable
    assert vv.defect_truncate_table(SINGLE, seed) is None


def test_inject_offsource_grows_page_with_novel_tokens():
    seed = vv.seed_for("doc")
    new, meta = vv.defect_inject_offsource(SINGLE, seed)
    assert len(new[0]) > len(SINGLE[0])
    assert meta["sentences_injected"] >= 2
    # injected tokens are absent from the original source
    assert "plingbort" in new[0] or "blivetscarn" in new[0] or "vandertweed" in new[0]


def test_severe_loss_keeps_small_prefix():
    seed = vv.seed_for("doc")
    new, meta = vv.defect_severe_loss(MULTI, seed)
    assert meta["fraction_kept"] == 0.1
    for k in MULTI:
        assert len(new[k]) <= max(1, len(MULTI[k]) // 10)
        assert MULTI[k].startswith(new[k])


def test_parse_gt_pages_markers_and_plain():
    plain = vv.parse_gt_pages("just one page of text")
    assert plain == {0: "just one page of text"}
    marked = vv.parse_gt_pages("<!-- page: 1 -->\nfirst\n<!-- page: 2 -->\nsecond")
    assert marked == {0: "first", 1: "second"}


def test_parse_page_spec():
    assert vv.parse_page_spec("1") == [0]
    assert vv.parse_page_spec("13") == [12]
    assert vv.parse_page_spec("1-2") == [0, 1]
    assert vv.parse_page_spec("1-3") == [0, 1, 2]
