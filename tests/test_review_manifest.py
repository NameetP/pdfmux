"""Tests for the review manifest and `pdfmux review` (scan / resolve / export)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from pdfmux import review_manifest as rm


def _flag(fid: str, page: int = 1, kind: str = "low_confidence", text: str = "x") -> dict:
    return {
        "id": fid,
        "page": page,
        "kind": kind,
        "text": text,
        "reason": f"reason {fid}",
        "score": 0.5,
        "line_index": 0,
        "bbox": [1.0, 2.0, 3.0, 4.0],
    }


class TestRecordRun:
    def test_first_run_opens_everything(self):
        m = rm.new_manifest()
        s = rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a"), _flag("b")], now="t1")
        assert [f["id"] for f in s.new] == ["a", "b"] and not s.still_open
        assert m["documents"]["h"]["flags"]["a"]["state"] == "open"
        assert m["documents"]["h"]["flags"]["a"]["first_seen"] == "t1"

    def test_second_run_skips_resolved_and_reports_new(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a"), _flag("b")], now="t1")
        rm.set_state(m, ["a"], "resolved", note="checked", now="t2")
        s = rm.record_run(
            m, sha256="h", document="a.pdf", flags=[_flag("a"), _flag("b"), _flag("c")], now="t3"
        )
        assert [f["id"] for f in s.new] == ["c"]
        assert [f["id"] for f in s.still_open] == ["b"]
        assert s.resolved_skipped == 1
        assert [f["id"] for f in s.to_review] == ["c", "b"]
        entry = m["documents"]["h"]["flags"]["a"]
        assert entry["state"] == "resolved" and entry["note"] == "checked"
        assert entry["last_seen"] == "t3" and entry["first_seen"] == "t1"

    def test_missing_flag_goes_gone_then_reopens(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a")])
        s = rm.record_run(m, sha256="h", document="a.pdf", flags=[])
        assert s.gone == ["a"] and m["documents"]["h"]["flags"]["a"]["state"] == "gone"
        s = rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a")])
        assert [f["id"] for f in s.reopened] == ["a"]

    def test_resolved_flag_that_disappears_stays_resolved(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a")])
        rm.set_state(m, ["a"], "resolved")
        rm.record_run(m, sha256="h", document="a.pdf", flags=[])
        assert m["documents"]["h"]["flags"]["a"]["state"] == "resolved"

    def test_run_cannot_overwrite_review_state(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a")])
        rm.set_state(m, ["a"], "resolved", note="ok")
        hostile = {**_flag("a"), "state": "open", "note": "x"}
        rm.record_run(m, sha256="h", document="a.pdf", flags=[hostile])
        entry = m["documents"]["h"]["flags"]["a"]
        assert entry["state"] == "resolved" and entry["note"] == "ok"

    def test_moved_file_keeps_its_review(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="old/a.pdf", flags=[_flag("a")])
        rm.set_state(m, ["a"], "resolved")
        s = rm.record_run(m, sha256="h", document="new/a.pdf", flags=[_flag("a")])
        assert s.resolved_skipped == 1 and m["documents"]["h"]["path"] == "new/a.pdf"

    def test_flags_without_ids_are_ignored(self):
        m = rm.new_manifest()
        s = rm.record_run(m, sha256="h", document="a.pdf", flags=[{"kind": "x"}])
        assert not s.new


class TestSetState:
    def test_prefix_and_unknown(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("abcdef123456")])
        changed, unknown = rm.set_state(m, ["abcdef", "zzz"], "resolved")
        assert changed == ["abcdef123456"] and unknown == ["zzz"]

    def test_short_prefix_is_not_enough(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("abcdef123456")])
        assert rm.set_state(m, ["abc"], "resolved") == ([], ["abc"])

    def test_reopen_clears_resolved_at(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a")])
        rm.set_state(m, ["a"], "resolved", now="t")
        rm.set_state(m, ["a"], "open")
        assert "resolved_at" not in m["documents"]["h"]["flags"]["a"]

    def test_bad_state(self):
        with pytest.raises(ValueError):
            rm.set_state(rm.new_manifest(), ["a"], "done")


class TestPersistence:
    def test_round_trip_and_atomic_save(self, tmp_path):
        path = tmp_path / "sub" / rm.MANIFEST_NAME
        m = rm.new_manifest()
        rm.record_run(m, sha256="h", document="a.pdf", flags=[_flag("a")])
        rm.save(path, m)
        assert rm.load(path) == m
        assert [p.name for p in path.parent.iterdir()] == [rm.MANIFEST_NAME]

    def test_missing_file_is_empty_manifest(self, tmp_path):
        assert rm.load(tmp_path / "nope.json") == rm.new_manifest()

    @pytest.mark.parametrize("content", ["{not json", '{"version": 99}', "[]"])
    def test_bad_file_raises(self, tmp_path, content):
        path = tmp_path / "m.json"
        path.write_text(content)
        with pytest.raises(rm.ManifestError):
            rm.load(path)

    def test_default_manifest_path(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")
        assert rm.default_manifest_path([tmp_path]) == tmp_path / rm.MANIFEST_NAME
        assert rm.default_manifest_path([pdf]) == tmp_path / rm.MANIFEST_NAME
        assert rm.default_manifest_path([tmp_path / "x.json"]) == tmp_path / "x.json"

    def test_sha256(self, tmp_path):
        p = tmp_path / "a.pdf"
        p.write_bytes(b"abc")
        assert rm.file_sha256(p).startswith("ba7816bf")


class TestExport:
    def _manifest(self):
        m = rm.new_manifest()
        rm.record_run(m, sha256="h1", document="one.pdf", flags=[_flag("a")])
        rm.record_run(
            m, sha256="h2", document="two.pdf", flags=[_flag("b", 2), _flag("c", 1, "table_cell")]
        )
        rm.set_state(m, ["c"], "resolved")
        return m

    def test_rows_worst_document_first_and_state_filter(self):
        m = self._manifest()
        rm.record_run(m, sha256="h2", document="two.pdf", flags=[_flag("b", 2), _flag("d", 3)])
        assert [r["id"] for r in rm.rows(m)] == ["b", "d", "a"]
        assert [r["id"] for r in rm.rows(m, ("resolved",))] == ["c"]
        assert len(rm.rows(m, rm.STATES)) == 4

    def test_csv_is_spreadsheet_ready(self):
        text = rm.to_csv(rm.rows(self._manifest(), rm.STATES))
        rows = list(csv.DictReader(io.StringIO(text)))
        assert list(rows[0]) == list(rm.EXPORT_COLUMNS)
        assert json.loads(rows[0]["bbox"]) == [1.0, 2.0, 3.0, 4.0]
        assert {r["document"] for r in rows} == {"one.pdf", "two.pdf"}

    def test_json_export(self):
        data = json.loads(rm.to_json(rm.rows(self._manifest())))
        assert {r["id"] for r in data} == {"a", "b"}


# ---------------------------------------------------------------------------
# CLI: the "done when" from the Instinct report, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Two PDFs and a fake process() whose flags the test controls."""
    from pdfmux import cli

    (tmp_path / "a.pdf").write_bytes(b"%PDF-a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-b")
    (tmp_path / "notes.txt").write_text("not a pdf")
    flags = {"a.pdf": [_flag("a1"), _flag("a2", 2)], "b.pdf": [_flag("b1")]}

    def fake_process(path, output_format="json", quality="standard"):
        name = Path(path).name
        if name == "broken.pdf":
            raise RuntimeError("cannot open")
        return SimpleNamespace(review_flags=[dict(f) for f in flags[name]])

    monkeypatch.setattr(cli, "process", fake_process)
    return tmp_path, flags


def _run(*args):
    from pdfmux.cli import app

    return CliRunner().invoke(app, list(args))


class TestReviewCli:
    def test_scan_resolve_rescan_export(self, folder):
        path, flags = folder
        first = _run("review", "scan", str(path))
        assert first.exit_code == 0, first.output
        assert "3 new, 0 reopened, 0 still open" in first.output
        manifest = path / rm.MANIFEST_NAME
        assert manifest.exists()

        resolved = _run("review", "resolve", "a1", "b1", "-m", str(path), "--note", "checked")
        assert resolved.exit_code == 0 and "Resolved 2 flag(s)" in resolved.output

        flags["a.pdf"].append(_flag("a3", 3))  # e.g. found by a newer pdfmux
        second = _run("review", "scan", str(path))
        assert second.exit_code == 0, second.output
        assert "1 new, 0 reopened, 1 still open" in second.output
        assert "a3" in second.output and "a2" in second.output
        assert "a1" not in second.output and "b1" not in second.output

        out = path / "flags.csv"
        exported = _run("review", "export", str(path), "-o", str(out))
        assert exported.exit_code == 0 and "Wrote 2 flag(s)" in exported.output
        ids = [r["id"] for r in csv.DictReader(out.open())]
        assert sorted(ids) == ["a2", "a3"]

        everything = _run("review", "export", str(path), "--state", "all", "--format", "json")
        assert {r["id"] for r in json.loads(everything.output)} == {"a1", "a2", "a3", "b1"}

    def test_scan_survives_a_broken_file(self, folder):
        path, _ = folder
        (path / "broken.pdf").write_bytes(b"%PDF-x")
        result = _run("review", "scan", str(path))
        assert result.exit_code == 1
        assert "broken.pdf" in result.output and "3 new" in result.output

    def test_no_recursive(self, folder):
        path, flags = folder
        sub = path / "sub"
        sub.mkdir()
        (sub / "a.pdf").write_bytes(b"%PDF-sub")
        result = _run("review", "scan", str(path), "--no-recursive")
        assert "3 new" in result.output

    def test_empty_folder(self, tmp_path):
        result = _run("review", "scan", str(tmp_path))
        assert result.exit_code == 1 and "No PDFs found" in result.output

    def test_resolve_unknown_id(self, folder):
        path, _ = folder
        _run("review", "scan", str(path))
        result = _run("review", "resolve", "nope", "-m", str(path))
        assert result.exit_code == 1 and "Unknown flag id: nope" in result.output

    def test_resolve_and_export_without_manifest(self, tmp_path):
        assert _run("review", "resolve", "a", "-m", str(tmp_path)).exit_code == 2
        result = _run("review", "export", str(tmp_path))
        assert result.exit_code == 2 and "review scan" in result.output

    def test_export_rejects_bad_options(self, folder):
        path, _ = folder
        _run("review", "scan", str(path))
        assert _run("review", "export", str(path), "--format", "xlsx").exit_code == 2
        assert _run("review", "export", str(path), "--state", "done").exit_code == 2

    def test_corrupt_manifest_is_reported_not_overwritten(self, folder):
        path, _ = folder
        manifest = path / rm.MANIFEST_NAME
        manifest.write_text("{broken")
        result = _run("review", "scan", str(path))
        assert result.exit_code == 2 and manifest.read_text() == "{broken"
