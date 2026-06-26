"""Tests for the runtime calibration loop (Build #4 Part B).

Covers the fitting math (isotonic PAVA, Platt, ECE) and the closed loop:
``pdfmux calibrate`` → write policy → runtime reload → calibrated confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
from typer.testing import CliRunner

import pdfmux
from pdfmux.calibration import (
    _pava,
    expected_calibration_error,
    fit_calibration,
    fit_isotonic,
    fit_platt,
)
from pdfmux.cli import app
from pdfmux.policy import load_policy_file, policy_from_dict

runner = CliRunner()


class TestPAVA:
    def test_already_monotonic_unchanged(self) -> None:
        assert _pava([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]

    def test_pools_violators(self) -> None:
        out = _pava([1.0, 0.0])  # violators pool to their mean
        assert out == [0.5, 0.5]

    def test_output_is_monotonic(self) -> None:
        out = _pava([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        assert out == sorted(out)


class TestFitIsotonic:
    def test_monotonic_knots(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
        labels = [0, 0, 0, 1, 1, 1]
        cal = fit_isotonic(scores, labels)
        assert cal.method == "isotonic"
        assert list(cal.knots_y) == sorted(cal.knots_y)  # non-decreasing
        assert cal.apply(0.1) <= cal.apply(0.9)

    def test_separable_maps_extremes(self) -> None:
        scores = [0.05, 0.1, 0.15, 0.85, 0.9, 0.95]
        labels = [0, 0, 0, 1, 1, 1]
        cal = fit_isotonic(scores, labels)
        assert cal.apply(0.05) < 0.5 < cal.apply(0.95)


class TestFitPlatt:
    def test_separable_data_learns_direction(self) -> None:
        scores = [0.1] * 10 + [0.9] * 10
        labels = [0] * 10 + [1] * 10
        cal = fit_platt(scores, labels)
        assert cal.method == "platt"
        assert cal.apply(0.9) > cal.apply(0.1)
        assert cal.apply(0.9) > 0.5 > cal.apply(0.1)


class TestECE:
    def test_perfect_calibration_low_ece(self) -> None:
        # Predictions equal observed frequencies → near-zero ECE.
        probs = [0.0] * 50 + [1.0] * 50
        labels = [0] * 50 + [1] * 50
        assert expected_calibration_error(probs, labels) < 0.01

    def test_miscalibration_detected(self) -> None:
        # Confident-but-wrong predictions → high ECE.
        probs = [0.9] * 50
        labels = [0] * 50
        assert expected_calibration_error(probs, labels) > 0.5


class TestFitCalibrationImprovesECE:
    def test_isotonic_reduces_ece_on_miscalibrated_scores(self) -> None:
        # Raw scores are systematically over-confident: a score of 0.8 is only
        # 'good' half the time. Calibration should pull that down and cut ECE.
        scores, labels = [], []
        for _ in range(40):
            scores.append(0.8)
            labels.append(1)
        for _ in range(40):
            scores.append(0.8)
            labels.append(0)
        for _ in range(20):
            scores.append(0.2)
            labels.append(0)
        cal = fit_calibration(scores, labels, method="isotonic")
        assert cal.ece_after <= cal.ece_before
        assert cal.n_samples == 100
        # 0.8 raw should calibrate toward ~0.5 (its true good-rate).
        assert 0.3 < cal.apply(0.8) < 0.7


# ---------------------------------------------------------------------------
# End-to-end: pdfmux calibrate writes a policy; the runtime reloads + applies it.
# ---------------------------------------------------------------------------


def _good_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This is a clean, readable page of digital text with plenty of words. "
        "It comfortably exceeds the good-text threshold and audits as good.",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()


def _empty_pdf(path: Path) -> None:
    doc = fitz.open()
    doc.new_page()  # blank → audits empty/bad
    doc.save(str(path))
    doc.close()


def _build_labelled_dir(d: Path) -> None:
    rows = ["filename,label"]
    for i in range(4):
        _good_pdf(d / f"good_{i}.pdf")
        rows.append(f"good_{i}.pdf,good")
    for i in range(4):
        _empty_pdf(d / f"bad_{i}.pdf")
        rows.append(f"bad_{i}.pdf,bad")
    (d / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestCalibrateClosedLoop:
    def test_calibrate_writes_policy_and_runtime_applies_it(self, tmp_path: Path) -> None:
        labelled = tmp_path / "labelled"
        labelled.mkdir()
        _build_labelled_dir(labelled)
        policy_path = tmp_path / "policy.json"

        result = runner.invoke(
            app,
            ["calibrate", str(labelled), "--method", "isotonic", "--out", str(policy_path)],
        )
        assert result.exit_code == 0, result.output
        assert policy_path.exists()

        # The written policy carries a calibration and a calibration-marked id.
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        policy = policy_from_dict(data)
        assert policy.calibration is not None
        assert policy.calibration.method == "isotonic"
        assert "-cal-" in policy.policy_id

        # Runtime reload: load_policy_file applies the fitted policy.
        loaded = load_policy_file(policy_path)
        assert loaded.calibration is not None
        assert "-cal-" in loaded.policy_id

    def test_calibrate_requires_labels_csv(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["calibrate", str(empty)])
        assert result.exit_code == 2

    def test_runtime_emits_calibrated_policy_id(self, tmp_path: Path, monkeypatch) -> None:
        labelled = tmp_path / "labelled"
        labelled.mkdir()
        _build_labelled_dir(labelled)
        policy_path = tmp_path / "policy.json"
        assert (
            runner.invoke(app, ["calibrate", str(labelled), "--out", str(policy_path)]).exit_code
            == 0
        )

        # Point the runtime at the fitted policy; a fresh extraction should
        # report the calibration-marked policy_id.
        monkeypatch.setenv("PDFMUX_POLICY_FILE", str(policy_path))
        good = tmp_path / "probe.pdf"
        _good_pdf(good)
        out = pdfmux.extract_json(good, quality="standard")
        assert "-cal-" in out["policy_id"]
