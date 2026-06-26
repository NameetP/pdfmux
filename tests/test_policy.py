"""Tests for the versioned policy object + runtime calibration (Build #4)."""

from __future__ import annotations

import json
from pathlib import Path

import pdfmux
from pdfmux.audit import score_page
from pdfmux.policy import (
    BASE_POLICY_ID,
    DEFAULT_POLICY,
    Calibration,
    Policy,
    load_policy,
    load_policy_file,
    policy_from_dict,
    policy_to_dict,
)


class TestPolicyDefaults:
    """The default policy must reproduce the historical hardcoded constants."""

    def test_default_id(self) -> None:
        assert DEFAULT_POLICY.policy_id == BASE_POLICY_ID == "pdfmux-policy-v1.7"

    def test_audit_thresholds_unchanged(self) -> None:
        assert DEFAULT_POLICY.empty_text_threshold == 20
        assert DEFAULT_POLICY.minimal_text_threshold == 50
        assert DEFAULT_POLICY.good_text_threshold == 200

    def test_repair_guard_defaults_unchanged(self) -> None:
        assert DEFAULT_POLICY.native_trust_threshold == 0.80
        assert DEFAULT_POLICY.repair_margin == 0.0

    def test_budget_defaults_unchanged(self) -> None:
        assert DEFAULT_POLICY.ocr_budget_ratio == 0.30
        assert DEFAULT_POLICY.image_heavy_threshold == 0.50
        assert DEFAULT_POLICY.budget_lower_bound == 0.25
        assert DEFAULT_POLICY.budget_offset == 0.10

    def test_score_page_default_matches_explicit_default(self) -> None:
        """Passing DEFAULT_POLICY must equal the implicit default — no drift."""
        text = "Some readable prose that is comfortably above the empty threshold here."
        assert score_page(text) == score_page(text, policy=DEFAULT_POLICY)


class TestPolicyEnvOverrides:
    def test_env_override_changes_value_and_id(self, monkeypatch) -> None:
        monkeypatch.setenv("PDFMUX_NATIVE_TRUST", "0.95")
        p = load_policy()
        assert p.native_trust_threshold == 0.95
        assert p.policy_id.startswith(BASE_POLICY_ID + "+")  # honest suffix

    def test_no_override_keeps_base_id(self, monkeypatch) -> None:
        monkeypatch.delenv("PDFMUX_NATIVE_TRUST", raising=False)
        monkeypatch.delenv("PDFMUX_REPAIR_MARGIN", raising=False)
        monkeypatch.delenv("PDFMUX_OCR_BUDGET", raising=False)
        monkeypatch.delenv("PDFMUX_STRICT_GATE", raising=False)
        assert load_policy().policy_id == BASE_POLICY_ID

    def test_invalid_override_ignored(self, monkeypatch) -> None:
        monkeypatch.setenv("PDFMUX_REPAIR_MARGIN", "not-a-number")
        assert load_policy().policy_id == BASE_POLICY_ID


class TestPolicySerialization:
    def test_roundtrip(self) -> None:
        p = Policy(calibration=Calibration(method="platt", platt_a=2.0, platt_b=-1.0))
        restored = policy_from_dict(policy_to_dict(p))
        assert restored.policy_id == p.policy_id
        assert restored.native_trust_threshold == p.native_trust_threshold
        assert restored.calibration is not None
        assert restored.calibration.method == "platt"
        assert restored.calibration.platt_a == 2.0

    def test_isotonic_knots_survive_json(self) -> None:
        cal = Calibration(method="isotonic", knots_x=(0.0, 0.5, 1.0), knots_y=(0.1, 0.6, 0.95))
        p = Policy(calibration=cal)
        # JSON makes tuples into lists; policy_from_dict must restore tuples.
        restored = policy_from_dict(json.loads(json.dumps(policy_to_dict(p))))
        assert restored.calibration is not None
        assert restored.calibration.knots_x == (0.0, 0.5, 1.0)

    def test_load_policy_file_missing_returns_env_base(self, tmp_path) -> None:
        missing = tmp_path / "nope.json"
        assert load_policy_file(missing).policy_id == BASE_POLICY_ID

    def test_load_policy_file_broken_never_raises(self, tmp_path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not valid json", encoding="utf-8")
        # Must fall back to the base policy, never raise.
        assert load_policy_file(broken).policy_id == BASE_POLICY_ID

    def test_load_policy_file_applies_fitted(self, tmp_path) -> None:
        p = Policy(
            policy_id="pdfmux-policy-fitted-test",
            calibration=Calibration(method="platt", platt_a=3.0, platt_b=-1.5, n_samples=50),
        )
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(policy_to_dict(p)), encoding="utf-8")
        loaded = load_policy_file(path)
        assert loaded.policy_id == "pdfmux-policy-fitted-test"
        assert loaded.calibration is not None
        assert loaded.calibration.method == "platt"


class TestCalibration:
    def test_identity_is_noop(self) -> None:
        cal = Calibration()  # identity
        for raw in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert cal.apply(raw) == raw

    def test_platt_is_monotonic(self) -> None:
        cal = Calibration(method="platt", platt_a=4.0, platt_b=-2.0)
        vals = [cal.apply(x / 10) for x in range(11)]
        assert vals == sorted(vals)
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_isotonic_interpolates_and_is_monotonic(self) -> None:
        cal = Calibration(method="isotonic", knots_x=(0.0, 0.5, 1.0), knots_y=(0.0, 0.4, 1.0))
        assert cal.apply(0.0) == 0.0
        assert cal.apply(1.0) == 1.0
        assert abs(cal.apply(0.25) - 0.2) < 1e-9  # halfway up the first segment
        vals = [cal.apply(x / 20) for x in range(21)]
        assert vals == sorted(vals)

    def test_isotonic_clamps_outside_knots(self) -> None:
        cal = Calibration(method="isotonic", knots_x=(0.2, 0.8), knots_y=(0.3, 0.9))
        assert cal.apply(0.0) == 0.3
        assert cal.apply(1.0) == 0.9


class TestPolicyIdInOutput:
    def test_json_emits_policy_id(self, digital_pdf: Path) -> None:
        data = pdfmux.extract_json(digital_pdf)
        assert data["schema_version"] == "1.4.0"
        assert data["policy_id"] == BASE_POLICY_ID
