"""Versioned extraction policy — the single source of truth for every tunable.

Every threshold, penalty, and budget parameter that shapes extraction lives in
one frozen :class:`Policy` object carrying a ``policy_id``. The id is emitted in
the JSON output so a result is reproducible: *same policy_id + same input →
same decisions*. ``PDFMUX_*`` environment overrides are folded in at load time
(:func:`load_policy`); when any override changes a value the ``policy_id`` is
suffixed with a short content hash so a tuned run never masquerades as the
canonical policy.

The policy also carries an optional :class:`Calibration` — a fitted monotonic
map from the raw 5-check audit score to a calibrated probability, produced by
``pdfmux calibrate`` and loaded at runtime. The closed loop is
``calibrate → write policy → reload``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

BASE_POLICY_ID = "pdfmux-policy-v1.7"


@dataclass(frozen=True)
class Calibration:
    """A fitted monotonic map: raw audit score → calibrated probability.

    ``method`` is ``"identity"`` (no-op, the default), ``"isotonic"`` (a
    piecewise-linear map between sorted knots fitted by Pool-Adjacent-Violators),
    or ``"platt"`` (a sigmoid ``1/(1+e^-(a·x+b))``). The map MUST be monotonic
    non-decreasing so a higher raw score never yields a lower probability.
    """

    method: str = "identity"
    knots_x: tuple[float, ...] = ()  # isotonic: sorted raw-score knots
    knots_y: tuple[float, ...] = ()  # isotonic: calibrated value at each knot
    platt_a: float = 1.0
    platt_b: float = 0.0
    # Provenance + quality of the fit (for the reliability report).
    fitted_on: str = ""
    n_samples: int = 0
    ece_before: float = 0.0
    ece_after: float = 0.0

    def apply(self, raw: float) -> float:
        """Map a raw audit score (0–1) to a calibrated probability (0–1)."""
        raw = max(0.0, min(1.0, raw))
        if self.method == "platt":
            return 1.0 / (1.0 + math.exp(-(self.platt_a * raw + self.platt_b)))
        if self.method == "isotonic" and self.knots_x:
            xs, ys = self.knots_x, self.knots_y
            if raw <= xs[0]:
                return ys[0]
            if raw >= xs[-1]:
                return ys[-1]
            for i in range(1, len(xs)):
                if raw <= xs[i]:
                    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                    if x1 == x0:
                        return y1
                    t = (raw - x0) / (x1 - x0)
                    return y0 + t * (y1 - y0)
            return ys[-1]
        return raw  # identity / unknown


@dataclass(frozen=True)
class Policy:
    """The complete, versioned set of extraction tunables (policy-as-data)."""

    policy_id: str = BASE_POLICY_ID

    # --- Audit: text-length thresholds (chars) ---
    empty_text_threshold: int = 20
    minimal_text_threshold: int = 50
    good_text_threshold: int = 200
    page_window: int = 50

    # --- Audit: score_page penalties + bands ---
    density_penalty_minimal: float = 0.3  # char_count in [empty, minimal)
    density_penalty_good_noimg: float = 0.1  # char_count in [minimal, good), no images
    density_penalty_good_img: float = 0.2  # char_count in [minimal, good), has images
    alpha_ratio_low: float = 0.3
    alpha_ratio_mid: float = 0.5
    alpha_penalty_low: float = 0.25
    alpha_penalty_mid: float = 0.1
    word_len_min: float = 2.0
    word_len_max: float = 25.0
    word_len_penalty: float = 0.15
    whitespace_run_count: int = 10  # > this many wide runs → penalty
    whitespace_penalty: float = 0.1
    mojibake_high: int = 5  # > this → high penalty
    mojibake_penalty_high: float = 0.2
    mojibake_penalty_low: float = 0.05
    structure_bonus: float = 0.03  # any markdown heading present

    # --- OCR budget ---
    ocr_budget_ratio: float = 0.30  # default cap, fraction of pages
    image_heavy_threshold: float = 0.50  # ≥ this graphical-ratio → OCR all
    budget_lower_bound: float = 0.25  # > this graphical-ratio → generous budget
    budget_offset: float = 0.10  # added to graphical_ratio in the generous band

    # --- Script detection ---
    arabic_ratio_threshold: float = 0.05
    arabic_sample_limit: int = 20

    # --- Table detection ---
    table_score_threshold: int = 2

    # --- Quality gate ---
    strict_gate: float = 0.75

    # --- Monotonic repair guard (§5.8) ---
    native_trust_threshold: float = 0.80
    repair_margin: float = 0.0
    repair_alpha_collapse_drop: float = 0.25  # hard-fail: alpha-ratio drop
    repair_shrink_fraction: float = 0.50  # hard-fail: full replacement keeping < this

    # --- Runtime calibration (Part B) ---
    calibration: Calibration | None = None


DEFAULT_POLICY = Policy()

# Environment overrides → (Policy field). Reading at load time keeps the legacy
# PDFMUX_* knobs working while centralising them in the policy object.
_ENV_FLOAT_OVERRIDES: dict[str, str] = {
    "PDFMUX_OCR_BUDGET": "ocr_budget_ratio",
    "PDFMUX_NATIVE_TRUST": "native_trust_threshold",
    "PDFMUX_REPAIR_MARGIN": "repair_margin",
    "PDFMUX_STRICT_GATE": "strict_gate",
}


def _short_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def load_policy(base: Policy = DEFAULT_POLICY) -> Policy:
    """Return the active policy: ``base`` with ``PDFMUX_*`` overrides applied.

    If any override changes a value, the ``policy_id`` is suffixed with
    ``+<hash>`` of the overridden fields so the emitted id is honest.
    """
    overrides: dict[str, float] = {}
    for env_name, field_name in _ENV_FLOAT_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != getattr(base, field_name):
            overrides[field_name] = value

    if not overrides:
        return base

    suffix = _short_hash(overrides)
    return replace(base, policy_id=f"{base.policy_id}+{suffix}", **overrides)


def policy_to_dict(policy: Policy) -> dict:
    """Serialize a policy (incl. calibration) to a plain dict for JSON."""
    return asdict(policy)


def policy_from_dict(data: dict) -> Policy:
    """Rebuild a Policy from a dict produced by :func:`policy_to_dict`."""
    data = dict(data)
    cal = data.pop("calibration", None)
    fields = {f for f in Policy.__dataclass_fields__ if f != "calibration"}
    kwargs = {k: v for k, v in data.items() if k in fields}
    calibration = None
    if cal:
        cal = dict(cal)
        cal_fields = set(Calibration.__dataclass_fields__)
        cal_kwargs = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in cal.items() if k in cal_fields
        }
        calibration = Calibration(**cal_kwargs)
    return Policy(calibration=calibration, **kwargs)


def default_policy_path() -> Path:
    """Where a fitted policy is written/loaded by ``pdfmux calibrate``.

    Honours ``PDFMUX_POLICY_FILE``; otherwise ``~/.config/pdfmux/policy.json``.
    """
    env = os.environ.get("PDFMUX_POLICY_FILE")
    if env:
        return Path(env)
    return Path.home() / ".config" / "pdfmux" / "policy.json"


def load_policy_file(path: Path | None = None, base: Policy = DEFAULT_POLICY) -> Policy:
    """Load a fitted policy from disk if present, then apply env overrides.

    Returns the env-overridden ``base`` when no policy file exists, so the
    runtime always has a usable policy.
    """
    path = path or default_policy_path()
    policy = base
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            policy = policy_from_dict(data)
    except Exception:
        policy = base  # a broken policy file must never break extraction
    return load_policy(policy)
