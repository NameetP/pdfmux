"""Calibration fitting — turn raw audit scores into calibrated probabilities.

Stdlib-only (no sklearn/numpy). Two monotonic methods over the single raw
audit score:

- **Isotonic** — Pool-Adjacent-Violators (PAVA). Non-parametric, monotonic
  non-decreasing; the most faithful when enough labelled data exists.
- **Platt** — a one-feature logistic ``1/(1+e^-(a·x+b))`` fit by gradient
  descent. Smoother, better on small samples.

Plus :func:`expected_calibration_error` (ECE) to score a fit, and
:func:`fit_calibration` which fits, measures ECE before/after, and returns a
:class:`~pdfmux.policy.Calibration` ready to write into a policy file. The
closed loop is: fit here → write policy → runtime reload → ``Calibration.apply``.

The calibration *math* is public domain; the patentable element is the closed
loop over the audit features (fit → write policy → reload → apply), not the
arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import replace

from pdfmux.policy import Calibration


def _pava(values: list[float]) -> list[float]:
    """Pool-Adjacent-Violators: nearest monotonic non-decreasing fit (L2).

    Returns a list the same length as ``values`` where each element is the
    fitted (pooled-mean) value, guaranteed non-decreasing.
    """
    # Each block: [pooled_mean, weight (count)].
    blocks: list[list[float]] = []
    for v in values:
        blocks.append([float(v), 1.0])
        # Merge backwards while the previous block violates monotonicity.
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0] + 1e-12:
            v1, w1 = blocks.pop()
            v0, w0 = blocks.pop()
            nw = w0 + w1
            blocks.append([(v0 * w0 + v1 * w1) / nw, nw])
    out: list[float] = []
    for mean, weight in blocks:
        out.extend([mean] * int(weight))
    return out


def fit_isotonic(scores: list[float], labels: list[int]) -> Calibration:
    """Fit an isotonic (PAVA) calibration map of P(good) against raw score."""
    if not scores:
        return Calibration(method="identity")
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    xs = [float(p[0]) for p in pairs]
    ys = [float(p[1]) for p in pairs]
    fitted = _pava(ys)

    # Collapse tied x values to a single knot (averaging their fitted y).
    knots_x: list[float] = []
    knots_y: list[float] = []
    i = 0
    n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        avg = sum(fitted[i : j + 1]) / (j + 1 - i)
        knots_x.append(xs[i])
        knots_y.append(max(0.0, min(1.0, avg)))
        i = j + 1

    return Calibration(method="isotonic", knots_x=tuple(knots_x), knots_y=tuple(knots_y))


def fit_platt(
    scores: list[float],
    labels: list[int],
    *,
    iters: int = 3000,
    lr: float = 0.5,
) -> Calibration:
    """Fit a one-feature logistic (Platt) map by batch gradient descent."""
    if not scores:
        return Calibration(method="identity")
    a, b = 1.0, 0.0
    n = len(scores)
    for _ in range(iters):
        grad_a = grad_b = 0.0
        for x, t in zip(scores, labels, strict=True):
            z = a * x + b
            # numerically-stable sigmoid
            p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            err = p - t
            grad_a += err * x
            grad_b += err
        a -= lr * grad_a / n
        b -= lr * grad_b / n
    return Calibration(method="platt", platt_a=a, platt_b=b)


def expected_calibration_error(
    probs: list[float],
    labels: list[int],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: Σ_bins |acc - conf| · (n_bin / N)."""
    if not probs:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, t in zip(probs, labels, strict=True):
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx].append((p, t))
    n = len(probs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(t for _, t in b) / len(b)
        ece += abs(acc - conf) * len(b) / n
    return ece


def fit_calibration(
    scores: list[float],
    labels: list[int],
    *,
    method: str = "isotonic",
    fitted_on: str = "",
) -> Calibration:
    """Fit a calibration map and annotate it with ECE before/after + provenance."""
    labels = [int(round(label)) for label in labels]
    cal = fit_platt(scores, labels) if method == "platt" else fit_isotonic(scores, labels)
    ece_before = expected_calibration_error(scores, labels)
    ece_after = expected_calibration_error([cal.apply(s) for s in scores], labels)
    return replace(
        cal,
        fitted_on=fitted_on,
        n_samples=len(scores),
        ece_before=round(ece_before, 6),
        ece_after=round(ece_after, 6),
    )
