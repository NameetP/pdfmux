"""ML-based heading classification — supplements heuristic detection.

Uses a lightweight sklearn model trained on font features to classify
text lines as headings. Works alongside the existing heuristic pipeline
in headings.py, not as a replacement.

The model is loaded lazily on first use. If the model file is missing
or sklearn is not installed, gracefully falls back to heuristic-only.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fitz

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "models" / "heading_classifier.pkl"
_model_cache: dict | None = None


def _load_model() -> dict | None:
    """Lazy-load the heading classifier model."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not _MODEL_PATH.exists():
        logger.debug("ML heading model not found at %s", _MODEL_PATH)
        return None

    try:
        import pickle

        with open(_MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
        return _model_cache
    except Exception as e:
        logger.warning("Failed to load ML heading model: %s", e)
        return None


def classify_headings(
    candidates: list,
    body_size: float,
    page: fitz.Page,
    threshold: float = 0.65,
) -> dict[str, int]:
    """Classify heading candidates using the ML model.

    Returns a heading_map (text → level) compatible with _assign_levels().
    Only returns candidates above the probability threshold.

    Args:
        candidates: List of _HeadingCandidate from _build_font_census()
        body_size: Detected body font size
        page: fitz.Page object for position info
        threshold: Minimum probability to classify as heading (default 0.65)

    Returns:
        Dict mapping heading text to level (always 1 for now)
    """
    model_data = _load_model()
    if model_data is None:
        return {}

    model = model_data["model"]
    feature_cols = model_data["feature_cols"]

    page_height = page.rect.height

    if body_size <= 0 or page_height <= 0:
        return {}

    heading_map: dict[str, int] = {}

    for c in candidates:
        text = c.text.strip()
        if len(text) < 2 or len(text) > 120:
            continue

        # Extract features (same order as training)
        features = {
            "size_ratio": c.size / body_size if body_size > 0 else 1.0,
            "is_bold": int(c.is_bold),
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_period": int(text.rstrip().endswith(".")),
            "is_all_caps": int(text.isupper() and any(ch.isalpha() for ch in text)),
            "is_numeric": int(text.strip().strip(".").isdigit()),
            "starts_with_number": int(bool(re.match(r"^\d+[\.\)]\s", text))),
            "y_position_pct": c.y_position / page_height if page_height > 0 else 0.5,
            "char_density": sum(1 for ch in text if not ch.isspace()) / max(len(text), 1),
            "has_colon": int(":" in text),
            "has_question_mark": int("?" in text),
        }

        # Build feature vector in correct order
        import numpy as np

        x_features = np.array([[features[col] for col in feature_cols]], dtype=np.float32)

        prob = model.predict_proba(x_features)[0, 1]
        if prob >= threshold:
            heading_map[text] = 1

    return heading_map
