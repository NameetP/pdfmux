"""Configuration profiles — save common settings as named bundles.

Profiles bundle the most common CLI flags (quality, mode, schema, chunking,
LLM provider/model, etc.) under a short name so users don't have to retype
them on every invocation.

Storage:
    ~/.config/pdfmux/profiles.yaml  (XDG_CONFIG_HOME aware)

Built-in presets are always available even if no config file exists. User
profiles override built-ins of the same name.

Public API:
    load_profile(name)    → dict of settings
    save_profile(name, settings)
    list_profiles()       → list of (name, source) tuples
    delete_profile(name)
    apply_profile_defaults(profile, explicit) → merge profile into explicit flags
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

# Each preset is a dict of CLI option names → values. Keys match the
# parameter names used by `pdfmux convert` (quality, mode, schema, chunk,
# max_tokens, overlap, format, llm_provider, llm_model, budget, confidence).
BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "invoices": {
        "quality": "high",
        "mode": "balanced",
        "schema": "invoice",
        "format": "json",
        "confidence": True,
    },
    "receipts": {
        "quality": "standard",
        "mode": "economy",
        "schema": "receipt",
        "format": "json",
        "confidence": True,
    },
    "papers": {
        "quality": "high",
        "mode": "balanced",
        "chunk": True,
        "max_tokens": 800,
        "overlap": 80,
        "format": "llm",
    },
    "contracts": {
        "quality": "high",
        "mode": "premium",
        "format": "markdown",
        "confidence": True,
    },
    "bulk-rag": {
        "quality": "standard",
        "mode": "economy",
        "chunk": True,
        "max_tokens": 500,
        "overlap": 50,
        "format": "llm",
    },
}


# Settings keys that are valid for profiles. Keeps save_profile honest.
ALLOWED_KEYS: set[str] = {
    "quality",
    "mode",
    "schema",
    "format",
    "chunk",
    "max_tokens",
    "overlap",
    "confidence",
    "llm_provider",
    "llm_model",
    "budget",
}


# ---------------------------------------------------------------------------
# Storage location
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    """Return the pdfmux config directory, respecting XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.environ.get("PDFMUX_CONFIG_DIR")
    if base:
        return Path(base) / "pdfmux"
    return Path.home() / ".config" / "pdfmux"


def _profiles_path() -> Path:
    return _config_dir() / "profiles.yaml"


# ---------------------------------------------------------------------------
# Read/write helpers
# ---------------------------------------------------------------------------


def _load_user_profiles() -> dict[str, dict[str, Any]]:
    """Load user-defined profiles from disk. Returns empty dict if missing."""
    path = _profiles_path()
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        return {}
    # Filter to dicts only
    return {k: v for k, v in profiles.items() if isinstance(v, dict)}


def _write_user_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    """Write user profiles to disk, creating parent dirs as needed."""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "Saving profiles requires PyYAML. Install with: pip install pyyaml"
        ) from e

    path = _profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"profiles": profiles}
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_profile(name: str) -> dict[str, Any]:
    """Return the settings for a profile by name.

    User profiles take precedence over built-ins. Raises KeyError if neither
    has the profile.
    """
    name = name.strip()
    user = _load_user_profiles()
    if name in user:
        return dict(user[name])
    if name in BUILTIN_PROFILES:
        return dict(BUILTIN_PROFILES[name])
    raise KeyError(f"profile '{name}' not found")


def save_profile(name: str, settings: dict[str, Any]) -> Path:
    """Persist a profile under the given name. Returns the file path written.

    Validates that keys are all in ALLOWED_KEYS. Unknown keys raise ValueError.
    """
    name = name.strip()
    if not name:
        raise ValueError("profile name cannot be empty")

    cleaned: dict[str, Any] = {}
    for k, v in settings.items():
        if k not in ALLOWED_KEYS:
            raise ValueError(f"unknown profile key '{k}'. Allowed: {sorted(ALLOWED_KEYS)}")
        if v is None:
            continue
        cleaned[k] = v

    user = _load_user_profiles()
    user[name] = cleaned
    _write_user_profiles(user)
    return _profiles_path()


def list_profiles() -> list[tuple[str, str]]:
    """List all known profiles with their source.

    Returns list of (name, source) where source is 'builtin' or 'user'.
    User profiles override built-ins; the user version wins in this list too.
    """
    user = _load_user_profiles()
    out: dict[str, str] = {}
    for name in BUILTIN_PROFILES:
        out[name] = "builtin"
    for name in user:
        out[name] = "user"
    return sorted(out.items(), key=lambda x: x[0])


def delete_profile(name: str) -> bool:
    """Remove a user profile. Returns True if deleted, False if absent.

    Built-in profiles cannot be deleted — raises ValueError.
    """
    name = name.strip()
    user = _load_user_profiles()
    if name in user:
        del user[name]
        _write_user_profiles(user)
        return True
    if name in BUILTIN_PROFILES:
        raise ValueError(
            f"'{name}' is a built-in profile and cannot be deleted. "
            f"Override it by saving a profile with the same name."
        )
    return False


def apply_profile_defaults(
    profile_name: str | None,
    explicit: dict[str, Any],
) -> dict[str, Any]:
    """Merge a profile's settings into explicit flags.

    Explicit values (not None / not the sentinel default) win. Returns a new
    dict with profile values filled in for any keys the user did not set.

    `explicit` should map flag-name → user-supplied-value. Use None to mean
    "not supplied"; for booleans treat False as user-supplied unless you
    pass None explicitly.
    """
    if not profile_name:
        return dict(explicit)

    profile = load_profile(profile_name)
    merged = dict(explicit)
    for k, v in profile.items():
        if merged.get(k) is None:
            merged[k] = v
    return merged
