"""Provider discovery — built-in, config file, and entry_points.

Discovery sources (in order):
  1. Built-in providers (Gemini, Claude, OpenAI, Ollama) — always registered
  2. Config file providers (~/.pdfmux/providers.yaml or ./.pdfmux.yaml)
  3. Entry-point plugins (pdfmux.providers group)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pdfmux.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Built-in provider classes (imported lazily to avoid hard deps)
_BUILTIN_CLASSES: list[type[LLMProvider]] = []


def _load_builtins() -> list[type[LLMProvider]]:
    """Import built-in provider classes."""
    global _BUILTIN_CLASSES
    if _BUILTIN_CLASSES:
        return _BUILTIN_CLASSES

    from pdfmux.providers.claude import ClaudeProvider
    from pdfmux.providers.gemini import GeminiProvider
    from pdfmux.providers.ollama import OllamaProvider
    from pdfmux.providers.openai_native import OpenAINativeProvider

    _BUILTIN_CLASSES = [GeminiProvider, ClaudeProvider, OpenAINativeProvider, OllamaProvider]
    return _BUILTIN_CLASSES


def _load_config_providers() -> list[LLMProvider]:
    """Load providers from YAML config files."""
    config_paths = [
        Path(os.environ.get("PDFMUX_CONFIG", "")) if os.environ.get("PDFMUX_CONFIG") else None,
        Path.cwd() / ".pdfmux.yaml",
        Path.home() / ".pdfmux" / "providers.yaml",
    ]

    for path in config_paths:
        if path and path.is_file():
            return _parse_config(path)

    return []


def _parse_config(path: Path) -> list[LLMProvider]:
    """Parse a YAML config file into provider instances."""
    try:
        import yaml
    except ImportError:
        logger.debug("pyyaml not installed — skipping config file %s", path)
        return []

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load provider config %s: %s", path, e)
        return []

    if not config or "providers" not in config:
        return []

    from pdfmux.providers.openai_compatible import OpenAICompatibleProvider

    providers = []
    for name, cfg in config["providers"].items():
        if cfg.get("type") != "openai_compatible":
            logger.warning("Unknown provider type '%s' for '%s' — skipping", cfg.get("type"), name)
            continue

        try:
            provider = OpenAICompatibleProvider(
                name=name,
                base_url=cfg["base_url"],
                api_key_env=cfg["api_key_env"],
                models=cfg.get("models", []),
                default_model=cfg.get("default_model", ""),
            )
            providers.append(provider)
            logger.debug("Loaded config provider: %s (%s)", name, cfg["base_url"])
        except (KeyError, TypeError) as e:
            logger.warning("Invalid provider config for '%s': %s", name, e)

    return providers


def _load_entrypoint_providers() -> list[LLMProvider]:
    """Load providers from Python entry_points (pdfmux.providers group)."""
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="pdfmux.providers")
    except Exception:
        return []

    providers = []
    for ep in eps:
        try:
            cls = ep.load()
            provider = cls()
            providers.append(provider)
            logger.debug("Loaded entry_point provider: %s", ep.name)
        except Exception as e:
            logger.warning("Failed to load entry_point provider '%s': %s", ep.name, e)

    return providers


def discover_all_providers() -> dict[str, LLMProvider]:
    """Discover all providers from all sources.

    Returns dict mapping provider name to instance.
    Later sources override earlier ones (entry_points > config > built-in).
    """
    providers: dict[str, LLMProvider] = {}

    # 1. Built-in
    for cls in _load_builtins():
        p = cls()
        providers[p.name] = p

    # 2. Config file
    for p in _load_config_providers():
        providers[p.name] = p

    # 3. Entry points
    for p in _load_entrypoint_providers():
        providers[p.name] = p

    return providers


def resolve_provider(provider_name: str | None = None, model: str | None = None) -> LLMProvider:
    """Resolve which LLM provider to use.

    Priority:
      1. Explicit PDFMUX_LLM_PROVIDER env var or provider_name arg
      2. Auto-detect from available API keys (Gemini > Claude > OpenAI > Ollama)

    Raises:
        ValueError: If no provider is available.
    """
    all_providers = discover_all_providers()
    name = provider_name or os.environ.get("PDFMUX_LLM_PROVIDER")

    if name:
        name = name.lower().strip()
        if name in all_providers:
            provider = all_providers[name]
            if not provider.sdk_installed():
                raise ValueError(
                    f"Provider '{name}' selected but SDK not installed. "
                    f"Install with: pip install pdfmux[llm-{name}]"
                )
            return provider
        valid = ", ".join(all_providers.keys())
        raise ValueError(f"Unknown LLM provider '{name}'. Available: {valid}")

    # Auto-detect: try built-ins first (stable order), then custom
    for cls in _load_builtins():
        p = cls()
        if p.available():
            logger.info("Auto-detected LLM provider: %s", p.name)
            return p

    # Try config/entry_point providers
    for name, p in all_providers.items():
        if p.available():
            logger.info("Auto-detected LLM provider: %s (custom)", name)
            return p

    raise ValueError(
        "No LLM provider available. Set one of: "
        "GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or configure a custom provider in ~/.pdfmux/providers.yaml"
    )


def available_providers() -> list[LLMProvider]:
    """Return all providers that have SDK installed and credentials configured."""
    return [p for p in discover_all_providers().values() if p.available()]


def all_provider_status() -> list[dict[str, str | bool]]:
    """Return status of all providers for doctor command."""
    result = []
    for name, p in discover_all_providers().items():
        result.append(
            {
                "name": name,
                "sdk_installed": p.sdk_installed(),
                "has_credentials": p.has_credentials(),
                "available": p.available(),
                "default_model": p.default_model or "(requires PDFMUX_LLM_MODEL)",
                "custom": not isinstance(p, tuple(_load_builtins())),
            }
        )
    return result
