"""Base classes for LLM providers.

Every provider implements the LLMProvider ABC. The provider registry
discovers providers from built-in classes, config files, and entry_points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a model available from a provider."""

    id: str
    supports_vision: bool = True
    capabilities: tuple[str, ...] = ()  # "ocr", "tables", "handwriting", "structured", "charts"
    input_cost_per_mtok: float | None = None  # $ per million input tokens
    output_cost_per_mtok: float | None = None  # $ per million output tokens
    max_input_tokens: int = 128_000


@dataclass(frozen=True)
class CostEstimate:
    """Estimated or actual cost for an extraction."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMProvider(ABC):
    """Base class for all LLM vision providers.

    Subclass this to add a new provider. Implement the abstract methods
    and the provider will be auto-discovered if registered.
    """

    name: str = ""
    default_model: str = ""

    @abstractmethod
    def available(self) -> bool:
        """True if SDK installed AND credentials configured."""
        ...

    @abstractmethod
    def extract_page(self, image_bytes: bytes, prompt: str, model: str | None = None) -> str:
        """Send a page image to the LLM and return extracted text."""
        ...

    def sdk_installed(self) -> bool:
        """True if the provider's Python SDK is importable."""
        return False

    def has_credentials(self) -> bool:
        """True if API key or endpoint is configured."""
        return False

    def supported_models(self) -> list[ModelInfo]:
        """Return list of models this provider offers."""
        if self.default_model:
            return [ModelInfo(id=self.default_model)]
        return []

    def estimate_cost(self, image_bytes_count: int, prompt_tokens: int = 200) -> CostEstimate:
        """Estimate cost for extracting one page. Override for accuracy."""
        return CostEstimate()

    def extract_page_with_cost(
        self, image_bytes: bytes, prompt: str, model: str | None = None
    ) -> tuple[str, CostEstimate]:
        """Extract page and return text + cost. Override for real cost tracking."""
        text = self.extract_page(image_bytes, prompt, model)
        cost = self.estimate_cost(len(image_bytes))
        return text, cost
