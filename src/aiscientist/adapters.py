"""Stable JSON boundary for future experiment integrations.

The first release deliberately ships only local providers.  Integrations such
as LLM Lab, Agent Arena, Paper2Lab, and Benchmark Factory can implement the
same request/response contract later without importing this project's internal
classes or sharing a database.  The adapter is intentionally boring: one JSON
object in, one JSON object out, with provider errors represented as data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from .core.models import ExperimentConfig, ProviderResult
from .core.providers import (
    ExperimentProvider,
    MockLLMExperimentProvider,
    SyntheticProvider,
)


ADAPTER_PROTOCOL_VERSION = "1"


class AdapterError(RuntimeError):
    """Raised for malformed adapter requests or unknown local providers."""


@dataclass(frozen=True)
class AdapterRequest:
    """JSON-compatible request sent across an integration boundary."""

    provider: str
    config: dict[str, Any]
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterRequest":
        if not isinstance(payload, Mapping):
            raise AdapterError("adapter request must be a JSON object")
        provider = str(payload.get("provider", "synthetic")).strip().lower()
        raw_config = payload.get("config", payload.get("experiment"))
        if not isinstance(raw_config, Mapping):
            raise AdapterError("adapter request requires a config object")
        raw_seed = payload.get("seed")
        seed = None if raw_seed is None else int(raw_seed)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AdapterError("metadata must be an object")
        return cls(provider=provider, config=dict(raw_config), seed=seed, metadata=dict(metadata))


@dataclass(frozen=True)
class AdapterResponse:
    """Provider-neutral response that can be persisted or forwarded."""

    provider: str
    result: ProviderResult
    protocol_version: str = ADAPTER_PROTOCOL_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "provider": self.provider,
            "result": self.result.to_dict(),
            "metadata": dict(self.metadata),
        }


class ExperimentAdapter(ABC):
    """Minimal interface future external adapters must implement."""

    name = "adapter"

    @abstractmethod
    def execute(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedError


class LocalJsonAdapter(ExperimentAdapter):
    """Execute the two development providers through a JSON boundary.

    No network client, API key, or paid SDK is reachable from this class.  A
    caller can still inject a test provider explicitly, but the default
    registry is restricted to ``SyntheticProvider`` and
    ``MockLLMExperimentProvider``.
    """

    name = "local-json"

    def __init__(self, providers: Mapping[str, ExperimentProvider] | None = None):
        self.providers: dict[str, ExperimentProvider] = {
            "synthetic": SyntheticProvider(),
            "syntheticprovider": SyntheticProvider(),
            "mock": MockLLMExperimentProvider(),
            "mockllm": MockLLMExperimentProvider(),
            "mockllmexperimentprovider": MockLLMExperimentProvider(),
        }
        if providers:
            self.providers.update({str(k).lower(): v for k, v in providers.items()})

    def execute(self, request: AdapterRequest) -> AdapterResponse:
        provider = self.providers.get(request.provider)
        if provider is None:
            raise AdapterError(
                f"unknown local provider {request.provider!r}; "
                f"available: {', '.join(sorted(self.providers))}"
            )
        try:
            config = ExperimentConfig.from_dict(request.config)
            result = provider.run(config, request.seed)
        except Exception as exc:  # provider boundary: return inspectable data
            result = ProviderResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return AdapterResponse(
            provider=getattr(provider, "name", request.provider),
            result=result,
            metadata=dict(request.metadata),
        )

    def execute_dict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.execute(AdapterRequest.from_dict(payload)).to_dict()

    def execute_json(self, text: str) -> str:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"invalid adapter JSON: {exc}") from exc
        return json.dumps(self.execute_dict(payload), ensure_ascii=False, sort_keys=True)


class ReservedExternalAdapter(ExperimentAdapter):
    """Named placeholder for a future external integration.

    Keeping these names in the public surface makes the integration plan
    explicit while ensuring development cannot accidentally spend money or
    perform network I/O.
    """

    def __init__(self, name: str):
        self.name = name

    def execute(self, request: AdapterRequest) -> AdapterResponse:  # noqa: ARG002
        raise NotImplementedError(
            f"{self.name} adapter is reserved for a future integration; "
            "use LocalJsonAdapter with a synthetic/mock provider during development"
        )


class LLMLabAdapter(ReservedExternalAdapter):
    def __init__(self):
        super().__init__("LLM Lab")


class AgentArenaAdapter(ReservedExternalAdapter):
    def __init__(self):
        super().__init__("Agent Arena")


class BenchmarkFactoryAdapter(ReservedExternalAdapter):
    def __init__(self):
        super().__init__("Synthetic Benchmark Factory")


class Paper2LabAdapter(ReservedExternalAdapter):
    def __init__(self):
        super().__init__("Paper2Lab")


__all__ = [
    "ADAPTER_PROTOCOL_VERSION",
    "AdapterError",
    "AdapterRequest",
    "AdapterResponse",
    "ExperimentAdapter",
    "LocalJsonAdapter",
    "ReservedExternalAdapter",
    "LLMLabAdapter",
    "AgentArenaAdapter",
    "BenchmarkFactoryAdapter",
    "Paper2LabAdapter",
]
