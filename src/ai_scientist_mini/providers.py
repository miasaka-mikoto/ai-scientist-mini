"""Experiment provider interfaces and zero-cost deterministic providers."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import random
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .models import ExperimentDesign, FailureType


class ProviderError(RuntimeError):
    """A provider could not execute a design."""

    def __init__(self, message: str, failure_type: FailureType = FailureType.INFRASTRUCTURE_FAILURE):
        super().__init__(message)
        self.failure_type = failure_type


@dataclass
class ProviderResult:
    """Provider output before it is wrapped in an :class:`ExperimentRun`."""

    metrics: dict[str, float] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    api_cost: float = 0.0
    data_hash: str = ""
    environment: dict[str, str] = field(default_factory=dict)


class ExperimentProvider(ABC):
    """Common provider contract.

    Implementations must be deterministic when given the same design and
    seed.  No implementation in this package performs network requests.
    """

    name: str = "provider"

    @abstractmethod
    def execute(self, design: ExperimentDesign, *, seed: int, run_index: int = 0) -> ProviderResult:
        raise NotImplementedError

    def run(self, design: ExperimentDesign, *, seed: int, run_index: int = 0) -> ProviderResult:
        """Alias used by older adapters."""

        return self.execute(design, seed=seed, run_index=run_index)


def _hash_data(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SyntheticProvider(ExperimentProvider):
    """A reproducible benchmark with known latent strategy quality.

    The benchmark is intentionally transparent: strategy means and noise are
    published as attributes, allowing a study to validate its analysis and
    hypothesis update rules without claiming real-world scientific discovery.
    """

    name = "SyntheticProvider"
    strategy_means: Mapping[str, float] = {
        "no memory": 0.52,
        "sliding window": 0.66,
        "summary": 0.73,
        "episodic": 0.81,
        "vector": 0.77,
    }

    def __init__(self, *, noise: float = 0.055, latency_base: float = 0.05):
        self.noise = max(0.0, float(noise))
        self.latency_base = max(0.0, float(latency_base))

    @classmethod
    def strategy_for(cls, design: ExperimentDesign) -> str:
        cfg = design.config or {}
        for key in ("strategy", "memory_strategy", "variant", "name"):
            value = cfg.get(key)
            if value:
                return str(value)
        if design.independent_variable:
            return str(design.independent_variable)
        return design.name or "No Memory"

    def execute(self, design: ExperimentDesign, *, seed: int, run_index: int = 0) -> ProviderResult:
        if design.sample_size < 1:
            raise ProviderError("sample_size must be positive", FailureType.INVALID_DESIGN)
        rng = random.Random(int(seed))
        strategy = self.strategy_for(design)
        key = strategy.strip().lower()
        mean = self.strategy_means.get(key)
        if mean is None:
            # Unknown strategies are valid exploratory arms, but are centered
            # at a neutral score rather than silently receiving a winner's score.
            mean = 0.60
        score = min(1.0, max(0.0, rng.gauss(mean, self.noise)))
        # Keep a second observable metric for dashboard/report examples.
        latency = max(0.001, self.latency_base + rng.random() * 0.04 + (0.02 if key == "vector" else 0.0))
        success = 1.0 if score >= float((design.config or {}).get("success_threshold", 0.60)) else 0.0
        canonical = {
            "dataset": design.dataset,
            "strategy": strategy,
            "sample_size": design.sample_size,
            "seed": int(seed),
            "run_index": int(run_index),
        }
        return ProviderResult(
            metrics={"score": score, "latency_seconds": latency, "success": success},
            raw_result={"strategy": strategy, "latent_mean": mean, "sample_size": design.sample_size},
            logs=[f"Synthetic benchmark evaluated strategy={strategy!r}", f"seed={seed}"],
            runtime_seconds=0.001,
            api_cost=0.0,
            data_hash=_hash_data(canonical),
            environment={"provider": self.name, "python": platform.python_version(), "platform": platform.platform()},
        )


class PythonFunctionProvider(ExperimentProvider):
    """Execute a user-supplied local Python function.

    The callable may accept ``(config, seed)``, ``(design, seed)`` or a single
    argument.  It may return a number, a mapping of metrics, or a
    :class:`ProviderResult`.  This is intentionally local-only and does not
    evaluate arbitrary source text.
    """

    name = "PythonFunctionProvider"

    def __init__(self, function: Callable[..., Any], *, function_name: str | None = None):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function
        self.function_name = function_name or getattr(function, "__name__", "callable")

    def _call(self, design: ExperimentDesign, seed: int) -> Any:
        try:
            sig = inspect.signature(self.function)
            params = list(sig.parameters.values())
            positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if any(p.kind == p.VAR_POSITIONAL for p in params) or len(positional) >= 2:
                return self.function(design.config, seed)
            if len(positional) == 1:
                return self.function(design.config)
            return self.function()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Python function failed: {exc}", FailureType.INFRASTRUCTURE_FAILURE) from exc

    def execute(self, design: ExperimentDesign, *, seed: int, run_index: int = 0) -> ProviderResult:
        started = time.perf_counter()
        value = self._call(design, int(seed))
        runtime = time.perf_counter() - started
        if isinstance(value, ProviderResult):
            if not value.environment:
                value.environment = {"provider": self.name, "function": self.function_name}
            if not value.data_hash:
                value.data_hash = _hash_data({"config": design.config, "seed": seed})
            value.runtime_seconds = value.runtime_seconds or runtime
            return value
        if isinstance(value, Mapping):
            metrics_raw = value.get("metrics", value)
            metrics = {}
            for key, item in metrics_raw.items():
                try:
                    metrics[str(key)] = float(item)
                except (TypeError, ValueError):
                    continue
            raw = dict(value)
        else:
            try:
                metrics = {design.metric or "score": float(value)}
            except (TypeError, ValueError) as exc:
                raise ProviderError("Python function must return a number or metrics mapping", FailureType.INVALID_DESIGN) from exc
            raw = {"value": value}
        return ProviderResult(
            metrics=metrics,
            raw_result=raw,
            logs=[f"Executed local function {self.function_name}"],
            runtime_seconds=runtime,
            api_cost=0.0,
            data_hash=_hash_data({"config": design.config, "seed": seed}),
            environment={"provider": self.name, "function": self.function_name},
        )


class MockLLMExperimentProvider(SyntheticProvider):
    """Mock model provider reserved for future LLM-backed experiments.

    It intentionally inherits the deterministic synthetic behavior.  The
    separate type lets integration adapters select an LLM-shaped provider
    without introducing an API key or network dependency during development.
    """

    name = "MockLLMExperimentProvider"

    def execute(self, design: ExperimentDesign, *, seed: int, run_index: int = 0) -> ProviderResult:
        result = super().execute(design, seed=seed, run_index=run_index)
        result.raw_result["model"] = "mock-llm"
        result.logs.insert(0, "No external model API used; mock response generated locally")
        return result


class ProviderRegistry:
    """Small name-to-provider registry used by JSON/CLI adapters."""

    def __init__(self):
        self._providers: dict[str, ExperimentProvider] = {}

    def register(self, provider: ExperimentProvider, name: str | None = None) -> ExperimentProvider:
        key = name or provider.name
        self._providers[key] = provider
        return provider

    def get(self, name: str) -> ExperimentProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown experiment provider: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._providers)

