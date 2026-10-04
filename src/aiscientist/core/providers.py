"""Offline experiment providers.

Providers are intentionally tiny and deterministic.  They are the seam where
future integrations (LLM Lab, Agent Arena, Benchmark Factory) can be attached;
the development build never reaches the network or spends an API dollar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import inspect
import math
import random
import time
from typing import Any, Callable, Mapping

from .models import ExperimentConfig, FailureType, ProviderResult


class ExperimentProvider(ABC):
    """Common provider interface used by the scientific engine."""

    name = "provider"
    estimated_cost_per_run = 0.0

    @abstractmethod
    def run(self, config: ExperimentConfig, seed: int | None = None) -> ProviderResult:
        raise NotImplementedError

    def estimate_cost(self, config: ExperimentConfig) -> float:
        return float(self.estimated_cost_per_run)


def _strategy_name(config: ExperimentConfig) -> str:
    value = config.independent_variable
    if isinstance(value, Mapping):
        value = value.get("strategy", value.get("name", value.get("value", "unknown")))
    return str(value)


class SyntheticProvider(ExperimentProvider):
    """Deterministic synthetic benchmark provider.

    It models a memory strategy study with plausible quality/latency tradeoffs,
    but accepts custom ``parameters`` so users can create other safe toy
    experiments.  A local ``random.Random`` is seeded per run; global RNG state
    is never touched.
    """

    name = "SyntheticProvider"

    DEFAULT_PROFILES: dict[str, dict[str, float]] = {
        "No Memory": {"score": 0.43, "retention": 0.22, "latency": 0.10},
        "Sliding Window": {"score": 0.62, "retention": 0.51, "latency": 0.22},
        "Summary": {"score": 0.74, "retention": 0.68, "latency": 0.30},
        "Episodic": {"score": 0.84, "retention": 0.79, "latency": 0.39},
        "Vector": {"score": 0.80, "retention": 0.76, "latency": 0.55},
        # lowercase aliases make ad-hoc configs less surprising
        "no_memory": {"score": 0.43, "retention": 0.22, "latency": 0.10},
        "sliding_window": {"score": 0.62, "retention": 0.51, "latency": 0.22},
        "summary": {"score": 0.74, "retention": 0.68, "latency": 0.30},
        "episodic": {"score": 0.84, "retention": 0.79, "latency": 0.39},
        "vector": {"score": 0.80, "retention": 0.76, "latency": 0.55},
    }

    def __init__(self, profiles: Mapping[str, Mapping[str, float]] | None = None, *, noise: float = 0.08):
        self.profiles = {k: dict(v) for k, v in (profiles or self.DEFAULT_PROFILES).items()}
        self.noise = max(0.0, float(noise))

    def run(self, config: ExperimentConfig, seed: int | None = None) -> ProviderResult:
        started = time.perf_counter()
        run_seed = config.seed if seed is None else int(seed)
        rng = random.Random(run_seed)
        strategy = _strategy_name(config)
        profile = self.profiles.get(strategy, self.profiles.get(strategy.lower()))
        if profile is None:
            # Unknown treatments are still runnable: an explicitly configured
            # baseline can provide ``effect``/``score`` in parameters.
            params = config.parameters or {}
            profile = {
                "score": float(params.get("score", params.get("effect", 0.5))),
                "retention": float(params.get("retention", 0.5)),
                "latency": float(params.get("latency", 0.3)),
            }

        params = config.parameters or {}
        sample_size = max(1, int(params.get("sample_size", config.sample_size)))
        sigma = float(params.get("noise", self.noise))
        score_samples = [min(1.0, max(0.0, rng.gauss(float(profile.get("score", 0.5)), sigma))) for _ in range(sample_size)]
        retention_samples = [min(1.0, max(0.0, rng.gauss(float(profile.get("retention", 0.5)), sigma * 0.8))) for _ in range(sample_size)]
        latency_samples = [max(0.001, rng.gauss(float(profile.get("latency", 0.3)), sigma * 0.4)) for _ in range(sample_size)]
        score = sum(score_samples) / sample_size
        retention = sum(retention_samples) / sample_size
        latency = sum(latency_samples) / sample_size
        threshold = float((config.success_criteria or {}).get("min_score", 0.65)) if isinstance(config.success_criteria, Mapping) else 0.65
        success = score >= threshold
        runtime = time.perf_counter() - started
        return ProviderResult(
            metrics={"score": score, "retention": retention, "latency": latency, "success_rate": 1.0 if success else 0.0},
            observations=[f"Synthetic benchmark evaluated {strategy} on {sample_size} samples.", f"score={score:.3f}, retention={retention:.3f}, latency={latency:.3f}"],
            raw_output={"strategy": strategy, "samples": score_samples, "retention_samples": retention_samples, "latency_samples": latency_samples, "seed": run_seed},
            success=True,  # A low score is a negative result, not an infrastructure failure.
            runtime_seconds=runtime,
            estimated_cost=0.0,
        )


class PythonFunctionProvider(ExperimentProvider):
    """Adapter for a user-owned local Python function.

    The callable may return ``ProviderResult``, a metrics mapping, a numeric
    score, or ``(metrics, observations)``.  Exceptions are captured as
    infrastructure failures so one bad run does not lose the audit trail.
    """

    name = "PythonFunctionProvider"

    def __init__(self, function: Callable[..., Any]):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def run(self, config: ExperimentConfig, seed: int | None = None) -> ProviderResult:
        started = time.perf_counter()
        run_seed = config.seed if seed is None else int(seed)
        try:
            result = self._call(config, run_seed)
            normalized = self._normalize(result)
            normalized.runtime_seconds = time.perf_counter() - started
            return normalized
        except Exception as exc:  # noqa: BLE001 - provider boundary must record failures
            return ProviderResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                failure_type=FailureType.INFRASTRUCTURE,
                runtime_seconds=time.perf_counter() - started,
            )

    def _call(self, config: ExperimentConfig, seed: int) -> Any:
        # Prefer explicit two-argument convention, then gracefully support
        # simple one-argument and zero-argument functions.
        try:
            sig = inspect.signature(self.function)
            positional = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            required = [p for p in positional if p.default is p.empty]
        except (TypeError, ValueError):
            positional, required = [], []
        if len(required) >= 2 or len(positional) >= 2:
            return self.function(config, seed)
        if len(required) == 1 or len(positional) == 1:
            return self.function(config)
        return self.function()

    @staticmethod
    def _normalize(value: Any) -> ProviderResult:
        if isinstance(value, ProviderResult):
            return value
        if isinstance(value, Mapping):
            # Explicit provider-style mapping is supported.
            if "metrics" in value:
                return ProviderResult(
                    metrics={str(k): float(v) for k, v in dict(value.get("metrics", {})).items()},
                    observations=list(value.get("observations", [])),
                    raw_output=value.get("raw_output", value),
                    success=bool(value.get("success", True)),
                    error=value.get("error"),
                )
            return ProviderResult(metrics={str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}, raw_output=value)
        if isinstance(value, (int, float)):
            return ProviderResult(metrics={"score": float(value)}, raw_output=value)
        if isinstance(value, tuple) and value:
            metrics = value[0] if isinstance(value[0], Mapping) else {"score": float(value[0])}
            observations = list(value[1]) if len(value) > 1 and isinstance(value[1], (list, tuple)) else []
            return ProviderResult(metrics={str(k): float(v) for k, v in metrics.items()}, observations=observations, raw_output=value)
        raise TypeError("Python experiment function returned unsupported value")


class MockLLMExperimentProvider(ExperimentProvider):
    """Offline stand-in for a future LLM-backed experiment.

    It hashes the prompt/config into deterministic pseudo-token behaviour and
    never imports an SDK, reads an API key or performs network I/O.  Memory
    strategy configs are delegated to :class:`SyntheticProvider` so the demo
    remains scientifically interpretable.
    """

    name = "MockLLMExperimentProvider"

    def __init__(self, *, temperature: float = 0.2):
        self.temperature = max(0.0, float(temperature))
        self.synthetic = SyntheticProvider()

    def run(self, config: ExperimentConfig, seed: int | None = None) -> ProviderResult:
        strategy = _strategy_name(config)
        if strategy in self.synthetic.profiles:
            result = self.synthetic.run(config, seed)
            result.observations.insert(0, "Mock LLM evaluator: deterministic local response (no API call).")
            result.raw_output = {"mock_model": "rule-based-local", "delegate": result.raw_output}
            return result
        run_seed = config.seed if seed is None else int(seed)
        prompt = str((config.parameters or {}).get("prompt", config.procedure))
        digest = hashlib.sha256(f"{prompt}|{run_seed}".encode("utf-8")).digest()
        base = 0.45 + (int.from_bytes(digest[:4], "big") / 2**32) * 0.45
        # Temperature only controls deterministic dispersion; no global random.
        score = max(0.0, min(1.0, base + self.temperature * ((digest[4] / 255.0) - 0.5) * 0.1))
        return ProviderResult(
            metrics={"score": score, "token_efficiency": 0.5 + digest[5] / 510.0, "success_rate": 1.0 if score >= 0.65 else 0.0},
            observations=["Mock LLM generated a deterministic synthetic answer; no external model was contacted."],
            raw_output={"text": f"mock-answer-{digest.hex()[:12]}", "seed": run_seed},
            estimated_cost=0.0,
        )


__all__ = [
    "ExperimentProvider",
    "MockLLMExperimentProvider",
    "PythonFunctionProvider",
    "SyntheticProvider",
]
