"""Transparent, dependency-light statistical analysis.

The analyzer is deliberately modest: it uses standard-library statistics and
normal approximations, and exposes every intermediate value so a report can
trace a claim back to concrete runs.  It is not intended to replace a full
statistics package for confirmatory research.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

from .models import AnalysisSummary, ExperimentRun


class ResultAnalyzer:
    """Compute summary statistics and compare experiment groups."""

    Z_95 = 1.96

    @staticmethod
    def _numbers(values: Iterable[Any]) -> list[float]:
        result: list[float] = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result.append(number)
        return result

    @classmethod
    def summarize(
        cls,
        values: Iterable[Any],
        *,
        metric: str = "score",
        success_values: Iterable[Any] | None = None,
        baseline_values: Iterable[Any] | None = None,
    ) -> AnalysisSummary:
        numbers = cls._numbers(values)
        n = len(numbers)
        if not n:
            return AnalysisSummary(metric=metric, values=[], sample_count=0, notes=["No numeric observations"])
        mean = statistics.fmean(numbers)
        median = statistics.median(numbers)
        std = statistics.stdev(numbers) if n > 1 else 0.0
        margin = cls.Z_95 * std / math.sqrt(n) if n > 1 else 0.0
        baseline = cls._numbers(baseline_values or [])
        baseline_mean = statistics.fmean(baseline) if baseline else None
        effect = cls.effect_size(numbers, baseline) if baseline else None
        success_rate = None
        if success_values is not None:
            flags = [bool(v) for v in success_values]
            success_rate = sum(flags) / len(flags) if flags else 0.0
        return AnalysisSummary(
            metric=metric,
            values=numbers,
            mean=mean,
            median=median,
            std=std,
            confidence_interval=(mean - margin, mean + margin),
            effect_size=effect,
            success_rate=success_rate,
            sample_count=n,
            baseline_mean=baseline_mean,
        )

    @classmethod
    def from_runs(
        cls,
        runs: Iterable[ExperimentRun],
        *,
        metric: str = "score",
        success_threshold: float | None = None,
        baseline_values: Iterable[Any] | None = None,
    ) -> AnalysisSummary:
        runs = list(runs)
        values = [r.metrics.get(metric) for r in runs if metric in r.metrics]
        successes = None
        if success_threshold is not None:
            successes = [float(r.metrics.get(metric, float("nan"))) >= success_threshold for r in runs]
        return cls.summarize(
            values,
            metric=metric,
            success_values=successes,
            baseline_values=baseline_values,
        )

    @staticmethod
    def effect_size(values: Sequence[float], baseline: Sequence[float]) -> float | None:
        """Return a simple Cohen-d-like standardized mean difference."""

        a = [float(v) for v in values]
        b = [float(v) for v in baseline]
        if not a or not b:
            return None
        mean_diff = statistics.fmean(a) - statistics.fmean(b)
        # Pooled sample standard deviation.  A zero-variance comparison has a
        # useful signed result instead of raising a division-by-zero error.
        var_a = statistics.variance(a) if len(a) > 1 else 0.0
        var_b = statistics.variance(b) if len(b) > 1 else 0.0
        dof = max(1, len(a) + len(b) - 2)
        pooled = math.sqrt(max(0.0, ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / dof))
        if pooled == 0:
            if mean_diff > 0:
                return float("inf")
            if mean_diff < 0:
                return float("-inf")
            return 0.0
        return mean_diff / pooled

    @classmethod
    def compare(
        cls,
        groups: dict[str, Iterable[Any]],
        *,
        metric: str = "score",
        baseline: str | None = None,
    ) -> dict[str, AnalysisSummary]:
        """Summarize named groups, optionally against one baseline group."""

        clean = {name: cls._numbers(values) for name, values in groups.items()}
        baseline_values = clean.get(baseline, []) if baseline else []
        return {
            name: cls.summarize(values, metric=metric, baseline_values=baseline_values if name != baseline else None)
            for name, values in clean.items()
        }

