"""Small, dependency-free statistical helpers for AI Scientist Mini.

The project deliberately keeps the first analysis backend boring and
reproducible.  None of the functions in this module call a model or an
external service; they operate only on the values supplied by an experiment.
``numpy`` (or ``scipy``) is optional.  When it is installed, numpy-like
iterables work transparently, but the standard library implementation is used
by default so a fresh checkout can run with Python alone.

The public API is intentionally useful both as individual functions and as a
single :class:`AnalysisSummary` object::

    summary = AnalysisSummary.from_values(
        [0.51, 0.55, 0.53], outcomes=[True, True, False]
    )
    summary.mean, summary.confidence_interval

Confidence intervals use a normal critical value by default.  A ``method``
of ``"t"`` can be requested for a small-sample Student-t approximation (and
uses SciPy when available).  This is a deliberately explicit approximation,
not a claim that a statistical test has discovered a scientific truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, inf, isfinite, sqrt
from numbers import Number, Real
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "AnalysisSummary",
    "ResultAnalyzer",
    "mean",
    "median",
    "std",
    "standard_deviation",
    "confidence_interval",
    "effect_size",
    "cohens_d",
    "success_rate",
    "summarize",
    "analyze",
    "analyze_values",
    "compare_groups",
]


NumberLike = Real


def _materialize(values: Iterable[NumberLike] | NumberLike) -> list[float]:
    """Return finite floats from a numeric iterable.

    A surprising number of experiment bugs are caused by silently dropping a
    NaN or by treating a string as an iterable of observations.  Analysis is a
    boundary in the scientific audit trail, so invalid values fail loudly and
    with a useful error instead.
    """

    if isinstance(values, Number) and not isinstance(values, (str, bytes)):
        raw: Iterable[Any] = [values]
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError("values must be a numeric iterable, not a string")
        try:
            raw = iter(values)  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive branch
            # A zero-dimensional numpy scalar is not iterable but exposes the
            # same ``item`` protocol as ordinary numpy scalars.  Supporting it
            # here keeps numpy optional without importing it.
            item = getattr(values, "item", None)
            if callable(item):
                raw = [item()]
            else:
                raise TypeError("values must be a numeric iterable") from exc

    result: list[float] = []
    for value in raw:
        # bool is a Number and is intentionally accepted as 0/1.  This is
        # convenient when analysing binary benchmark metrics.
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"non-numeric observation: {value!r}") from exc
        if not isfinite(number):
            raise ValueError(f"observations must be finite; got {value!r}")
        result.append(number)
    return result


def _require_nonempty(values: Iterable[NumberLike] | NumberLike) -> list[float]:
    result = _materialize(values)
    if not result:
        raise ValueError("at least one observation is required")
    return result


def mean(values: Iterable[NumberLike] | NumberLike) -> float:
    """Return the arithmetic mean of ``values``.

    ``ValueError`` is raised for an empty iterable rather than returning zero;
    zero is a valid result and would otherwise hide a missing experiment run.
    """

    observations = _require_nonempty(values)
    return fsum(observations) / len(observations)


def median(values: Iterable[NumberLike] | NumberLike) -> float:
    """Return the median of ``values`` (with standard even-sample behaviour)."""

    observations = _require_nonempty(values)
    observations.sort()
    n = len(observations)
    middle = n // 2
    if n % 2:
        return observations[middle]
    return (observations[middle - 1] + observations[middle]) / 2.0


def _variance(observations: Sequence[float], ddof: int = 0) -> float:
    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    n = len(observations)
    if n <= ddof:
        raise ValueError(
            f"at least {ddof + 1} observations are required for ddof={ddof}"
        )
    centre = fsum(observations) / n
    # A two-pass calculation is stable enough for benchmark-sized values and
    # avoids a dependency on numpy just for variance.
    return fsum((value - centre) ** 2 for value in observations) / (n - ddof)


def std(
    values: Iterable[NumberLike] | NumberLike,
    ddof: int = 0,
    *,
    sample: bool | None = None,
) -> float:
    """Return standard deviation.

    By default this is the population standard deviation (``ddof=0``),
    matching ``numpy.std``.  Pass ``sample=True`` or ``ddof=1`` for the sample
    estimator.  ``sample`` is provided as a readable convenience for callers
    writing experiment configuration files.
    """

    if sample is not None:
        if not isinstance(sample, bool):
            raise TypeError("sample must be a bool when supplied")
        ddof = 1 if sample else 0
    observations = _require_nonempty(values)
    return sqrt(_variance(observations, ddof=ddof))


standard_deviation = std


# Common Student-t critical values for a two-sided 95% interval.  The table
# keeps the no-SciPy path useful and deterministic; arbitrary confidence
# levels use a Cornish-Fisher approximation below.
_T95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _critical_value(confidence: float, n: int, method: str) -> float:
    alpha = (1.0 - confidence) / 2.0
    z = NormalDist().inv_cdf(1.0 - alpha)
    normalised = method.strip().lower().replace("-", "_")
    if normalised in {"normal", "z", "gaussian"}:
        return z
    if normalised not in {"t", "student_t", "student"}:
        raise ValueError("method must be 'normal' or 't'")

    degrees = max(1, n - 1)
    # Prefer scipy if present, but never require it for the core application.
    try:  # pragma: no cover - availability depends on the host environment
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(1.0 - alpha, degrees))
    except Exception:
        pass

    if abs(confidence - 0.95) < 1e-12 and degrees in _T95:
        return _T95[degrees]

    # Cornish-Fisher expansion of the t quantile.  It converges quickly for
    # ordinary confidence levels and approaches the normal quantile as n grows.
    nu = float(degrees)
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2
    correction = (
        (z3 + z) / (4.0 * nu)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * nu**2)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * nu**3)
    )
    return z + correction


def confidence_interval(
    values: Iterable[NumberLike] | NumberLike,
    confidence: float = 0.95,
    *,
    method: str = "normal",
    ddof: int = 1,
    return_margin: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Return a two-sided confidence interval for the sample mean.

    Parameters
    ----------
    confidence:
        Coverage in ``(0, 1)``; ``0.95`` means a 95% interval.
    method:
        ``"normal"`` (default) or ``"t"``.  The latter uses a Student-t
        critical value where possible.
    ddof:
        Variance degrees of freedom used for the standard error.  ``1`` is
        the conventional sample estimate.
    return_margin:
        If true, return ``(lower, upper, margin)`` as a convenience.

    Empty observations return ``(0.0, 0.0)`` and a single observation has a
    zero-width interval because there is no estimable sampling variance.  This
    makes dashboards deterministic while the accompanying sample size still
    exposes the limitation.
    """

    if not isinstance(confidence, Real) or not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    observations = _materialize(values)
    if not observations:
        empty = (0.0, 0.0, 0.0)
        return empty if return_margin else empty[:2]
    n = len(observations)
    centre = fsum(observations) / n
    if n <= ddof:
        margin = 0.0
    else:
        standard_error = sqrt(_variance(observations, ddof=ddof) / n)
        margin = _critical_value(float(confidence), n, method) * standard_error
    interval = (centre - margin, centre + margin)
    if return_margin:
        return interval[0], interval[1], margin
    return interval


def _is_scalar(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, (str, bytes))


def effect_size(
    group_a: Iterable[NumberLike] | NumberLike,
    group_b: Iterable[NumberLike] | NumberLike | None = None,
    *,
    paired: bool = False,
    kind: str = "cohen_d",
) -> float | None:
    """Return a simple standardised mean difference.

    ``kind="cohen_d"`` uses the pooled sample standard deviation for two
    independent groups.  ``paired=True`` standardises paired differences.
    ``group_b`` may be a scalar, in which case a one-sample effect against the
    scalar baseline is calculated.  ``kind="hedges_g"`` applies the usual
    small-sample correction and ``kind="glass_delta"`` uses group B's sample
    deviation as the denominator.

    If both groups have zero variance, a zero difference returns ``0.0`` and a
    non-zero difference returns signed infinity.  That is preferable to a NaN
    because it preserves the direction of an unambiguous deterministic result.
    """

    if group_b is None:
        return None  # type: ignore[return-value]
    normalised_kind = kind.strip().lower().replace("-", "_")
    if normalised_kind not in {"cohen_d", "d", "hedges_g", "g", "glass", "glass_delta", "delta"}:
        raise ValueError("kind must be 'cohen_d', 'hedges_g', or 'glass_delta'")
    a = _require_nonempty(group_a)
    if _is_scalar(group_b):
        b_scalar = float(group_b)  # type: ignore[arg-type]
        if not isfinite(b_scalar):
            raise ValueError("baseline must be finite")
        difference = mean(a) - b_scalar
        if len(a) < 2:
            denominator = 0.0
        else:
            denominator = std(a, sample=True)
        value = _standardise(difference, denominator)
        if normalised_kind in {"hedges_g", "g"}:
            value *= _hedges_correction(len(a) - 1)
        return value

    b = _require_nonempty(group_b)
    difference = mean(a) - mean(b)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired groups must contain the same number of observations")
        differences = [left - right for left, right in zip(a, b)]
        denominator = std(differences, sample=True) if len(differences) >= 2 else 0.0
        value = _standardise(mean(differences), denominator)
        if normalised_kind in {"hedges_g", "g"}:
            value *= _hedges_correction(len(differences) - 1)
        return value

    if normalised_kind in {"glass", "glass_delta", "delta"}:
        denominator = std(b, sample=True) if len(b) >= 2 else 0.0
    else:
        # A singleton group contributes zero variance, but it need not make
        # the *other* group's pooled comparison undefined.  This convention
        # keeps a treatment-vs-single-baseline comparison finite whenever the
        # treatment has replicates, while still returning a signed infinity
        # when both groups are deterministic singletons.
        degrees = len(a) + len(b) - 2
        if degrees > 0:
            var_a = _variance(a, ddof=1) if len(a) >= 2 else 0.0
            var_b = _variance(b, ddof=1) if len(b) >= 2 else 0.0
            denominator = sqrt(
                ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / degrees
            )
        else:
            denominator = 0.0
    value = _standardise(difference, denominator)
    if normalised_kind in {"hedges_g", "g"}:
        value *= _hedges_correction(len(a) + len(b) - 2)
    return value


def _standardise(difference: float, denominator: float) -> float:
    if denominator > 0.0:
        return difference / denominator
    if difference == 0.0:
        return 0.0
    return inf if difference > 0.0 else -inf


def _hedges_correction(degrees_of_freedom: int) -> float:
    # First-order correction; no gamma-function dependency is needed for the
    # small synthetic studies this package targets.
    if degrees_of_freedom <= 1:
        return 1.0
    return 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)


cohens_d = effect_size


_SUCCESS_WORDS = {
    "success",
    "succeeded",
    "successful",
    "complete",
    "completed",
    "pass",
    "passed",
    "ok",
    "true",
    "yes",
    "supported",
    "accepted",
}
_FAILURE_WORDS = {
    "failure",
    "failed",
    "error",
    "cancelled",
    "canceled",
    "rejected",
    "false",
    "no",
    "inconclusive",
    "invalid",
}


def _success_value(value: Any, matcher: Any = None) -> bool | None:
    if value is None:
        return None
    if callable(matcher):
        return bool(matcher(value))
    if matcher is not None:
        return value == matcher
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        number = float(value)
        if not isfinite(number):
            return None
        return number > 0.0
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in _SUCCESS_WORDS or token in {"1", "1.0"}:
            return True
        if token in _FAILURE_WORDS or token in {"0", "0.0"}:
            return False
        # Do not silently interpret arbitrary labels as success/failure.
        return None
    return bool(value)


def success_rate(
    outcomes: Iterable[Any] | Number,
    success: Any = None,
    *,
    total: int | float | None = None,
    return_counts: bool = False,
) -> float | tuple[float, int, int]:
    """Return the proportion of successful outcomes.

    Outcomes may be booleans, 0/1 values, common queue/status strings, or any
    values matched by ``success`` (a target value or a predicate).  Unknown
    values are excluded from the observed denominator; this prevents a missing
    result from being mistaken for a failed scientific result.  Set ``total``
    to include reserved/unobserved slots in the denominator.

    For convenience, ``success_rate(successes, total)`` is also accepted when
    both positional arguments are scalar numbers (for example ``8, 10``).
    """

    if (
        total is None
        and _is_scalar(outcomes)
        and _is_scalar(success)
        and not isinstance(outcomes, bool)
        and not isinstance(success, bool)
    ):
        successes = float(outcomes)  # type: ignore[arg-type]
        denominator = float(success)  # type: ignore[arg-type]
        if denominator <= 0.0 or successes < 0.0 or successes > denominator:
            raise ValueError("success count must lie between zero and total")
        rate = successes / denominator
        return (rate, int(successes), int(denominator)) if return_counts else rate

    if isinstance(outcomes, (str, bytes)) or _is_scalar(outcomes):
        raw_outcomes: Iterable[Any] = [outcomes]
    else:
        raw_outcomes = outcomes  # type: ignore[assignment]

    successes_count = 0
    observed_count = 0
    for outcome in raw_outcomes:
        matched = _success_value(outcome, success)
        if matched is None:
            continue
        observed_count += 1
        successes_count += int(matched)

    if total is None:
        denominator = observed_count
    else:
        if not isinstance(total, Number) or float(total) < observed_count or float(total) < 0:
            raise ValueError("total must be at least the number of observed outcomes")
        denominator = int(total)
    rate = successes_count / denominator if denominator else 0.0
    if return_counts:
        return rate, successes_count, denominator
    return rate


@dataclass(slots=True)
class AnalysisSummary:
    """Serializable summary of one metric or comparison.

    The object contains only public, auditable statistics.  It deliberately
    does not contain hidden reasoning or generated prose.  ``values`` is kept
    when constructed through :meth:`from_values` so a report can optionally
    reproduce the calculation; callers may clear it before persisting if the
    raw observations are stored elsewhere.
    """

    n: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    effect_size: float | None = None
    success_rate: float | None = None
    confidence: float = 0.95
    values: tuple[float, ...] = field(default_factory=tuple, repr=False)
    successes: int | None = None
    observed_outcomes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    # Optional labels keep the summary convenient for grouped reports while
    # preserving the compact numeric core used by the engine.
    metric: str = "score"
    baseline_mean: float | None = None
    notes: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_values(
        cls,
        values: Iterable[NumberLike] | NumberLike,
        *,
        baseline: Iterable[NumberLike] | NumberLike | None = None,
        outcomes: Iterable[Any] | Number | None = None,
        success: Any = None,
        confidence: float = 0.95,
        method: str = "normal",
        ddof: int = 0,
        metric: str = "score",
        metadata: Mapping[str, Any] | None = None,
    ) -> "AnalysisSummary":
        # An empty result set is a legitimate state while a queue is waiting
        # for its first run.  Return an explicit ``n=0`` summary rather than
        # inventing a mean; the primitive ``mean``/``median`` functions still
        # raise on empty input so missing evidence cannot be mistaken for a
        # numeric observation.
        observations = _materialize(values)
        if not observations:
            measured_rate: float | None = None
            successes_count: int | None = None
            observed_count: int | None = None
            if outcomes is not None:
                measured_rate, successes_count, observed_count = success_rate(
                    outcomes, success=success, return_counts=True
                )
            return cls(
                n=0,
                mean=0.0,
                median=0.0,
                std=0.0,
                confidence_interval=(0.0, 0.0),
                effect_size=None,
                success_rate=measured_rate,
                confidence=float(confidence),
                values=(),
                successes=successes_count,
                observed_outcomes=observed_count,
                metadata=dict(metadata or {}),
                metric=str(metric),
                baseline_mean=None,
            )
        interval = confidence_interval(
            observations,
            confidence=confidence,
            method=method,
            # Keep the interval consistent with the requested std estimator.
            ddof=ddof,
        )
        baseline_mean: float | None = None
        if baseline is not None:
            # Materialise once so generator baselines remain reproducible and
            # both the effect size and the audit field describe identical
            # observations.
            baseline_values = _materialize(baseline)
            baseline_mean = mean(baseline_values) if baseline_values else None
            measured_effect = effect_size(observations, baseline_values) if baseline_values else None
        else:
            measured_effect = None
        measured_rate: float | None = None
        successes_count: int | None = None
        observed_count: int | None = None
        if outcomes is not None:
            measured_rate, successes_count, observed_count = success_rate(
                outcomes, success=success, return_counts=True
            )
        return cls(
            n=len(observations),
            mean=mean(observations),
            median=median(observations),
            std=std(observations, ddof=ddof),
            confidence_interval=interval,
            effect_size=measured_effect,
            success_rate=measured_rate,
            confidence=float(confidence),
            values=tuple(observations),
            successes=successes_count,
            observed_outcomes=observed_count,
            metadata=dict(metadata or {}),
            metric=str(metric),
            baseline_mean=baseline_mean,
        )

    @property
    def count(self) -> int:
        """Alias for ``n`` used by result-table serializers."""

        return self.n

    @property
    def sample_count(self) -> int:
        """Alias used by a few result-table and notebook adapters."""

        return self.n

    @property
    def std_dev(self) -> float:
        """Readable alias for ``std``."""

        return self.std

    @property
    def ci_lower(self) -> float:
        return self.confidence_interval[0]

    @property
    def ci_upper(self) -> float:
        return self.confidence_interval[1]

    @property
    def lower(self) -> float:
        return self.ci_lower

    @property
    def upper(self) -> float:
        return self.ci_upper

    @property
    def failures(self) -> int | None:
        if self.successes is None or self.observed_outcomes is None:
            return None
        return self.observed_outcomes - self.successes

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        """Return JSON-friendly public fields for logs and reports."""

        result: dict[str, Any] = {
            "n": self.n,
            "count": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "std_dev": self.std,
            "confidence_interval": [self.ci_lower, self.ci_upper],
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
            "effect_size": self.effect_size,
            "success_rate": self.success_rate,
            "successes": self.successes,
            "failures": self.failures,
            "observed_outcomes": self.observed_outcomes,
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "notes": list(self.notes),
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        if include_values:
            result["values"] = list(self.values)
        return result

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access in lightweight adapters."""

        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def summarize(
    values: Iterable[NumberLike] | NumberLike,
    **kwargs: Any,
) -> AnalysisSummary:
    """Convenience alias for :meth:`AnalysisSummary.from_values`."""

    return AnalysisSummary.from_values(values, **kwargs)


analyze = summarize

# Compatibility aliases used by the engine and lightweight adapters.  Keeping
# these names at module level avoids forcing callers to know whether they want
# the numeric primitive or the record-oriented ``ResultAnalyzer`` below.
analyze_values = summarize


def compare_groups(
    records: Any,
    metric: str = "score",
    group_field: str = "hypothesis_id",
    baseline_group: str | None = None,
) -> dict[str, Any]:
    """Top-level convenience wrapper around :class:`ResultAnalyzer`."""

    return ResultAnalyzer().compare_groups(
        records,
        metric=metric,
        group_field=group_field,
        baseline_group=baseline_group,
    )


# ---------------------------------------------------------------------------
# Record-oriented adapter
# ---------------------------------------------------------------------------
#
# ``AnalysisSummary`` above is the small numeric primitive used by providers.
# The UI/reporting code also needs to consume heterogeneous run records
# (ExperimentResult objects, dictionaries, or bare numbers).  Keeping this
# adapter here makes ``aiscientist.core.analysis`` useful on its own and keeps
# the public import stable for older adapters that used ``ResultAnalyzer``.


_MISSING = object()


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if value is None:
            continue
        if isinstance(value, Mapping) and name in value:
            return value[name]
        try:
            candidate = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if candidate is not None:
            return candidate
    return default


def _records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _record_rows(records: Any, metric: str = "score") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(_records(records)):
        if isinstance(record, Number) and not isinstance(record, bool):
            value = _as_finite_number(record)
            if value is not None:
                rows.append({"value": value, "index": index, "record": record})
            continue
        metrics = _field(record, "metrics", "measurements", "scores", default=None)
        value = _field(record, metric, "value", "metric_value", "score", default=_MISSING)
        if value is _MISSING and isinstance(metrics, Mapping):
            value = metrics.get(metric, _MISSING)
            if value is _MISSING and len(metrics) == 1:
                value = next(iter(metrics.values()))
        numeric = _as_finite_number(value)
        if numeric is None:
            continue
        rows.append(
            {
                "value": numeric,
                "index": index,
                "record": record,
                "run_id": _field(record, "run_id", "id", default=None),
                "experiment_id": _field(record, "experiment_id", "experiment", default=None),
                "hypothesis_id": _field(record, "hypothesis_id", "hypothesis", "group", default=None),
                "group": _field(record, "group", "hypothesis_id", "hypothesis", default=None),
                "success": _record_success(record),
                "status": _field(record, "status", "state", default=None),
            }
        )
    return rows


def _record_success(record: Any) -> bool | None:
    if isinstance(record, (bool, Number, str)):
        return _success_value(record)
    value = _field(record, "success", "passed", "is_success", "ok", default=_MISSING)
    if value is _MISSING:
        status = str(_field(record, "status", "state", default="")).strip().casefold()
        if status in _SUCCESS_WORDS:
            return True
        if status in _FAILURE_WORDS:
            return False
        return None
    return _success_value(value)


class ResultAnalyzer:
    """Analyze heterogeneous experiment records with auditable statistics.

    This is intentionally a thin record adapter around the pure functions in
    this module.  It returns ordinary dictionaries from :meth:`analyze` so
    existing report/CLI code can serialise the result directly.
    """

    def __init__(self, confidence_level: float = 0.95, *, method: str = "normal"):
        if not 0.0 < float(confidence_level) < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")
        self.confidence_level = float(confidence_level)
        self.method = method

    @staticmethod
    def confidence_interval(values: Iterable[NumberLike], level: float = 0.95) -> dict[str, Any]:
        observations = _materialize(values)
        if not observations:
            return {"level": float(level), "low": None, "high": None, "margin": None, "n": 0}
        low, high, margin = confidence_interval(
            observations,
            confidence=float(level),
            method="normal",
            ddof=1,
            return_margin=True,
        )
        return {
            "level": float(level),
            "low": low,
            "high": high,
            "margin": margin,
            "n": len(observations),
        }

    @staticmethod
    def effect_size(
        values: Iterable[NumberLike],
        baseline: Iterable[NumberLike] | NumberLike | None = None,
    ) -> float | None:
        if baseline is None:
            return None
        try:
            return effect_size(values, baseline)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def success_rate(records: Any) -> float | None:
        flags = [_record_success(record) for record in _records(records)]
        flags = [flag for flag in flags if flag is not None]
        return sum(flags) / len(flags) if flags else None

    def analyze(
        self,
        records: Any,
        metric: str = "score",
        baseline: Iterable[NumberLike] | NumberLike | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        source_records = _records(records)
        rows = _record_rows(source_records, metric)
        values = [row["value"] for row in rows]
        ci = self.confidence_interval(values, self.confidence_level)
        groups: dict[str, Any] = {}
        if group_by:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                raw = _field(row["record"], group_by, default=row.get(group_by))
                key = str(raw if raw is not None else "ungrouped")
                grouped.setdefault(key, []).append(row)
            for key, group_rows in grouped.items():
                groups[key] = self._summary_for_rows(group_rows, metric, baseline)
        result: dict[str, Any] = {
            "metric": metric,
            "n": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "std": std(values, sample=True) if len(values) > 1 else (0.0 if values else None),
            "confidence_interval": ci,
            "effect_size": self.effect_size(values, baseline),
            "success_rate": self.success_rate([row["record"] for row in rows]),
            "values": values,
            "groups": groups,
            "excluded": max(0, len(source_records) - len(rows)),
            "level": self.confidence_level,
        }
        result["ci_low"] = ci.get("low")
        result["ci_high"] = ci.get("high")
        return result

    def _summary_for_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
        baseline: Iterable[NumberLike] | NumberLike | None = None,
    ) -> dict[str, Any]:
        values = [float(row["value"]) for row in rows]
        ci = self.confidence_interval(values, self.confidence_level)
        return {
            "metric": metric,
            "n": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "std": std(values, sample=True) if len(values) > 1 else (0.0 if values else None),
            "confidence_interval": ci,
            "effect_size": self.effect_size(values, baseline),
            "success_rate": self.success_rate([row.get("record") for row in rows]),
            "values": values,
            "groups": {},
            "excluded": 0,
            "level": self.confidence_level,
            "ci_low": ci.get("low"),
            "ci_high": ci.get("high"),
        }

    def summarize(self, records: Any, metric: str = "score", **kwargs: Any) -> dict[str, Any]:
        return self.analyze(records, metric=metric, **kwargs)

    def compare_groups(
        self,
        records: Any,
        metric: str = "score",
        group_field: str = "hypothesis_id",
        baseline_group: str | None = None,
    ) -> dict[str, Any]:
        rows = _record_rows(records, metric)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            raw = _field(row["record"], group_field, default=row.get(group_field))
            key = str(raw if raw is not None else "ungrouped")
            grouped.setdefault(key, []).append(row)
        baseline_values = None
        if baseline_group is not None and baseline_group in grouped:
            baseline_values = [float(row["value"]) for row in grouped[baseline_group]]
        summaries = {
            key: self._summary_for_rows(
                group_rows,
                metric,
                baseline_values if key != baseline_group else None,
            )
            for key, group_rows in grouped.items()
        }
        pairwise: dict[str, float | None] = {}
        keys = list(grouped)
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                pairwise[f"{left}__vs__{right}"] = self.effect_size(
                    [row["value"] for row in grouped[left]],
                    [row["value"] for row in grouped[right]],
                )
        return {
            "metric": metric,
            "group_field": group_field,
            "baseline_group": baseline_group,
            "groups": summaries,
            "pairwise_effect_size": pairwise,
            "n": len(rows),
        }

    compare = compare_groups
    run = analyze
