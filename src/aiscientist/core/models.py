"""Domain models for :mod:`AI Scientist Mini`.

The core package deliberately keeps the models serialisable and provider agnostic.
Every object can be written to JSON (or to :class:`ResearchStore`) and restored
without importing a UI module.  This makes experiments reproducible and gives
the audit trail a stable schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
import platform
import sys
import uuid
from typing import Any, ClassVar, Mapping, TypeVar


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with a ``Z`` suffix."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}{value}" if prefix else value


class StrEnum(str, Enum):
    """Enum that serialises naturally as a string."""

    def __str__(self) -> str:  # pragma: no cover - useful for logs and CLI
        return self.value


class ProjectStatus(StrEnum):
    PLANNING = "Planning"
    RUNNING = "Running"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class HypothesisStatus(StrEnum):
    UNTESTED = "Untested"
    SUPPORTED = "Supported"
    REJECTED = "Rejected"
    INCONCLUSIVE = "Inconclusive"


class ExperimentStatus(StrEnum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETE = "Complete"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class FailureType(StrEnum):
    INFRASTRUCTURE = "Infrastructure Failure"
    INVALID_DESIGN = "Invalid Design"
    INSUFFICIENT_DATA = "Insufficient Data"
    NEGATIVE_RESULT = "Negative Result"


class SelectionStrategy(StrEnum):
    RANDOM = "Random"
    BEST_EXPECTED_INFORMATION = "Best Expected Information"
    UNCERTAINTY_FIRST = "Uncertainty First"
    FOLLOW_UP_FAILED_RESULT = "Follow-up Failed Result"


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses/enums into JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


T = TypeVar("T")


class Serializable:
    """Mixin supplying deterministic ``to_dict``/``to_json`` helpers."""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


@dataclass
class Budget(Serializable):
    """Hard limits for a study.

    ``estimated_api_cost`` is the configured API-cost allowance for the study.
    Built-in providers report zero cost, so the development default is safe.
    Set a positive allowance explicitly before enabling an external adapter.
    Counters are persisted so restarting the application cannot silently reset
    a safety limit.
    """

    max_experiments: int = 20
    max_runs: int = 100
    max_runtime_seconds: float = 3600.0
    estimated_api_cost: float = 0.0
    spent_experiments: int = 0
    spent_runs: int = 0
    runtime_seconds: float = 0.0
    spent_api_cost: float = 0.0

    # User-facing aliases used by some adapters.
    @property
    def maximum_experiments(self) -> int:
        return self.max_experiments

    @property
    def maximum_runs(self) -> int:
        return self.max_runs

    @property
    def maximum_runtime(self) -> float:
        return self.max_runtime_seconds

    @property
    def remaining_experiments(self) -> int:
        return max(0, self.max_experiments - self.spent_experiments)

    @property
    def remaining_runs(self) -> int:
        return max(0, self.max_runs - self.spent_runs)

    @property
    def remaining_runtime_seconds(self) -> float:
        return max(0.0, float(self.max_runtime_seconds) - float(self.runtime_seconds))

    @property
    def remaining_api_cost(self) -> float:
        return max(0.0, float(self.estimated_api_cost) - float(self.spent_api_cost))

    def can_schedule(
        self,
        *,
        experiments: int = 1,
        runs: int = 1,
        runtime_seconds: float = 0.0,
        cost: float = 0.0,
    ) -> bool:
        """Return whether a proposed reservation stays within every limit."""

        runtime_seconds = max(0.0, float(runtime_seconds))
        cost = max(0.0, float(cost))
        return (
            experiments >= 0
            and runs >= 0
            and self.spent_experiments + experiments <= self.max_experiments
            and self.spent_runs + runs <= self.max_runs
            and self.runtime_seconds + runtime_seconds <= self.max_runtime_seconds
            and self.spent_api_cost + cost <= self.estimated_api_cost
        )

    def consume(self, *, experiments: int = 0, runs: int = 0, runtime_seconds: float = 0.0, cost: float = 0.0) -> None:
        if not self.can_schedule(experiments=experiments, runs=runs, runtime_seconds=runtime_seconds, cost=cost):
            raise BudgetError("Research budget exhausted")
        self.spent_experiments += experiments
        self.spent_runs += runs
        self.runtime_seconds += max(0.0, runtime_seconds)
        self.spent_api_cost += max(0.0, cost)


class BudgetError(RuntimeError):
    """Raised before an operation would exceed a configured budget."""


@dataclass
class Project(Serializable):
    research_question: str
    scope: str = ""
    constraints: list[str] = field(default_factory=list)
    available_resources: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: ["score"])
    budget: Budget = field(default_factory=Budget)
    max_experiments: int | None = None
    status: ProjectStatus = ProjectStatus.PLANNING
    id: str = field(default_factory=lambda: new_id("project_"))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ProjectStatus(self.status)
        if isinstance(self.budget, dict):
            self.budget = Budget.from_dict(self.budget)
        if isinstance(self.metrics, str):
            self.metrics = [item.strip() for item in self.metrics.split(",") if item.strip()] or ["score"]
        if isinstance(self.constraints, str):
            self.constraints = [self.constraints]
        if self.max_experiments is not None:
            self.budget.max_experiments = int(self.max_experiments)
        else:
            self.max_experiments = self.budget.max_experiments

    def touch(self) -> None:
        self.updated_at = utc_now()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Project":
        payload = dict(data)
        payload["budget"] = Budget.from_dict(payload.get("budget", {}))
        return cls(**payload)


@dataclass
class Hypothesis(Serializable):
    statement: str
    rationale: str = ""
    predictions: list[Any] | dict[str, Any] = field(default_factory=list)
    test_method: str = ""
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    project_id: str | None = None
    id: str = field(default_factory=lambda: new_id("hyp_"))
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = HypothesisStatus(self.status)
        self.confidence = min(1.0, max(0.0, float(self.confidence)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Hypothesis":
        return cls(**dict(data))


@dataclass
class ExperimentConfig(Serializable):
    independent_variable: str | dict[str, Any]
    dependent_variable: str = "score"
    control: Any = None
    dataset: str = "synthetic"
    sample_size: int = 20
    seed: int = 0
    metric: str = "score"
    procedure: list[str] | str = field(default_factory=list)
    success_criteria: dict[str, Any] | str = field(default_factory=dict)
    hypothesis_id: str | None = None
    project_id: str | None = None
    name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    provider: str = "synthetic"
    runs_per_config: int = 1
    id: str = field(default_factory=lambda: new_id("exp_"))
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_size = max(1, int(self.sample_size))
        self.runs_per_config = max(1, int(self.runs_per_config))
        if not self.name:
            self.name = str(self.independent_variable)

    def canonical_payload(self, *, include_seed: bool = True) -> dict[str, Any]:
        payload = self.to_dict()
        # IDs/timestamps don't describe the experimental treatment.
        for key in ("id", "created_at"):
            payload.pop(key, None)
        if not include_seed:
            payload.pop("seed", None)
        return payload

    def config_hash(self, *, include_seed: bool = True) -> str:
        raw = json.dumps(self.canonical_payload(include_seed=include_seed), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(**dict(data))


@dataclass
class ExperimentRun(Serializable):
    experiment_id: str
    run_index: int = 0
    seed: int = 0
    status: ExperimentStatus = ExperimentStatus.QUEUED
    id: str = field(default_factory=lambda: new_id("run_"))
    result_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    failure_type: FailureType | None = None
    logs: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    code_version: str = ""
    data_hash: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ExperimentStatus(self.status)
        if isinstance(self.failure_type, str):
            try:
                self.failure_type = FailureType(self.failure_type)
            except ValueError:
                self.failure_type = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRun":
        return cls(**dict(data))


@dataclass
class ExperimentResult(Serializable):
    run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    success: bool = True
    id: str = field(default_factory=lambda: new_id("result_"))
    raw_output: Any = None
    failure_type: FailureType | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    data_hash: str = ""
    # Denormalised linkage makes report graphs/audit queries possible without
    # loading the run row first.  It is placed last so older positional calls
    # (run_id, metrics, observations, ...) remain valid.
    experiment_id: str | None = None

    def __post_init__(self) -> None:
        self.metrics = {str(k): float(v) for k, v in self.metrics.items() if _is_number(v)}
        if isinstance(self.failure_type, str):
            try:
                self.failure_type = FailureType(self.failure_type)
            except ValueError:
                self.failure_type = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentResult":
        return cls(**dict(data))


@dataclass
class QueueItem(Serializable):
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.QUEUED
    priority: int = 0
    enqueued_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ExperimentStatus(self.status)


@dataclass
class AuditEvent(Serializable):
    event_type: str
    message: str
    entity_type: str = ""
    entity_id: str = ""
    rule: str = ""
    actor: str = "RuleBasedScientist"
    details: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("audit_"))
    timestamp: str = field(default_factory=utc_now)


@dataclass
class JournalEntry(Serializable):
    why: str
    configuration: dict[str, Any] = field(default_factory=dict)
    what_happened: str = ""
    result_summary: str = ""
    next_step: str = ""
    experiment_id: str | None = None
    id: str = field(default_factory=lambda: new_id("journal_"))
    timestamp: str = field(default_factory=utc_now)


@dataclass
class MemoryRecord(Serializable):
    category: str
    content: str
    references: list[str] = field(default_factory=list)
    importance: float = 0.5
    id: str = field(default_factory=lambda: new_id("mem_"))
    created_at: str = field(default_factory=utc_now)


@dataclass
class ProviderResult(Serializable):
    """Provider-neutral output before it is attached to a run."""

    metrics: dict[str, float] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    raw_output: Any = None
    success: bool = True
    error: str | None = None
    failure_type: FailureType | None = None
    runtime_seconds: float = 0.0
    estimated_cost: float = 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def environment_snapshot() -> dict[str, Any]:
    """Small, non-sensitive reproducibility snapshot."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": os.path.basename(sys.executable),
    }


# ``Budget.from_dict`` and friends are intentionally assigned after the class
# definitions to keep the dataclasses readable while accepting old JSON files.
def _budget_from_dict(cls: type[Budget], data: Mapping[str, Any]) -> Budget:
    aliases = {
        "max_runtime": "max_runtime_seconds",
        "maximum_runtime": "max_runtime_seconds",
        "max_api_cost": "estimated_api_cost",
        "cost_limit": "estimated_api_cost",
    }
    normalized = dict(data)
    for old, new in aliases.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized[old]
    allowed = {f.name for f in fields(Budget)}
    return cls(**{k: v for k, v in normalized.items() if k in allowed})


Budget.from_dict = classmethod(_budget_from_dict)  # type: ignore[attr-defined]


__all__ = [
    "AuditEvent",
    "Budget",
    "BudgetError",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRun",
    "ExperimentStatus",
    "FailureType",
    "Hypothesis",
    "HypothesisStatus",
    "JournalEntry",
    "MemoryRecord",
    "Project",
    "ProjectStatus",
    "ProviderResult",
    "QueueItem",
    "SelectionStrategy",
    "environment_snapshot",
    "new_id",
    "utc_now",
]
