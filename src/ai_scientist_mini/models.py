"""Domain models for AI Scientist Mini.

The project intentionally keeps its domain layer dependency free.  Models are
plain dataclasses and can be serialised to JSON using :func:`to_dict`; this is
useful both for the desktop UI and for the future CLI/API adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Mapping, TypeVar
import uuid


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing ``Z``."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(prefix: str) -> str:
    """Create a short, readable identifier for a domain object."""

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class StrEnum(str, Enum):
    """Enum whose JSON representation is its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class HypothesisStatus(StrEnum):
    UNTESTED = "Untested"
    SUPPORTED = "Supported"
    REJECTED = "Rejected"
    INCONCLUSIVE = "Inconclusive"


class QueueStatus(StrEnum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETE = "Complete"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class FailureType(StrEnum):
    INFRASTRUCTURE_FAILURE = "Infrastructure Failure"
    INVALID_DESIGN = "Invalid Design"
    INSUFFICIENT_DATA = "Insufficient Data"
    NEGATIVE_RESULT = "Negative Result"


class ApprovalMode(StrEnum):
    AUTONOMOUS_SANDBOX = "Autonomous Sandbox"
    HUMAN_APPROVAL = "Human Approval"


class ProjectStatus(StrEnum):
    DRAFT = "Draft"
    ACTIVE = "Active"
    PAUSED = "Paused"
    COMPLETE = "Complete"
    FAILED = "Failed"


class SelectionStrategy(StrEnum):
    RANDOM = "Random"
    BEST_EXPECTED_INFORMATION = "Best Expected Information"
    UNCERTAINTY_FIRST = "Uncertainty First"
    FOLLOW_UP_FAILED_RESULT = "Follow-up Failed Result"


@dataclass
class ResearchBudget:
    """Hard limits for a study.

    ``estimated_api_cost`` is a ceiling in the project's currency.  Synthetic
    providers always report zero cost, but retaining the accounting fields
    makes switching to a real provider explicit and safe.
    """

    maximum_experiments: int = 20
    maximum_runs: int = 100
    maximum_runtime_seconds: float = 3600.0
    estimated_api_cost: float = 0.0
    runs_used: int = 0
    experiments_used: int = 0
    runtime_seconds_used: float = 0.0
    actual_api_cost: float = 0.0

    def can_start_experiment(self) -> bool:
        return self.experiments_used < max(0, self.maximum_experiments)

    def can_run(self, count: int = 1) -> bool:
        return count >= 0 and self.runs_used + count <= max(0, self.maximum_runs)

    def can_spend(self, estimated_cost: float = 0.0, estimated_runtime: float = 0.0) -> bool:
        return (
            self.actual_api_cost + max(0.0, estimated_cost) <= max(0.0, self.estimated_api_cost)
            and self.runtime_seconds_used + max(0.0, estimated_runtime)
            <= max(0.0, self.maximum_runtime_seconds)
        )

    def reserve_experiment(self, runs: int = 0) -> None:
        if not self.can_start_experiment():
            raise BudgetExceeded("Maximum experiments reached")
        if not self.can_run(runs):
            raise BudgetExceeded("Maximum runs reached")
        self.experiments_used += 1
        self.runs_used += max(0, runs)

    def consume_run(self, runtime_seconds: float = 0.0, api_cost: float = 0.0) -> None:
        # ``runs_used`` is incremented here only when a run was not reserved.
        self.runs_used += 1
        self.runtime_seconds_used += max(0.0, runtime_seconds)
        self.actual_api_cost += max(0.0, api_cost)
        if self.runs_used > self.maximum_runs:
            raise BudgetExceeded("Maximum runs reached")
        if self.runtime_seconds_used > self.maximum_runtime_seconds + 1e-9:
            raise BudgetExceeded("Maximum runtime reached")
        if self.actual_api_cost > self.estimated_api_cost + 1e-9:
            raise BudgetExceeded("Estimated API cost exceeded")

    @property
    def remaining_experiments(self) -> int:
        return max(0, self.maximum_experiments - self.experiments_used)

    @property
    def remaining_runs(self) -> int:
        return max(0, self.maximum_runs - self.runs_used)


class BudgetExceeded(RuntimeError):
    """Raised before an operation would violate a research budget."""


@dataclass
class Hypothesis:
    id: str = field(default_factory=lambda: make_id("hyp"))
    statement: str = ""
    rationale: str = ""
    predictions: list[str] = field(default_factory=list)
    test_method: str = ""
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    evidence: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
        if not isinstance(self.status, HypothesisStatus):
            self.status = HypothesisStatus(str(self.status))


@dataclass
class ExperimentDesign:
    id: str = field(default_factory=lambda: make_id("exp"))
    hypothesis_id: str = ""
    name: str = ""
    independent_variable: str = ""
    dependent_variable: str = ""
    control: str = ""
    dataset: str = "synthetic"
    sample_size: int = 5
    seed: int = 0
    metric: str = "score"
    procedure: list[str] = field(default_factory=list)
    success_criteria: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.sample_size = max(1, int(self.sample_size))
        self.seed = int(self.seed)


@dataclass
class AnalysisSummary:
    metric: str = "score"
    values: list[float] = field(default_factory=list)
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    confidence_interval: tuple[float, float] | None = None
    effect_size: float | None = None
    success_rate: float | None = None
    sample_count: int = 0
    baseline_mean: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ExperimentRun:
    id: str = field(default_factory=lambda: make_id("run"))
    experiment_id: str = ""
    run_index: int = 0
    seed: int = 0
    status: QueueStatus = QueueStatus.QUEUED
    metrics: dict[str, float] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    failure_type: FailureType | None = None
    started_at: str | None = None
    finished_at: str | None = None
    runtime_seconds: float = 0.0
    api_cost: float = 0.0
    environment: dict[str, str] = field(default_factory=dict)
    code_version: str = "dev"
    data_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, QueueStatus):
            self.status = QueueStatus(str(self.status))
        if self.failure_type is not None and not isinstance(self.failure_type, FailureType):
            self.failure_type = FailureType(str(self.failure_type))


@dataclass
class Experiment:
    design: ExperimentDesign
    id: str = ""
    status: QueueStatus = QueueStatus.QUEUED
    runs: list[ExperimentRun] = field(default_factory=list)
    analysis: AnalysisSummary | None = None
    failure_type: FailureType | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.design.id
        if not isinstance(self.status, QueueStatus):
            self.status = QueueStatus(str(self.status))
        if self.failure_type is not None and not isinstance(self.failure_type, FailureType):
            self.failure_type = FailureType(str(self.failure_type))


@dataclass
class Evidence:
    id: str = field(default_factory=lambda: make_id("evidence"))
    hypothesis_id: str = ""
    experiment_id: str = ""
    run_ids: list[str] = field(default_factory=list)
    claim: str = ""
    strength: float = 0.0
    rule: str = ""
    # Classification is about the observed outcome, not infrastructure
    # health.  In particular, ``negative_result`` must not be confused with a
    # failed experiment or an unavailable provider.
    classification: str = "evidence"
    created_at: str = field(default_factory=utc_now)


@dataclass
class JournalEntry:
    id: str = field(default_factory=lambda: make_id("journal"))
    event_type: str = "observation"
    summary: str = ""
    why: str = ""
    experiment_id: str | None = None
    hypothesis_id: str | None = None
    public: bool = True
    timestamp: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    id: str = field(default_factory=lambda: make_id("audit"))
    event_type: str = ""
    subject_id: str = ""
    rule: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)


@dataclass
class ResearchMemory:
    previous_experiments: list[str] = field(default_factory=list)
    failed_ideas: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


@dataclass
class ResearchProject:
    research_question: str
    name: str = "AI Scientist Study"
    id: str = field(default_factory=lambda: make_id("project"))
    scope: str = ""
    constraints: list[str] = field(default_factory=list)
    available_resources: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: ["score"])
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    max_experiments: int = 20
    status: ProjectStatus = ProjectStatus.DRAFT
    approval_mode: ApprovalMode = ApprovalMode.AUTONOMOUS_SANDBOX
    hypotheses: list[Hypothesis] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    journal: list[JournalEntry] = field(default_factory=list)
    audit_trail: list[AuditEvent] = field(default_factory=list)
    memory: ResearchMemory = field(default_factory=ResearchMemory)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.max_experiments = max(0, int(self.max_experiments))
        self.budget.maximum_experiments = min(self.budget.maximum_experiments, self.max_experiments)
        if not isinstance(self.status, ProjectStatus):
            self.status = ProjectStatus(str(self.status))
        if not isinstance(self.approval_mode, ApprovalMode):
            self.approval_mode = ApprovalMode(str(self.approval_mode))

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)

    def experiment(self, experiment_id: str) -> Experiment | None:
        return next((e for e in self.experiments if e.id == experiment_id), None)


# -- serialisation ---------------------------------------------------------

T = TypeVar("T")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    return value


def to_dict(value: Any) -> Any:
    """Convert a model (or nested collection) into JSON-compatible values."""

    return _json_value(value)


def _enum(enum_cls: type[Enum], value: Any) -> Any:
    if value is None:
        return None
    try:
        return enum_cls(value)
    except (TypeError, ValueError):
        return value


def project_from_dict(data: Mapping[str, Any]) -> ResearchProject:
    """Rehydrate a :class:`ResearchProject` from :func:`to_dict` output."""

    raw = dict(data)
    budget = ResearchBudget(**raw.pop("budget", {}))
    hypotheses = []
    for item in raw.pop("hypotheses", []):
        item = dict(item)
        item["status"] = _enum(HypothesisStatus, item.get("status", HypothesisStatus.UNTESTED))
        hypotheses.append(Hypothesis(**item))
    experiments = []
    for item in raw.pop("experiments", []):
        item = dict(item)
        design_data = dict(item.pop("design", {}))
        design = ExperimentDesign(**design_data)
        item["design"] = design
        item["status"] = _enum(QueueStatus, item.get("status", QueueStatus.QUEUED))
        item["failure_type"] = _enum(FailureType, item.get("failure_type"))
        runs = []
        for run_data in item.get("runs", []):
            run_data = dict(run_data)
            run_data["status"] = _enum(QueueStatus, run_data.get("status", QueueStatus.QUEUED))
            run_data["failure_type"] = _enum(FailureType, run_data.get("failure_type"))
            runs.append(ExperimentRun(**run_data))
        item["runs"] = runs
        if item.get("analysis") is not None:
            analysis = dict(item["analysis"])
            if analysis.get("confidence_interval") is not None:
                analysis["confidence_interval"] = tuple(analysis["confidence_interval"])
            item["analysis"] = AnalysisSummary(**analysis)
        experiments.append(Experiment(**item))
    evidence = [Evidence(**dict(item)) for item in raw.pop("evidence", [])]
    journal = [JournalEntry(**dict(item)) for item in raw.pop("journal", [])]
    audit = [AuditEvent(**dict(item)) for item in raw.pop("audit_trail", [])]
    memory = ResearchMemory(**raw.pop("memory", {}))
    raw["budget"] = budget
    raw["hypotheses"] = hypotheses
    raw["experiments"] = experiments
    raw["evidence"] = evidence
    raw["journal"] = journal
    raw["audit_trail"] = audit
    raw["memory"] = memory
    raw["status"] = _enum(ProjectStatus, raw.get("status", ProjectStatus.DRAFT))
    raw["approval_mode"] = _enum(ApprovalMode, raw.get("approval_mode", ApprovalMode.AUTONOMOUS_SANDBOX))
    return ResearchProject(**raw)
