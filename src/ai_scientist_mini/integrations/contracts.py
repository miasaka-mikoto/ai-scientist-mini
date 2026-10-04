"""Versioned, dependency-free contracts for experiment integrations.

These contracts are intentionally ordinary dataclasses.  They are useful to
the built-in engine, but are also easy for another language or process to
produce as JSON.  No contract contains model chain-of-thought; only public
decision summaries and reproducibility metadata are exchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, MutableMapping, Protocol, Sequence


API_VERSION = "1.0"


def utc_now_iso() -> str:
    """Return a stable, timezone-aware timestamp for an audit record."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ProviderCapabilities:
    """What an experiment provider can do.

    ``external`` is informational and never authorizes network or paid API
    usage.  The host application remains responsible for budget and approval
    checks before invoking an external adapter.
    """

    provider_type: str = "experiment"
    supports_repeated_runs: bool = True
    supports_cancellation: bool = False
    supports_streaming: bool = False
    supports_cost_estimate: bool = True
    external: bool = False
    capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ProviderCapabilities":
        if not value:
            return cls()
        allowed = {"provider_type", "supports_repeated_runs", "supports_cancellation",
                   "supports_streaming", "supports_cost_estimate", "external", "capabilities"}
        data = {k: v for k, v in value.items() if k in allowed}
        if "capabilities" in data:
            data["capabilities"] = tuple(str(x) for x in (data["capabilities"] or ()))
        return cls(**data)


@dataclass(frozen=True)
class AdapterDescriptor:
    """Public metadata used to discover an adapter without loading it."""

    adapter_id: str
    display_name: str
    version: str = API_VERSION
    transport: str = "local"
    provider_type: str = "experiment"
    description: str = ""
    external: bool = False
    requires_approval: bool = False
    command: tuple[str, ...] = ()
    endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command"] = list(self.command)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdapterDescriptor":
        data = dict(value)
        if "command" in data:
            data["command"] = tuple(str(x) for x in (data["command"] or ()))
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class ExperimentRequest:
    """A provider-neutral request for one experiment configuration.

    ``config`` is deliberately opaque JSON.  The Scientist engine owns the
    semantics; adapters only transport and execute it.
    """

    experiment_id: str
    research_question: str = ""
    hypothesis_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    runs: int = 1
    dataset: dict[str, Any] | list[Any] | str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    requested_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentRequest":
        data = dict(value)
        defaults = cls(experiment_id=str(data.pop("experiment_id", "experiment")))
        for name in ("research_question", "hypothesis_id", "config", "seed", "runs",
                     "dataset", "constraints", "requested_at", "metadata"):
            if name in data:
                setattr(defaults, name, data[name])
        defaults.config = dict(defaults.config or {})
        defaults.constraints = dict(defaults.constraints or {})
        defaults.metadata = dict(defaults.metadata or {})
        defaults.seed = int(defaults.seed)
        defaults.runs = max(1, int(defaults.runs))
        return defaults


@dataclass
class RunResult:
    """Public output for one run; errors are data, not raised conclusions."""

    run_id: str
    status: str = "complete"  # complete | failed | cancelled
    metrics: dict[str, float] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    duration_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    completed_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunResult":
        data = dict(value)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in allowed}
        if "metrics" in data:
            data["metrics"] = {str(k): float(v) for k, v in (data["metrics"] or {}).items()}
        return cls(**data)


@dataclass
class ExperimentResult:
    """Aggregate result returned by a provider or adapter."""

    experiment_id: str
    provider_id: str
    status: str = "complete"
    runs: list[RunResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    public_summary: str = ""
    error_type: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runs"] = [run.to_dict() for run in self.runs]
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentResult":
        data = dict(value)
        data["runs"] = [RunResult.from_dict(item) for item in (data.get("runs") or [])]
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in allowed}
        data["aggregate_metrics"] = {
            str(k): float(v) for k, v in (data.get("aggregate_metrics") or {}).items()
        }
        data["reproducibility"] = dict(data.get("reproducibility") or {})
        data["metadata"] = dict(data.get("metadata") or {})
        return cls(**data)


class ExperimentProvider(Protocol):
    """Minimal provider interface implemented by local or external engines.

    Providers must not make hidden budget decisions.  The host checks the
    request against its budget before calling ``run``.  A provider may return
    a failed :class:`ExperimentResult` for an experiment problem; transport
    errors should be represented with ``status='failed'`` where possible.
    """

    provider_id: str
    capabilities: ProviderCapabilities

    def run(self, request: ExperimentRequest, *, cancel: Callable[[], bool] | None = None) -> ExperimentResult:
        ...

    def estimate_cost(self, request: ExperimentRequest) -> float:
        ...

    def health(self) -> Mapping[str, Any]:
        ...


def provider_to_descriptor(provider: Any, *, adapter_id: str | None = None) -> AdapterDescriptor:
    """Build a descriptor from a duck-typed provider without importing it."""

    pid = str(adapter_id or getattr(provider, "provider_id", provider.__class__.__name__))
    caps = getattr(provider, "capabilities", ProviderCapabilities())
    if isinstance(caps, Mapping):
        caps = ProviderCapabilities.from_dict(caps)
    return AdapterDescriptor(
        adapter_id=pid,
        display_name=str(getattr(provider, "display_name", pid)),
        version=str(getattr(provider, "version", API_VERSION)),
        transport="local",
        provider_type=getattr(caps, "provider_type", "experiment"),
        description=str(getattr(provider, "description", "")),
        external=bool(getattr(caps, "external", False)),
    )

