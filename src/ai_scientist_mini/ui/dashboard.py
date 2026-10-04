"""Tkinter dashboard for AI Scientist Mini.

The research engine is intentionally not imported at module import time.  A
dashboard can therefore be used with the real engine, a test double, or the
small deterministic :class:`LocalDemoService` included here.  The service
boundary is duck typed: an application may provide any object implementing
one or more of ``snapshot``, ``start_demo``, ``create_study``,
``run_next``, ``run_all``, ``stop`` and ``generate_report``.

No model, network, or paid API is used by this module.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
import statistics
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Protocol, Sequence

try:  # Tkinter is in the Python standard library, but may be absent on Linux CI.
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except Exception:  # pragma: no cover - import is intentionally tolerant on headless systems.
    tk = None  # type: ignore[assignment]
    filedialog = messagebox = simpledialog = ttk = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Small, toolkit-independent data contracts
# ---------------------------------------------------------------------------


@dataclass
class DashboardSnapshot:
    """Serializable state consumed by the dashboard.

    The core engine is free to return a dictionary or its own dataclass.  This
    canonical representation makes the view deterministic and also provides
    a useful integration contract for future CLI/API adapters.
    """

    question: str = ""
    status: str = "Idle"
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    best_hypothesis: dict[str, Any] | None = None
    next_experiment: dict[str, Any] | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    journal: list[dict[str, Any] | str] = field(default_factory=list)
    report_path: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_any(cls, value: Any) -> "DashboardSnapshot":
        """Coerce dictionaries/objects returned by a core service.

        Unknown fields are ignored.  A service can expose a ``snapshot``
        property, a ``get_snapshot`` method, or simply return a mapping.
        """

        if value is None:
            return cls()
        if isinstance(value, cls):
            return copy.deepcopy(value)
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                value = value.to_dict()
            except Exception:
                pass
        if not isinstance(value, Mapping):
            # A few engines expose attributes rather than a mapping.
            value = {
                key: getattr(value, key, None)
                for key in (
                    "question",
                    "status",
                    "hypotheses",
                    "experiments",
                    "evidence",
                    "best_hypothesis",
                    "next_experiment",
                    "budget",
                    "graph",
                    "journal",
                    "report_path",
                    "message",
                )
            }

        def seq(name: str) -> list[Any]:
            raw = value.get(name, [])
            if raw is None:
                return []
            if isinstance(raw, (str, bytes)):
                return [raw]
            try:
                return list(raw)
            except TypeError:
                return [raw]

        def mapping(name: str) -> dict[str, Any]:
            raw = value.get(name, {})
            if raw is None:
                return {}
            if isinstance(raw, Mapping):
                return dict(raw)
            if dataclasses.is_dataclass(raw):
                return dataclasses.asdict(raw)
            if hasattr(raw, "__dict__"):
                return dict(vars(raw))
            return {"value": raw}

        def optional_mapping(name: str) -> dict[str, Any] | None:
            raw = value.get(name)
            if raw is None:
                return None
            if isinstance(raw, Mapping):
                return dict(raw)
            if dataclasses.is_dataclass(raw):
                return dataclasses.asdict(raw)
            if hasattr(raw, "__dict__"):
                return dict(vars(raw))
            return {"value": raw}

        return cls(
            question=str(value.get("question", value.get("research_question", "")) or ""),
            status=str(value.get("status", "Idle") or "Idle"),
            hypotheses=[_row_to_dict(item) for item in seq("hypotheses")],
            experiments=[_row_to_dict(item) for item in seq("experiments")],
            evidence=[_row_to_dict(item) for item in seq("evidence")],
            best_hypothesis=optional_mapping("best_hypothesis"),
            next_experiment=optional_mapping("next_experiment"),
            budget=mapping("budget"),
            graph=mapping("graph"),
            journal=[_row_to_dict(item) if not isinstance(item, str) else item for item in seq("journal")],
            report_path=str(value.get("report_path", "") or ""),
            message=str(value.get("message", "") or ""),
        )


def _row_to_dict(value: Any) -> dict[str, Any]:
    """Convert one arbitrary row to a display-friendly dictionary."""

    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            result = value.to_dict()
            if isinstance(result, Mapping):
                return dict(result)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class DashboardViewModel:
    """Stable, testable projection of a :class:`DashboardSnapshot`."""

    question: str
    status: str
    hypotheses: list[dict[str, Any]]
    experiments: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    best_hypothesis: dict[str, Any] | None
    next_experiment: dict[str, Any] | None
    budget: dict[str, Any]
    journal: list[dict[str, Any] | str]
    report_path: str
    message: str

    @classmethod
    def from_snapshot(cls, value: Any) -> "DashboardViewModel":
        snapshot = DashboardSnapshot.from_any(value)
        return cls(
            question=snapshot.question,
            status=snapshot.status,
            hypotheses=snapshot.hypotheses,
            experiments=snapshot.experiments,
            evidence=snapshot.evidence,
            best_hypothesis=snapshot.best_hypothesis,
            next_experiment=snapshot.next_experiment,
            budget=snapshot.budget,
            journal=snapshot.journal,
            report_path=snapshot.report_path,
            message=snapshot.message,
        )

    @property
    def confidence(self) -> float | None:
        item = self.best_hypothesis or max(
            self.hypotheses,
            key=lambda row: _as_float(row.get("confidence", row.get("score")), -1.0),
            default={},
        )
        raw = item.get("confidence", item.get("score"))
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def budget_text(self) -> str:
        if not self.budget:
            return "No budget configured"
        labels = (
            ("Experiments", "experiments", "max_experiments"),
            ("Runs", "runs", "max_runs"),
            ("Runtime", "runtime_seconds", "max_runtime_seconds"),
            ("Cost", "estimated_api_cost", "budget_cost"),
        )
        parts: list[str] = []
        for label, used_key, limit_key in labels:
            used = self.budget.get(used_key, self.budget.get(f"used_{used_key}"))
            limit = self.budget.get(limit_key)
            if used is None and limit is None:
                continue
            if used is not None and limit is not None:
                parts.append(f"{label}: {used}/{limit}")
            elif limit is not None:
                parts.append(f"{label}: 0/{limit}")
            else:
                parts.append(f"{label}: {used}")
        return "  ·  ".join(parts) if parts else json.dumps(self.budget, ensure_ascii=False)

    @property
    def best_text(self) -> str:
        item = self.best_hypothesis or max(
            self.hypotheses,
            key=lambda row: _as_float(row.get("confidence", row.get("score")), -1.0),
            default=None,
        )
        if not item:
            return "—"
        statement = item.get("statement", item.get("name", ""))
        confidence = item.get("confidence")
        if confidence is None:
            return str(statement or self.best_hypothesis.get("id", "—"))
        try:
            confidence = float(confidence)
            if confidence <= 1:
                confidence = confidence * 100
            return f"{statement} ({confidence:.1f}%)"
        except (TypeError, ValueError):
            return f"{statement} ({confidence})"


class StudyService(Protocol):
    """Optional protocol documented for core-engine adapters."""

    def snapshot(self) -> Any: ...


class StudyServiceAdapter:
    """Adapt several likely core-engine method names to one UI contract.

    This class intentionally uses duck typing instead of importing the core
    engine.  It lets the UI land before the engine API is frozen and keeps the
    eventual integration boundary small and explicit.
    """

    def __init__(self, service: Any | None = None):
        self.service = service

    def _call(self, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        if self.service is None:
            return None
        for name in names:
            method = getattr(self.service, name, None)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    # Core implementations may not accept all optional UI args.
                    if kwargs:
                        try:
                            return method(*args)
                        except TypeError:
                            continue
                    continue
        return None

    def get_snapshot(self) -> DashboardSnapshot:
        value = self._call(("snapshot", "get_snapshot", "dashboard_snapshot", "state"))
        if value is None and self.service is not None:
            value = getattr(self.service, "snapshot", None)
            if callable(value):
                value = None
        # The first-party ``ScientistEngine`` intentionally exposes a domain
        # project rather than a UI-specific snapshot.  Keeping this conversion
        # here means the engine remains GUI-free while the desktop can still be
        # launched directly with ``Dashboard(root, service=engine)``.
        if value is None and self.service is not None and getattr(self.service, "project", None) is not None:
            value = self._engine_project_snapshot(getattr(self.service, "project"))
        return DashboardSnapshot.from_any(value)

    def _engine_project_snapshot(self, project: Any) -> dict[str, Any]:
        """Project-to-dashboard projection for the built-in engine models."""

        def val(item: Any, name: str, default: Any = None) -> Any:
            raw = getattr(item, name, default)
            # StrEnum's __str__ is intentionally human-readable, while plain
            # Enum values need unwrapping for JSON and Treeview cells.
            return getattr(raw, "value", raw)

        hypotheses: list[dict[str, Any]] = []
        for h in list(getattr(project, "hypotheses", []) or []):
            hypotheses.append(
                {
                    "id": val(h, "id", ""),
                    "statement": val(h, "statement", ""),
                    "rationale": val(h, "rationale", ""),
                    "confidence": val(h, "confidence", 0.5),
                    "status": val(h, "status", "Untested"),
                    "evidence": list(val(h, "evidence", []) or []),
                    "metadata": dict(val(h, "metadata", {}) or {}),
                }
            )
        experiments: list[dict[str, Any]] = []
        for exp in list(getattr(project, "experiments", []) or []):
            design = getattr(exp, "design", None)
            analysis = getattr(exp, "analysis", None)
            cfg = dict(val(design, "config", {}) or {}) if design is not None else {}
            result = {}
            if analysis is not None:
                result = {
                    "mean": val(analysis, "mean"),
                    "median": val(analysis, "median"),
                    "std": val(analysis, "std"),
                    "confidence_interval": val(analysis, "confidence_interval"),
                    "effect_size": val(analysis, "effect_size"),
                    "success_rate": val(analysis, "success_rate"),
                    "sample_count": val(analysis, "sample_count", 0),
                }
            experiments.append(
                {
                    "id": val(exp, "id", val(design, "id", "") if design else ""),
                    "hypothesis_id": val(design, "hypothesis_id", "") if design else "",
                    "strategy": cfg.get("strategy", val(design, "name", "") if design else ""),
                    "configuration": cfg,
                    "status": val(exp, "status", "Queued"),
                    "seed": val(design, "seed", "") if design else "",
                    "runs": [
                        {
                            "id": val(run, "id", ""),
                            "status": val(run, "status", ""),
                            "seed": val(run, "seed", ""),
                            "metrics": dict(val(run, "metrics", {}) or {}),
                        }
                        for run in list(val(exp, "runs", []) or [])
                    ],
                    "result": result,
                    "failure_type": val(exp, "failure_type"),
                    "error": val(exp, "error"),
                }
            )
        evidence = [
            {
                "id": val(item, "id", ""),
                "hypothesis_id": val(item, "hypothesis_id", ""),
                "experiment_id": val(item, "experiment_id", ""),
                "run_ids": list(val(item, "run_ids", []) or []),
                "observation": val(item, "claim", ""),
                "claim": val(item, "claim", ""),
                "strength": val(item, "strength", 0.0),
                "rule": val(item, "rule", ""),
                "classification": val(item, "classification", "evidence"),
            }
            for item in list(getattr(project, "evidence", []) or [])
        ]
        budget_obj = getattr(project, "budget", None)
        budget = {}
        if budget_obj is not None:
            budget = {
                "used_experiments": val(budget_obj, "experiments_used", 0),
                "max_experiments": val(budget_obj, "maximum_experiments", val(project, "max_experiments", 0)),
                "used_runs": val(budget_obj, "runs_used", 0),
                "max_runs": val(budget_obj, "maximum_runs", 0),
                "runtime_seconds": val(budget_obj, "runtime_seconds_used", 0.0),
                "max_runtime_seconds": val(budget_obj, "maximum_runtime_seconds", 0.0),
                "estimated_api_cost": val(budget_obj, "actual_api_cost", 0.0),
                "budget_cost": val(budget_obj, "estimated_api_cost", 0.0),
            }
        best = max(hypotheses, key=lambda row: _as_float(row.get("confidence"), 0.0), default=None)
        queued = [row for row in experiments if str(row.get("status")) == "Queued"]
        next_experiment = queued[0] if queued else None
        journal = []
        for entry in list(getattr(project, "journal", []) or []):
            journal.append(
                {
                    "time": val(entry, "timestamp", ""),
                    "event": val(entry, "event_type", ""),
                    "summary": val(entry, "summary", ""),
                    "why": val(entry, "why", ""),
                    "experiment_id": val(entry, "experiment_id"),
                    "hypothesis_id": val(entry, "hypothesis_id"),
                }
            )
        graph_nodes: list[dict[str, Any]] = [{"id": "Q", "label": "Question", "type": "question"}]
        graph_edges: list[dict[str, Any]] = []
        for h in hypotheses:
            graph_nodes.append({"id": h["id"], "label": h.get("metadata", {}).get("strategy", h["id"]), "type": "hypothesis"})
            graph_edges.append({"source": "Q", "target": h["id"]})
        for exp in experiments:
            graph_nodes.append({"id": exp["id"], "label": exp["id"], "type": "experiment"})
            graph_edges.append({"source": exp["hypothesis_id"], "target": exp["id"]})
            if exp.get("result"):
                rid = f"R-{exp['id']}"
                graph_nodes.append({"id": rid, "label": "Result", "type": "result"})
                graph_edges.append({"source": exp["id"], "target": rid})
        for ev in evidence:
            graph_nodes.append({"id": ev["id"], "label": ev["id"], "type": "evidence"})
            graph_edges.append({"source": f"R-{ev['experiment_id']}", "target": ev["id"]})
            graph_edges.append({"source": ev["id"], "target": ev["hypothesis_id"], "relation": "revises"})
        return {
            "question": val(project, "research_question", ""),
            "status": val(project, "status", "Idle"),
            "hypotheses": hypotheses,
            "experiments": experiments,
            "evidence": evidence,
            "best_hypothesis": best,
            "next_experiment": next_experiment,
            "budget": budget,
            "graph": {"nodes": graph_nodes, "edges": graph_edges},
            "journal": journal,
        }

    def start_demo(self) -> Any:
        result = self._call(("start_demo", "create_demo_study", "demo_study", "run_demo"))
        if result is not None:
            return result
        if self.service is not None and callable(getattr(self.service, "create_project", None)):
            self.service.create_project("Which memory strategy is most suitable for a long-term Agent?", name="Memory Strategy Study")
            self.service.propose_hypotheses()
            designs = self.service.create_experiment_plan(seed=42, sample_size=5)
            self.service.queue_plan(designs)
            return self.get_snapshot()
        return None

    def create_study(self, question: str, **kwargs: Any) -> Any:
        result = self._call(
            ("create_study", "new_study", "create_project", "start_study"), question, **kwargs
        )
        if result is not None:
            return result
        if self.service is not None and callable(getattr(self.service, "create_project", None)):
            self.service.create_project(question)
            if callable(getattr(self.service, "propose_hypotheses", None)):
                self.service.propose_hypotheses()
            if callable(getattr(self.service, "create_experiment_plan", None)) and callable(getattr(self.service, "queue_plan", None)):
                self.service.queue_plan(self.service.create_experiment_plan(seed=42, sample_size=5))
            return self.get_snapshot()
        return None

    def resume(self) -> Any:
        result = self._call(("resume", "resume_latest", "load_latest", "load_state"))
        return result if result is not None else self.get_snapshot()

    def run_next(self) -> Any:
        result = self._call(("run_next", "run_next_experiment", "execute_next", "step"))
        if result is not None:
            return result
        if self.service is not None and callable(getattr(self.service, "select_next_experiment", None)):
            selected = self.service.select_next_experiment()
            if selected is not None and callable(getattr(self.service, "run_experiment", None)):
                project = getattr(self.service, "project", None)
                approval = str(getattr(project, "approval_mode", "Autonomous Sandbox")) == "Autonomous Sandbox"
                self.service.run_experiment(selected, runs=getattr(getattr(selected, "design", None), "sample_size", 5), approved=approval)
            return self.get_snapshot()
        return None

    def run_all(self) -> Any:
        result = self._call(("run_all", "run_all_experiments", "execute_all"))
        if result is not None:
            return result
        if self.service is not None and callable(getattr(self.service, "run_scientific_cycle", None)):
            self.service.run_scientific_cycle(rounds=2, runs_per_experiment=5, approved=True)
            return self.get_snapshot()
        return None

    def stop(self) -> Any:
        result = self._call(("stop", "stop_experiments", "cancel"))
        if result is not None:
            return result
        if self.service is not None and callable(getattr(self.service, "request_stop", None)):
            self.service.request_stop()
        return self.get_snapshot()

    def generate_report(self, output_path: str | None = None) -> Any:
        if output_path:
            result = self._call(("generate_report", "export_report", "write_report"), output_path)
            if result is not None:
                return result
        result = self._call(("generate_report", "export_report", "write_report"))
        if result is not None:
            return result
        # The core engine intentionally keeps report rendering in a separate
        # module.  A concise Markdown fallback keeps the UI's Report button
        # useful when an engine instance is passed directly.
        snapshot = self.get_snapshot()
        report = _snapshot_markdown_report(snapshot)
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report, encoding="utf-8")
            snapshot.report_path = str(target)
        return snapshot

    def graph(self) -> Any:
        return self._call(("graph", "build_graph", "hypothesis_graph"))


# ---------------------------------------------------------------------------
# Deterministic fallback service (also useful for UI smoke tests)
# ---------------------------------------------------------------------------


class LocalDemoService:
    """A tiny local synthetic study used when no engine is supplied.

    It deliberately exposes the same adapter methods as a future real engine.
    Values are deterministic for a seed and all costs are zero.  It is not
    intended to replace the scientific engine; it simply makes the dashboard
    immediately demonstrable and gives integration tests a stable fixture.
    """

    STRATEGIES = (
        ("H1", "No Memory", 0.38),
        ("H2", "Sliding Window", 0.52),
        ("H3", "Summary", 0.68),
        ("H4", "Episodic", 0.75),
        ("H5", "Vector", 0.71),
    )

    def __init__(self, storage_path: str | Path | None = None, seed: int = 42):
        self.storage_path = Path(storage_path) if storage_path else None
        self.seed = int(seed)
        self.question = ""
        self.status = "Idle"
        self.hypotheses: list[dict[str, Any]] = []
        self.experiments: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.journal: list[dict[str, Any]] = []
        self.next_experiment: dict[str, Any] | None = None
        self.round = 0
        self._queue: list[dict[str, Any]] = []
        self._stopped = False
        self._started_at: float | None = None
        self.max_experiments = 10
        self.max_runs = 40
        self.max_runtime_seconds = 30

    def _reset(self, question: str) -> None:
        self.question = question
        self.status = "Ready"
        self.hypotheses = []
        for hid, name, prior in self.STRATEGIES:
            self.hypotheses.append(
                {
                    "id": hid,
                    "statement": f"{name} improves long-term agent performance",
                    "name": name,
                    "rationale": "Candidate memory strategy generated by transparent rules.",
                    "confidence": prior,
                    "prior_confidence": prior,
                    "status": "Untested",
                    "evidence": [],
                }
            )
        self.experiments = []
        self.evidence = []
        self.journal = []
        self.round = 0
        self._queue = [self._make_config(hid, 1) for hid, _, _ in self.STRATEGIES]
        self.next_experiment = self._queue[0] if self._queue else None
        self._stopped = False
        self._started_at = time.time()
        self._log("study_created", "Generated five candidate memory strategies and an initial test queue.")

    def _make_config(self, hypothesis_id: str, round_no: int) -> dict[str, Any]:
        seed_material = f"{self.seed}:{hypothesis_id}:{round_no}".encode()
        stable_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        return {
            "id": f"EXP-{round_no}-{hypothesis_id}",
            "hypothesis_id": hypothesis_id,
            "round": round_no,
            "strategy": next((name for hid, name, _ in self.STRATEGIES if hid == hypothesis_id), hypothesis_id),
            "sample_size": 8,
            "runs": 4,
            "seed": stable_seed,
            "metric": "task_success_rate",
            "status": "Queued",
        }

    def _log(self, event: str, summary: str, **extra: Any) -> None:
        row = {"time": time.strftime("%H:%M:%S"), "event": event, "summary": summary}
        row.update(extra)
        self.journal.append(row)

    def start_demo(self) -> DashboardSnapshot:
        self._reset("Which memory strategy is most suitable for a long-term Agent?")
        return self.snapshot()

    def create_study(self, question: str, **kwargs: Any) -> DashboardSnapshot:
        self._reset(question.strip() or "Untitled research question")
        if "seed" in kwargs:
            try:
                self.seed = int(kwargs["seed"])
            except (TypeError, ValueError):
                pass
        return self.snapshot()

    def resume(self) -> DashboardSnapshot:
        if self.storage_path and self.storage_path.exists():
            try:
                payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._load_dict(payload)
            except Exception:
                pass
        if not self.question:
            self.start_demo()
        self.status = "Ready" if self.status in {"Idle", "Stopped"} else self.status
        self._log("resume", "Resumed the latest local study state.")
        return self.snapshot()

    def _load_dict(self, payload: Mapping[str, Any]) -> None:
        for name in ("question", "status", "hypotheses", "experiments", "evidence", "journal", "round", "seed"):
            if name in payload:
                setattr(self, name, copy.deepcopy(payload[name]))
        self._queue = copy.deepcopy(list(payload.get("queue", [])))
        self.next_experiment = copy.deepcopy(payload.get("next_experiment"))

    def _persist(self) -> None:
        if not self.storage_path:
            return
        payload = self._state_dict()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _state_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "hypotheses": self.hypotheses,
            "experiments": self.experiments,
            "evidence": self.evidence,
            "journal": self.journal,
            "round": self.round,
            "seed": self.seed,
            "queue": self._queue,
            "next_experiment": self.next_experiment,
        }

    def _metric_samples(self, config: Mapping[str, Any]) -> list[float]:
        strategy = str(config.get("hypothesis_id", "H1"))
        base = next((score for hid, _, score in self.STRATEGIES if hid == strategy), 0.5)
        rng = random.Random(int(config.get("seed", self.seed)))
        n = int(config.get("runs", 4))
        return [max(0.0, min(1.0, rng.gauss(base, 0.08))) for _ in range(max(1, n))]

    def run_next(self) -> DashboardSnapshot:
        if not self.question:
            self.start_demo()
        if self._stopped:
            self.status = "Stopped"
            return self.snapshot()
        if not self._queue:
            self._select_follow_up()
        if not self._queue:
            self.status = "Complete"
            self.next_experiment = None
            self._log("complete", "No eligible follow-up experiment remains within the research budget.")
            self._persist()
            return self.snapshot()
        if len(self.experiments) >= self.max_experiments:
            self.status = "Budget Exceeded"
            self._log("budget", "Experiment limit reached; execution stopped.")
            return self.snapshot()
        config = self._queue.pop(0)
        self.next_experiment = self._queue[0] if self._queue else None
        self.status = "Running"
        self._log("experiment_started", "Executed a deterministic synthetic experiment.", experiment_id=config["id"])
        samples = self._metric_samples(config)
        mean = statistics.fmean(samples)
        median = statistics.median(samples)
        std = statistics.stdev(samples) if len(samples) > 1 else 0.0
        se = std / math.sqrt(len(samples)) if samples else 0.0
        ci = (max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se))
        success = mean >= 0.60
        result = {
            "mean": mean,
            "median": median,
            "std": std,
            "confidence_interval": ci,
            "effect_size": mean - 0.50,
            "success_rate": mean,
            "samples": samples,
        }
        row = dict(config)
        row.update({"status": "Complete", "result": result, "completed_at": time.strftime("%H:%M:%S")})
        self.experiments.append(row)
        hid = str(config.get("hypothesis_id", ""))
        hypothesis = next((item for item in self.hypotheses if item.get("id") == hid), None)
        if hypothesis is not None:
            prior = float(hypothesis.get("confidence", 0.5))
            # Transparent update rule: move 15% of the way toward observed mean.
            updated = max(0.0, min(1.0, prior + 0.15 * (mean - 0.5)))
            hypothesis["confidence"] = updated
            hypothesis["status"] = "Supported" if success else "Inconclusive"
            evidence_id = f"E-{len(self.evidence)+1:03d}"
            hypothesis.setdefault("evidence", []).append(evidence_id)
            evidence = {
                "id": evidence_id,
                "hypothesis_id": hid,
                "experiment_id": config["id"],
                "run_ids": [f"{config['id']}-R{i+1}" for i in range(len(samples))],
                "observation": f"Observed mean {mean:.3f} for {config.get('strategy', hid)}.",
                "rule": "confidence := prior + 0.15 × (mean − 0.50); success if mean ≥ 0.60",
                "result": "positive" if success else "uncertain",
            }
            self.evidence.append(evidence)
            self._log(
                "hypothesis_updated",
                f"Updated {hid} from {prior:.3f} to {updated:.3f} using the transparent confidence rule.",
                hypothesis_id=hid,
                evidence_id=evidence_id,
            )
        self._select_follow_up()
        self.status = "Ready" if self._queue else "Complete"
        self._persist()
        return self.snapshot()

    def _select_follow_up(self) -> None:
        if self._queue:
            return
        if self.round >= 2 or len(self.experiments) >= self.max_experiments:
            self.next_experiment = None
            return
        # Uncertainty-first: the least confident untested/weakly tested strategy.
        # After the initial queue has completed, all hypotheses are eligible for
        # a registered follow-up.  The round index selects the next least
        # confident candidate, which keeps the rule deterministic and makes the
        # two-round demo visible in the dashboard.
        candidates = list(self.hypotheses)
        if not candidates:
            return
        self.round += 1
        ordered = sorted(candidates, key=lambda h: _as_float(h.get("confidence", 0.5), 0.5))
        chosen = ordered[min(self.round - 1, len(ordered) - 1)]
        config = self._make_config(str(chosen.get("id")), self.round + 1)
        config["runs"] = 4
        config["selection"] = "Uncertainty First"
        self._queue.append(config)
        self.next_experiment = config
        self._log("selection", "Selected the next experiment using Uncertainty First.", experiment_id=config["id"])

    def run_all(self) -> DashboardSnapshot:
        guard = 0
        while self._queue and not self._stopped and guard < self.max_experiments:
            self.run_next()
            guard += 1
        return self.snapshot()

    def stop(self) -> DashboardSnapshot:
        self._stopped = True
        self.status = "Stopped"
        self._log("stopped", "Execution stopped by the user.")
        self._persist()
        return self.snapshot()

    def generate_report(self, output_path: str | None = None) -> str:
        if not self.question:
            self.start_demo()
        best = self._best_hypothesis()
        lines = [
            "# AI Scientist Mini — Research Report",
            "",
            f"## Research Question\n{self.question}",
            "",
            "## Hypotheses",
        ]
        for h in self.hypotheses:
            lines.append(f"- **{h.get('id')}** {h.get('statement')} — {h.get('status')}, confidence {float(h.get('confidence', 0))*100:.1f}%")
        lines += ["", "## Methods", "SyntheticProvider; deterministic seed; transparent rule-based update.", "", "## Experiments"]
        for exp in self.experiments:
            result = exp.get("result", {})
            lines.append(f"- {exp.get('id')}: {exp.get('status')}; mean={float(result.get('mean', 0)):.3f}; seed={exp.get('seed')}")
        lines += ["", "## Interpretation", f"Best current hypothesis: {best.get('statement') if best else 'not enough evidence'}.", "", "## Limitations", "Synthetic observations are not evidence about the real world; failures are not hypothesis rejections.", "", "## Next Work", "Run a registered follow-up with an approved external provider only after reviewing budget and design."]
        report = "\n".join(lines) + "\n"
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            self._log("report", "Generated a public decision-summary report.", path=str(path))
            self._persist()
            return str(path)
        return report

    def _best_hypothesis(self) -> dict[str, Any] | None:
        return max(self.hypotheses, key=lambda h: float(h.get("confidence", 0.0)), default=None)

    def snapshot(self) -> DashboardSnapshot:
        best = self._best_hypothesis()
        elapsed = (time.time() - self._started_at) if self._started_at else 0.0
        return DashboardSnapshot(
            question=self.question,
            status=self.status,
            hypotheses=copy.deepcopy(self.hypotheses),
            experiments=copy.deepcopy(self.experiments),
            evidence=copy.deepcopy(self.evidence),
            best_hypothesis=copy.deepcopy(best),
            next_experiment=copy.deepcopy(self.next_experiment),
            budget={
                "used_experiments": len(self.experiments),
                "max_experiments": self.max_experiments,
                "used_runs": sum(len(e.get("result", {}).get("samples", [])) for e in self.experiments),
                "max_runs": self.max_runs,
                "runtime_seconds": round(elapsed, 2),
                "max_runtime_seconds": self.max_runtime_seconds,
                "estimated_api_cost": 0.0,
            },
            graph=self._graph(),
            journal=copy.deepcopy(self.journal),
            message="Synthetic local study; no external API calls.",
        )

    def _graph(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = [{"id": "Q", "label": "Question", "type": "question"}]
        edges: list[dict[str, Any]] = []
        for h in self.hypotheses:
            hid = str(h.get("id"))
            nodes.append({"id": hid, "label": str(h.get("name", hid)), "type": "hypothesis"})
            edges.append({"source": "Q", "target": hid})
        for exp in self.experiments:
            eid = str(exp.get("id"))
            hid = str(exp.get("hypothesis_id"))
            nodes.append({"id": eid, "label": eid, "type": "experiment"})
            edges.append({"source": hid, "target": eid})
            result_id = f"R-{eid}"
            nodes.append({"id": result_id, "label": "Result", "type": "result"})
            edges.append({"source": eid, "target": result_id})
        for evidence in self.evidence:
            evidence_id = str(evidence.get("id"))
            experiment_id = str(evidence.get("experiment_id", ""))
            hypothesis_id = str(evidence.get("hypothesis_id", ""))
            nodes.append({"id": evidence_id, "label": evidence_id, "type": "evidence"})
            if experiment_id:
                edges.append({"source": f"R-{experiment_id}", "target": evidence_id})
            if hypothesis_id:
                # The dotted-looking semantic edge is represented by the same
                # simple line primitive in the Tk canvas; the label remains
                # available to API/CLI graph consumers.
                edges.append({"source": evidence_id, "target": hypothesis_id, "relation": "revises"})
        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Controller (usable without Tk and easy to test)
# ---------------------------------------------------------------------------


class DashboardController:
    """Coordinate service operations and expose a snapshot for the view."""

    def __init__(self, service: Any | None = None, on_change: Callable[[DashboardSnapshot], None] | None = None):
        self.adapter = service if isinstance(service, StudyServiceAdapter) else StudyServiceAdapter(service or LocalDemoService())
        self.on_change = on_change
        self.last_error = ""
        self._snapshot = DashboardSnapshot.from_any(self.adapter.get_snapshot())

    @property
    def snapshot(self) -> DashboardSnapshot:
        return copy.deepcopy(self._snapshot)

    def refresh(self) -> DashboardSnapshot:
        self._snapshot = DashboardSnapshot.from_any(self.adapter.get_snapshot())
        self._notify()
        return self.snapshot

    def _notify(self) -> None:
        if self.on_change:
            self.on_change(self.snapshot)

    def _operation(self, method: str, *args: Any, **kwargs: Any) -> DashboardSnapshot:
        self.last_error = ""
        try:
            result = getattr(self.adapter, method)(*args, **kwargs)
            # Services may return a snapshot directly; otherwise refresh.
            self._snapshot = DashboardSnapshot.from_any(result) if result is not None else DashboardSnapshot.from_any(self.adapter.get_snapshot())
        except Exception as exc:  # UI should stay alive and surface a concise error.
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._snapshot.message = self.last_error
            traceback.print_exc()
        self._notify()
        return self.snapshot

    def start_demo(self) -> DashboardSnapshot:
        return self._operation("start_demo")

    def new_study(self, question: str, **kwargs: Any) -> DashboardSnapshot:
        return self._operation("create_study", question, **kwargs)

    def resume(self) -> DashboardSnapshot:
        return self._operation("resume")

    def run_next(self) -> DashboardSnapshot:
        return self._operation("run_next")

    def run_all(self) -> DashboardSnapshot:
        return self._operation("run_all")

    def stop(self) -> DashboardSnapshot:
        return self._operation("stop")

    def generate_report(self, output_path: str | None = None) -> DashboardSnapshot:
        return self._operation("generate_report", output_path)


# ---------------------------------------------------------------------------
# Tk view
# ---------------------------------------------------------------------------


def _display(value: Any, default: str = "—") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return str(value)


def _snapshot_markdown_report(snapshot: DashboardSnapshot) -> str:
    """Render a public decision-summary report from UI state.

    This deliberately contains no hidden reasoning: only registered
    hypotheses, configurations, aggregate results, evidence, limitations and
    next-work guidance are included.
    """

    vm = DashboardViewModel.from_snapshot(snapshot)
    lines = [
        "# AI Scientist Mini — Research Report",
        "",
        "## Research Question",
        vm.question or "(not specified)",
        "",
        "## Hypotheses",
    ]
    for row in vm.hypotheses:
        confidence = _as_float(row.get("confidence"), 0.0)
        confidence_text = f"{confidence * 100:.1f}%" if confidence <= 1 else f"{confidence:.1f}%"
        lines.append(f"- **{row.get('id', '—')}** {row.get('statement', row.get('name', ''))} — {row.get('status', 'Untested')}; confidence {confidence_text}")
    lines += ["", "## Methods", "Experiments were executed through the configured local adapter. Review the reproducibility fields (seed, dataset, code version and data hash) before interpreting results.", "", "## Experiments"]
    for row in vm.experiments:
        result = row.get("result", {}) if isinstance(row.get("result", {}), Mapping) else {}
        lines.append(f"- {row.get('id', '—')} — {row.get('status', '—')}; hypothesis={row.get('hypothesis_id', '—')}; seed={row.get('seed', '—')}; mean={_display(result.get('mean', '—'))}")
    lines += ["", "## Evidence"]
    for row in vm.evidence:
        lines.append(f"- {row.get('id', '—')}: {row.get('observation', row.get('claim', ''))} (rule: {row.get('rule', '—')})")
    lines += ["", "## Interpretation", f"Best current hypothesis: {vm.best_text}.", "", "## Limitations", "Synthetic or local results do not establish truth about the real world. Infrastructure failure, invalid design, insufficient data and negative results must be interpreted separately.", "", "## Next Work", f"{_display(vm.next_experiment, 'No registered follow-up remains within the current budget.')}", ""]
    return "\n".join(lines)


class Dashboard(ttk.Frame if ttk else object):  # type: ignore[misc]
    """Tkinter dashboard widget.

    Instantiate with an existing ``tk.Tk`` root and optionally a core service.
    ``Dashboard(root, service=my_engine)`` is the intended integration point.
    """

    BG = "#111827"
    PANEL = "#1f2937"
    PANEL_ALT = "#243447"
    FG = "#e5e7eb"
    MUTED = "#9ca3af"
    ACCENT = "#60a5fa"
    GOOD = "#34d399"
    WARN = "#fbbf24"
    BAD = "#f87171"

    def __init__(self, master: Any, service: Any | None = None, controller: DashboardController | None = None, **kwargs: Any):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this Python installation")
        super().__init__(master, **kwargs)
        self.master = master
        self.controller = controller or DashboardController(service, on_change=self._on_snapshot)
        if controller is not None:
            self.controller.on_change = self._on_snapshot
        self._build_style()
        self._build_widgets()
        self._on_snapshot(self.controller.snapshot)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dashboard.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Title.TLabel", background=self.BG, foreground=self.FG, font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=self.PANEL, foreground=self.FG)
        style.configure("PanelMuted.TLabel", background=self.PANEL, foreground=self.MUTED)
        style.configure("KpiValue.TLabel", background=self.PANEL, foreground=self.FG, font=("Segoe UI", 14, "bold"))
        style.configure("KpiLabel.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 8))
        style.configure("Accent.TButton", foreground="#0b1220")
        style.configure("Treeview", background="#172234", fieldbackground="#172234", foreground=self.FG, rowheight=25)
        style.configure("Treeview.Heading", background="#334155", foreground=self.FG, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#1d4ed8")])

    def _build_widgets(self) -> None:
        self.configure(style="Dashboard.TFrame")
        self.pack(fill="both", expand=True)
        header = ttk.Frame(self, style="Dashboard.TFrame", padding=(18, 14, 18, 8))
        header.pack(fill="x")
        title_box = ttk.Frame(header, style="Dashboard.TFrame")
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text="AI Scientist Mini", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="自动实验科学家 · transparent, reproducible research loop", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        controls = ttk.Frame(header, style="Dashboard.TFrame")
        controls.pack(side="right")
        self._button(controls, "New Study", self._new_study, accent=True)
        self._button(controls, "Demo Study", self._demo)
        self._button(controls, "Resume", self._resume)
        self._button(controls, "Run Next", self._run_next)
        self._button(controls, "Run All", self._run_all)
        self._button(controls, "Stop", self._stop)
        self._button(controls, "Report", self._report)

        qpanel = ttk.Frame(self, style="Panel.TFrame", padding=12)
        qpanel.pack(fill="x", padx=18, pady=(0, 10))
        ttk.Label(qpanel, text="CURRENT RESEARCH QUESTION", style="PanelMuted.TLabel").pack(anchor="w")
        self.question_var = tk.StringVar(value="—")
        ttk.Label(qpanel, textvariable=self.question_var, style="Panel.TLabel", wraplength=1000, font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 0))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(qpanel, textvariable=self.status_var, style="PanelMuted.TLabel").pack(anchor="w", pady=(5, 0))

        kpis = ttk.Frame(self, style="Dashboard.TFrame")
        kpis.pack(fill="x", padx=18, pady=(0, 10))
        self.kpi_vars: dict[str, tk.StringVar] = {}
        for key, label in (("hypotheses", "ACTIVE HYPOTHESES"), ("experiments", "EXPERIMENTS"), ("evidence", "EVIDENCE"), ("best", "BEST HYPOTHESIS"), ("confidence", "CONFIDENCE"), ("budget", "BUDGET")):
            panel = ttk.Frame(kpis, style="Panel.TFrame", padding=(10, 8))
            panel.pack(side="left", fill="both", expand=True, padx=(0, 7))
            ttk.Label(panel, text=label, style="KpiLabel.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            self.kpi_vars[key] = var
            ttk.Label(panel, textvariable=var, style="KpiValue.TLabel", wraplength=180).pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(self, style="Dashboard.TFrame")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)
        self.overview_tab = ttk.Frame(self.notebook, style="Dashboard.TFrame", padding=8)
        self.hypotheses_tab = ttk.Frame(self.notebook, style="Dashboard.TFrame", padding=8)
        self.experiments_tab = ttk.Frame(self.notebook, style="Dashboard.TFrame", padding=8)
        self.evidence_tab = ttk.Frame(self.notebook, style="Dashboard.TFrame", padding=8)
        self.journal_tab = ttk.Frame(self.notebook, style="Dashboard.TFrame", padding=8)
        self.graph_tab = ttk.Frame(self.notebook, style="Dashboard.TFrame", padding=8)
        for tab, label in ((self.overview_tab, "Overview"), (self.hypotheses_tab, "Hypotheses"), (self.experiments_tab, "Experiments"), (self.evidence_tab, "Evidence"), (self.journal_tab, "Journal"), (self.graph_tab, "Hypothesis Graph")):
            self.notebook.add(tab, text=label)
        self._build_overview()
        self._build_tables()
        self._build_journal()
        self._build_graph()
        self.message_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.message_var, style="Subtitle.TLabel", padding=(18, 0, 18, 8)).pack(fill="x")

    def _button(self, parent: Any, text: str, command: Callable[[], None], accent: bool = False) -> None:
        ttk.Button(parent, text=text, command=command, style="Accent.TButton" if accent else "TButton").pack(side="left", padx=(4, 0))

    def _build_overview(self) -> None:
        self.overview_tab.columnconfigure(0, weight=1)
        self.overview_tab.columnconfigure(1, weight=1)
        self.overview_tab.rowconfigure(1, weight=1)
        next_panel = ttk.Frame(self.overview_tab, style="Panel.TFrame", padding=12)
        next_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        ttk.Label(next_panel, text="NEXT EXPERIMENT", style="PanelMuted.TLabel").pack(anchor="w")
        self.next_var = tk.StringVar(value="No experiment selected")
        ttk.Label(next_panel, textvariable=self.next_var, style="Panel.TLabel", wraplength=500).pack(anchor="w", pady=(5, 0))
        best_panel = ttk.Frame(self.overview_tab, style="Panel.TFrame", padding=12)
        best_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        ttk.Label(best_panel, text="BEST CURRENT HYPOTHESIS", style="PanelMuted.TLabel").pack(anchor="w")
        self.best_var = tk.StringVar(value="No evidence yet")
        ttk.Label(best_panel, textvariable=self.best_var, style="Panel.TLabel", wraplength=500).pack(anchor="w", pady=(5, 0))
        self.overview_text = tk.Text(self.overview_tab, height=10, bg="#172234", fg=self.FG, insertbackground=self.FG, relief="flat", wrap="word")
        self.overview_text.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        self.overview_text.configure(state="disabled")

    def _tree(self, parent: Any, columns: Sequence[tuple[str, str, int]], height: int = 10) -> Any:
        frame = ttk.Frame(parent, style="Dashboard.TFrame")
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=height)
        for name, heading, width in columns:
            tree.heading(name, text=heading)
            tree.column(name, width=width, minwidth=max(45, width // 2), anchor="w")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _build_tables(self) -> None:
        self.hypotheses_tree = self._tree(self.hypotheses_tab, (("id", "ID", 65), ("statement", "Statement", 420), ("confidence", "Confidence", 95), ("status", "Status", 100), ("evidence", "Evidence", 120)))
        self.experiments_tree = self._tree(self.experiments_tab, (("id", "ID", 110), ("hypothesis", "Hypothesis", 95), ("strategy", "Configuration", 170), ("status", "Status", 95), ("mean", "Mean", 90), ("seed", "Seed", 110)))
        self.evidence_tree = self._tree(self.evidence_tab, (("id", "ID", 75), ("hypothesis", "Hypothesis", 95), ("experiment", "Experiment", 115), ("observation", "Observation", 390), ("rule", "Update Rule", 300)))

    def _build_journal(self) -> None:
        self.journal_text = tk.Text(self.journal_tab, bg="#172234", fg=self.FG, insertbackground=self.FG, relief="flat", wrap="word")
        self.journal_text.pack(fill="both", expand=True)
        self.journal_text.configure(state="disabled")

    def _build_graph(self) -> None:
        graph_toolbar = ttk.Frame(self.graph_tab, style="Dashboard.TFrame")
        graph_toolbar.pack(fill="x", pady=(0, 5))
        ttk.Label(graph_toolbar, text="Question → Hypothesis → Experiment → Evidence", style="Subtitle.TLabel").pack(side="left")
        ttk.Button(graph_toolbar, text="Refresh Graph", command=self._refresh).pack(side="right")
        self.graph_canvas = tk.Canvas(self.graph_tab, background="#0f172a", highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True)
        self.graph_canvas.bind("<Configure>", lambda _event: self._draw_graph(self.controller.snapshot))

    def _on_snapshot(self, snapshot: DashboardSnapshot) -> None:
        if not hasattr(self, "question_var"):
            return
        vm = DashboardViewModel.from_snapshot(snapshot)
        self.question_var.set(vm.question or "—")
        self.status_var.set(f"Status: {vm.status}")
        self.kpi_vars["hypotheses"].set(str(len(vm.hypotheses)))
        self.kpi_vars["experiments"].set(str(len(vm.experiments)))
        self.kpi_vars["evidence"].set(str(len(vm.evidence)))
        self.kpi_vars["best"].set(_display((vm.best_hypothesis or {}).get("name", "—")))
        confidence = vm.confidence
        self.kpi_vars["confidence"].set("—" if confidence is None else f"{confidence*100:.1f}%" if confidence <= 1 else f"{confidence:.1f}%")
        self.kpi_vars["budget"].set(vm.budget_text)
        next_exp = vm.next_experiment or {}
        self.next_var.set(_display(next_exp, "No experiment selected"))
        self.best_var.set(vm.best_text)
        overview_lines = [
            "Decision summary",
            f"Question: {vm.question or '—'}",
            f"Status: {vm.status}",
            f"Budget: {vm.budget_text}",
            f"Best hypothesis: {vm.best_text}",
            f"Next experiment: {_display(next_exp, 'none')}",
            "",
            "This dashboard shows public decision summaries only; private model chain-of-thought is neither requested nor stored.",
        ]
        self._set_text(self.overview_text, "\n".join(overview_lines))
        self._fill_hypotheses(vm.hypotheses)
        self._fill_experiments(vm.experiments)
        self._fill_evidence(vm.evidence)
        self._fill_journal(vm.journal)
        self._draw_graph(snapshot)
        message = vm.message or "Ready"
        if self.controller.last_error:
            message = self.controller.last_error
        self.message_var.set(message)

    @staticmethod
    def _set_text(widget: Any, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _fill_hypotheses(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for item in self.hypotheses_tree.get_children():
            self.hypotheses_tree.delete(item)
        for row in rows:
            confidence = row.get("confidence", "")
            try:
                confidence = f"{float(confidence)*100:.1f}%" if float(confidence) <= 1 else f"{float(confidence):.1f}%"
            except (TypeError, ValueError):
                confidence = _display(confidence)
            self.hypotheses_tree.insert("", "end", values=(row.get("id", ""), row.get("statement", row.get("name", "")), confidence, row.get("status", ""), _display(row.get("evidence", ""))))

    def _fill_experiments(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for item in self.experiments_tree.get_children():
            self.experiments_tree.delete(item)
        for row in rows:
            result = row.get("result", {}) or {}
            mean = result.get("mean", row.get("mean", "")) if isinstance(result, Mapping) else ""
            self.experiments_tree.insert("", "end", values=(row.get("id", ""), row.get("hypothesis_id", ""), row.get("strategy", row.get("configuration", "")), row.get("status", ""), _display(mean), row.get("seed", "")))

    def _fill_evidence(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for item in self.evidence_tree.get_children():
            self.evidence_tree.delete(item)
        for row in rows:
            self.evidence_tree.insert("", "end", values=(row.get("id", ""), row.get("hypothesis_id", ""), row.get("experiment_id", ""), row.get("observation", ""), row.get("rule", "")))

    def _fill_journal(self, rows: Iterable[Any]) -> None:
        lines: list[str] = []
        for row in rows:
            if isinstance(row, str):
                lines.append(row)
            else:
                row = _row_to_dict(row)
                stamp = row.get("time", "")
                event = row.get("event", "")
                summary = row.get("summary", row.get("message", ""))
                lines.append(f"[{stamp}] {event}: {summary}")
        self._set_text(self.journal_text, "\n".join(lines) if lines else "No journal entries yet.")

    def _draw_graph(self, snapshot: Any) -> None:
        if not hasattr(self, "graph_canvas"):
            return
        canvas = self.graph_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 600)
        height = max(canvas.winfo_height(), 350)
        snap = DashboardSnapshot.from_any(snapshot)
        graph = snap.graph or {}
        nodes = graph.get("nodes", []) if isinstance(graph, Mapping) else []
        edges = graph.get("edges", []) if isinstance(graph, Mapping) else []
        if not nodes:
            nodes = [{"id": "Q", "label": "Question", "type": "question"}]
        positions: dict[str, tuple[float, float]] = {}
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for node in nodes:
            row = _row_to_dict(node)
            groups.setdefault(str(row.get("type", "other")), []).append(row)
        order = ["question", "hypothesis", "experiment", "evidence", "result"]
        columns = [key for key in order if key in groups] + [key for key in groups if key not in order]
        col_width = width / max(1, len(columns))
        for index, kind in enumerate(columns):
            items = groups[kind]
            for item_index, node in enumerate(items):
                x = col_width * (index + 0.5)
                y = 45 + (height - 85) * ((item_index + 1) / (len(items) + 1))
                positions[str(node.get("id"))] = (x, y)
        for edge in edges:
            edge = _row_to_dict(edge)
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            if source in positions and target in positions:
                canvas.create_line(*positions[source], *positions[target], fill="#64748b", arrow="last", width=1)
        colors = {"question": "#fbbf24", "hypothesis": "#60a5fa", "experiment": "#34d399", "evidence": "#c084fc", "result": "#fb7185"}
        for kind, items in groups.items():
            for node in items:
                node_id = str(node.get("id"))
                x, y = positions.get(node_id, (width / 2, height / 2))
                color = colors.get(kind, "#94a3b8")
                label = str(node.get("label", node_id))
                if len(label) > 22:
                    label = label[:20] + "…"
                canvas.create_oval(x - 48, y - 18, x + 48, y + 18, fill=color, outline="")
                canvas.create_text(x, y, text=label, fill="#0f172a", font=("Segoe UI", 8, "bold"))

    def _refresh(self) -> None:
        self.controller.refresh()

    def _demo(self) -> None:
        self.controller.start_demo()

    def _resume(self) -> None:
        self.controller.resume()

    def _run_next(self) -> None:
        self.controller.run_next()

    def _run_all(self) -> None:
        self.controller.run_all()

    def _stop(self) -> None:
        self.controller.stop()

    def _new_study(self) -> None:
        if simpledialog is None:
            return
        question = simpledialog.askstring("New Research Question", "What do you want to investigate?", parent=self.master)
        if question:
            self.controller.new_study(question)

    def _report(self) -> None:
        if filedialog is None:
            return
        path = filedialog.asksaveasfilename(parent=self.master, title="Save research report", defaultextension=".md", filetypes=(("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")))
        if not path:
            return
        self.controller.generate_report(path)
        if messagebox is not None:
            messagebox.showinfo("Report", f"Report saved to:\n{path}", parent=self.master)


def launch(service: Any | None = None, *, title: str = "AI Scientist Mini", geometry: str = "1280x820") -> Any:
    """Launch the desktop dashboard and return the root after its event loop.

    This function is kept tiny so packaging tools can use it as their console
    entry point.  It raises a clear error in headless environments.
    """

    if tk is None or ttk is None:
        raise RuntimeError("Tkinter is not available; install the Tk runtime to launch the desktop UI")
    root = tk.Tk()
    root.title(title)
    root.geometry(geometry)
    root.minsize(960, 600)
    Dashboard(root, service=service)
    root.mainloop()
    return root


__all__ = [
    "Dashboard",
    "DashboardController",
    "DashboardSnapshot",
    "DashboardViewModel",
    "LocalDemoService",
    "StudyServiceAdapter",
    "launch",
]
