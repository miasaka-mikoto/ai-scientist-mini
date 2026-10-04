"""Tkinter dashboard for AI Scientist Mini.

The UI deliberately depends on a very small, duck-typed engine adapter.  The
research engine can therefore evolve independently of the desktop shell (and
can be replaced by a test double in CI).  Importing this module never creates
 a Tk root, which keeps it safe to import on headless build machines.

Run from a source checkout with::

    python -m aiscientist.ui        # normal dashboard
    python -m aiscientist.ui --headless  # smoke/demo JSON output

Only the standard library is used.  Charts and the hypothesis graph are drawn
with Tkinter Canvas rather than requiring matplotlib.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import importlib
import json
import math
import os
import queue
import random
import statistics
import sys
import threading
import time
import traceback
import types
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:  # Tk is optional on a headless machine.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - exercised only on minimal Python images
    tk = None  # type: ignore[assignment]
    filedialog = messagebox = ttk = None  # type: ignore[assignment]


APP_TITLE = "AI Scientist Mini · 自动实验科学家"
DEFAULT_RESEARCH_QUESTION = "Which memory strategy is more suitable for a long-term Agent?"


# ---------------------------------------------------------------------------
# Serialization and engine adaptation


def _jsonable(value: Any, _seen: Optional[set[int]] = None) -> Any:
    """Convert dataclasses/enums/objects into JSON-friendly values.

    Engine implementations are not required to use a specific model library.
    This conservative conversion lets the dashboard consume dataclasses,
    pydantic-like models, named tuples, or ordinary dictionaries alike.
    """

    seen = _seen if _seen is not None else set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, seen) for v in value]
    oid = id(value)
    if oid in seen:
        return "<recursive>"
    seen.add(oid)
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name), seen) for f in dataclasses.fields(value)}
    if hasattr(value, "value") and not callable(getattr(value, "value", None)):
        # enum.Enum and similar wrappers
        try:
            return _jsonable(value.value, seen)
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(), seen)
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _jsonable(value.dict(), seen)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {str(k): _jsonable(v, seen) for k, v in vars(value).items() if not str(k).startswith("_")}
        except Exception:
            pass
    return str(value)


def _first(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Get a value by trying snake/camel/case-insensitive aliases."""
    if not mapping:
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return default


def _records(value: Any) -> list[dict[str, Any]]:
    """Normalize a collection of record objects to dictionaries."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # A mapping can itself be one record, or an id -> record mapping.
        if any(k in value for k in ("id", "ID", "statement", "name", "status", "config")):
            return [_jsonable(value)]
        return [{"id": str(k), **(_jsonable(v) if isinstance(_jsonable(v), Mapping) else {"value": _jsonable(v)})} for k, v in value.items()]
    if isinstance(value, (str, bytes)):
        return [{"value": value}]
    try:
        return [dict(x) if isinstance(x, Mapping) else (_jsonable(x) if isinstance(_jsonable(x), Mapping) else {"value": _jsonable(x)}) for x in value]
    except TypeError:
        x = _jsonable(value)
        return [x] if isinstance(x, dict) else [{"value": x}]


class EngineAdapter:
    """Small compatibility layer between the dashboard and a scientist engine.

    Supported engine method aliases are intentionally broad.  A production
    engine may expose ``run_next_experiment`` while a test double exposes
    ``run_next``; both work without UI changes.  Methods return the engine's
    result and the adapter always refreshes a plain snapshot afterward.
    """

    def __init__(self, engine: Any | None = None):
        self.engine = engine if engine is not None else LocalDemoEngine()
        self._local_snapshot: dict[str, Any] | None = None
        self._approved = False

    def _call(self, names: Iterable[str], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            fn = getattr(self.engine, name, None)
            if callable(fn):
                try:
                    return fn(*args, **kwargs)
                except TypeError:
                    # A few engines accept only a subset of kwargs.  Retry
                    # without optional kwargs, then positional args.
                    try:
                        return fn(*args)
                    except TypeError:
                        if args:
                            return fn()
                        raise
        return None

    def snapshot(self) -> dict[str, Any]:
        raw: Any = None
        for name in ("snapshot", "get_snapshot", "state_dict", "to_dict", "get_state"):
            fn = getattr(self.engine, name, None)
            if callable(fn):
                try:
                    raw = fn()
                    break
                except TypeError:
                    continue
        if raw is None:
            raw = getattr(self.engine, "state", None)
        if raw is None:
            raw = getattr(self.engine, "project", None)
        data = _jsonable(raw)
        if not isinstance(data, dict):
            data = {"value": data}
        # Some engines return the project as the top-level object.  Preserve
        # it while adding conventional collection keys for the UI.
        if "project" not in data:
            # A resumed SQLite store can contain earlier projects (including
            # the blank bootstrap project).  The dashboard must show the
            # engine's active project, not merely the first record returned
            # by storage.
            active_project = _jsonable(getattr(self.engine, "project", {}))
            if isinstance(active_project, Mapping) and active_project:
                data["project"] = active_project
            elif isinstance(data.get("projects"), list) and data["projects"]:
                data["project"] = data["projects"][-1]
            else:
                data["project"] = {}
        for key in ("hypotheses", "experiments", "runs", "queue", "evidence", "journal", "observations", "decisions"):
            if key not in data:
                attr = getattr(self.engine, key, None)
                if attr is not None:
                    data[key] = _jsonable(attr)
        project = data.get("project") if isinstance(data.get("project"), Mapping) else {}
        if "budget" not in data and isinstance(project, Mapping) and project.get("budget") is not None:
            data["budget"] = _jsonable(project.get("budget"))
        # ScientificEngine keeps aggregate analyses on demand rather than in
        # its compact SQLite snapshot.  Enrich rows for the dashboard only;
        # this does not write or alter the research state.
        if not data.get("analyses"):
            analyze_all = getattr(self.engine, "analyze_all", None)
            if callable(analyze_all):
                try:
                    data["analyses"] = _jsonable(analyze_all())
                except Exception:
                    data["analyses"] = {}
        analyses = data.get("analyses") if isinstance(data.get("analyses"), Mapping) else {}
        experiments = data.get("experiments") if isinstance(data.get("experiments"), list) else []
        run_rows = data.get("runs") if isinstance(data.get("runs"), list) else []
        run_counts: dict[str, int] = {}
        for run in run_rows:
            if isinstance(run, Mapping):
                rid = _first(run, "experiment_id", default="")
                run_counts[str(rid)] = run_counts.get(str(rid), 0) + 1
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            eid = str(_first(exp, "id", default=""))
            summary = analyses.get(eid, {}) if isinstance(analyses, Mapping) else {}
            if isinstance(summary, Mapping):
                for source, target in (("mean", "mean"), ("median", "median"), ("std", "std"), ("confidence_interval", "confidence_interval"), ("success_rate", "success_rate")):
                    if target not in exp and source in summary:
                        exp[target] = summary[source]
            exp.setdefault("runs", run_counts.get(eid, 0))
            exp.setdefault("status", "Complete" if exp["runs"] else "Queued")
        if not data.get("evidence"):
            evidence: list[dict[str, Any]] = []
            for hypothesis in data.get("hypotheses", []) if isinstance(data.get("hypotheses"), list) else []:
                if not isinstance(hypothesis, Mapping):
                    continue
                for item in _records(hypothesis.get("evidence")):
                    item = dict(item)
                    item.setdefault("hypothesis_id", _first(hypothesis, "id", default=""))
                    evidence.append(item)
            data["evidence"] = evidence
        if "best_hypothesis" not in data:
            best_fn = getattr(self.engine, "best_hypothesis", None)
            if callable(best_fn):
                try:
                    data["best_hypothesis"] = _jsonable(best_fn())
                except Exception:
                    pass
        if "approval_mode" not in data:
            data["approval_mode"] = getattr(self.engine, "approval_mode", "Autonomous Sandbox")
        # Keep last known state if an engine's snapshot method is transient.
        self._local_snapshot = data
        return data

    def create_project(self, **kwargs: Any) -> Any:
        result = self._call(("create_project", "new_project", "start_project"), **kwargs)
        if result is None and isinstance(self.engine, LocalDemoEngine):
            result = self.engine.create_project(**kwargs)
        return result

    def run_demo(self) -> Any:
        result = self._call(("run_demo", "run_demo_study", "demo_study", "run_full_demo"))
        if result is None:
            # ScientificEngine exposes the same bounded loop as ``run_study``.
            result = self._call(("run_study",), first_round_runs=4, second_round_runs=3, second_round_count=2)
        return result

    def run_next(self) -> Any:
        result = self._call(("run_next_experiment", "run_next", "execute_next", "run_cycle", "step"))
        if result is not None:
            return result
        # Minimal compatibility path for ScientificEngine: lazily prepare a
        # design, select one unseen config, and execute one bounded run.
        self._call(("generate_hypotheses",))
        self._call(("design_experiments",))
        candidate = self._call(("select_next_experiment",))
        if candidate is None:
            return None
        result = self._call(("run_experiment",), candidate, runs=1, approval=self._approved)
        self._approved = False
        return result

    def run_all(self) -> Any:
        return self._call(("run_all", "run_until_done", "run_scientific_loop", "run_loop"))

    def generate_report(self) -> Any:
        result = self._call(("generate_report", "build_report", "report", "make_report"))
        if result is None:
            result = self._call(("generate_report_data",))
        return result

    def stop(self) -> Any:
        return self._call(("stop", "request_stop", "cancel", "cancel_all"))

    def approve(self) -> Any:
        self._approved = True
        return self._call(("approve", "approve_pending", "approve_next", "resume", "clear_stop"))

    def set_approval_mode(self, mode: str) -> Any:
        result = self._call(("set_approval_mode", "set_human_approval", "set_mode"), mode)
        if result is None:
            try:
                setattr(self.engine, "approval_mode", mode)
            except Exception:
                pass
        return result

    def save(self, path: str | os.PathLike[str]) -> Any:
        result = self._call(("save", "save_state", "save_project"), str(path))
        if result is None:
            Path(path).write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def load(self, path: str | os.PathLike[str]) -> Any:
        result = self._call(("load", "load_state", "load_project"), str(path))
        if result is None and isinstance(self.engine, LocalDemoEngine):
            self.engine.load(path)
        return result


def load_default_engine() -> Any:
    """Lazily discover a project engine, falling back to the local demo.

    The UI is intentionally not coupled to a module that may still be under
    development.  Once the core team exposes any of the conventional factory
    names below, a normal launch will pick it up automatically.  Import
    failures are swallowed only at this boundary; they never hide errors from
    an explicitly supplied engine.
    """
    candidates = (
        ("aiscientist.engine", ("ScientistEngine", "ResearchEngine", "AIScientist")),
        ("aiscientist.scientist", ("Scientist", "RuleBasedScientist")),
        ("aiscientist.core.engine", ("ScientificEngine", "ScientistEngine", "ResearchEngine")),
    )
    for module_name, class_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None or not callable(cls):
                continue
            for kwargs in ({}, {"offline": True}, {"mode": "offline"}):
                try:
                    instance = cls(**kwargs)
                    # Give a freshly constructed ScientificEngine the same
                    # meaningful question used by the built-in demo.  The
                    # user can replace it with New Project immediately.
                    project = getattr(instance, "project", None)
                    if project is not None and not str(getattr(project, "research_question", "") or "").strip():
                        creator = getattr(instance, "create_project", None)
                        if callable(creator):
                            try:
                                creator(research_question=DEFAULT_RESEARCH_QUESTION)
                            except Exception:
                                pass
                    return instance
                except TypeError:
                    continue
                except Exception:
                    break
    return LocalDemoEngine()


# ---------------------------------------------------------------------------
# A deterministic fallback engine.  It keeps the UI useful before the core
# package is installed and doubles as a smoke-test fixture.


class LocalDemoEngine:
    """Tiny deterministic in-process demo used only when no engine is passed."""

    STRATEGIES = ("No Memory", "Sliding Window", "Summary", "Episodic", "Vector")
    MEANS = {"No Memory": 0.43, "Sliding Window": 0.56, "Summary": 0.64, "Episodic": 0.72, "Vector": 0.79}

    def __init__(self) -> None:
        self.approval_mode = "Autonomous Sandbox"
        self._stop = False
        self.create_project(
            research_question="Which memory strategy is most suitable for a long-term agent?",
            scope="Synthetic benchmark; deterministic local data",
            constraints="No external APIs; no network; reproducible seeds",
            metrics="task_success, retention, latency",
            budget={"max_experiments": 10, "max_runs": 40, "max_runtime": 60, "estimated_api_cost": 0},
            max_experiments=10,
        )

    def create_project(self, **kwargs: Any) -> dict[str, Any]:
        q = kwargs.get("research_question") or kwargs.get("question") or "Untitled research question"
        budget = kwargs.get("budget") if isinstance(kwargs.get("budget"), Mapping) else {}
        self.project = {
            "research_question": q,
            "scope": kwargs.get("scope", ""),
            "constraints": kwargs.get("constraints", ""),
            "available_resources": kwargs.get("available_resources", "SyntheticProvider, PythonFunctionProvider"),
            "metrics": kwargs.get("metrics", "success_rate"),
            "budget": {
                "max_experiments": kwargs.get("max_experiments", budget.get("max_experiments", 10)),
                "max_runs": budget.get("max_runs", 40),
                "max_runtime": budget.get("max_runtime", 60),
                "estimated_api_cost": budget.get("estimated_api_cost", 0),
                "experiments_used": 0,
                "runs_used": 0,
                "runtime_used": 0.0,
                "cost_used": 0.0,
            },
            "status": "Active",
        }
        self.hypotheses = [
            {"id": f"H{i+1}", "statement": f"{s} will maximize long-term task success.", "rationale": "Candidate memory strategy", "predictions": {"strategy": s}, "confidence": 0.20, "status": "Untested", "evidence": []}
            for i, s in enumerate(self.STRATEGIES)
        ]
        self.experiments: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.queue: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.journal: list[dict[str, Any]] = [{"time": _dt.datetime.now().isoformat(timespec="seconds"), "event": "Project created", "summary": "Generated five candidate memory hypotheses."}]
        self.round = 0
        self._stop = False
        return self.snapshot()

    def _experiment(self, strategy: str, round_no: int, n: int = 4) -> dict[str, Any]:
        eid = f"EXP-{len(self.experiments)+1:03d}"
        seed = 1000 + len(self.experiments)
        base = self.MEANS[strategy]
        vals = [round(max(0.0, min(1.0, base + ((seed + i * 17) % 9 - 4) / 100)), 4) for i in range(n)]
        exp = {"id": eid, "hypothesis_id": next((h["id"] for h in self.hypotheses if h["predictions"]["strategy"] == strategy), ""), "name": f"{strategy} benchmark R{round_no}", "strategy": strategy, "round": round_no, "status": "Complete", "config": {"strategy": strategy, "sample_size": n, "seed": seed, "metric": "success_rate"}, "runs": n, "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4), "std": round(statistics.stdev(vals), 4) if n > 1 else 0.0, "success_rate": round(statistics.mean(vals), 4), "values": vals, "control": "No Memory" if strategy != "No Memory" else "None", "created_at": _dt.datetime.now().isoformat(timespec="seconds")}
        for i, val in enumerate(vals):
            self.runs.append({"id": f"{eid}-RUN-{i+1}", "experiment_id": eid, "seed": seed + i, "status": "Complete", "metric": "success_rate", "value": val})
        return exp

    def run_next(self) -> dict[str, Any] | None:
        if self._stop:
            return None
        b = self.project["budget"]
        if b["experiments_used"] >= b["max_experiments"]:
            self.project["status"] = "Budget Exhausted"
            return None
        # First round explores all candidates. Second round follows the best.
        if self.round == 0:
            strategy = self.STRATEGIES[len(self.experiments)]
            round_no = 1
            if len(self.experiments) + 1 >= len(self.STRATEGIES):
                self.round = 1
        else:
            ranked = sorted(self.hypotheses, key=lambda h: h["confidence"], reverse=True)
            strategy = ranked[0]["predictions"]["strategy"]
            round_no = 2
        exp = self._experiment(strategy, round_no, 4 if round_no == 1 else 6)
        self.experiments.append(exp)
        b["experiments_used"] += 1
        b["runs_used"] += exp["runs"]
        b["runtime_used"] = round(b["runtime_used"] + 0.01, 3)
        for h in self.hypotheses:
            if h["predictions"]["strategy"] == strategy:
                old = h["confidence"]
                h["confidence"] = round(min(0.99, old * 0.55 + exp["mean"] * 0.45), 4)
                h["status"] = "Supported" if exp["mean"] >= 0.68 else ("Inconclusive" if exp["mean"] >= 0.5 else "Rejected")
                h["evidence"].append(exp["id"])
                self.evidence.append({"id": f"E-{len(self.evidence)+1:03d}", "hypothesis_id": h["id"], "experiment_id": exp["id"], "result": exp["mean"], "rule": "confidence = 0.55*prior + 0.45*observed_mean; status threshold 0.68/0.50"})
                break
        self.journal.append({"time": _dt.datetime.now().isoformat(timespec="seconds"), "event": "Experiment complete", "summary": f"Ran {strategy} with seed {exp['config']['seed']}; mean success rate {exp['mean']:.3f}."})
        return exp

    def run_demo(self) -> dict[str, Any]:
        self._stop = False
        # A demo always gives five exploration experiments and one follow-up.
        while len(self.experiments) < 5 and not self._stop:
            self.run_next()
        if not self._stop:
            self.run_next()
        self.project["status"] = "Complete" if not self._stop else "Stopped"
        self.journal.append({"time": _dt.datetime.now().isoformat(timespec="seconds"), "event": "Demo complete", "summary": "Two-round Memory Strategy Study finished."})
        return self.snapshot()

    def run_all(self) -> dict[str, Any]:
        return self.run_demo()

    def generate_report(self) -> str:
        best = max(self.hypotheses, key=lambda h: h["confidence"], default=None)
        lines = [f"# AI Scientist Mini Research Report", "", f"## Research Question\n{self.project['research_question']}", "", "## Methods", "Rule-Based Scientist with SyntheticProvider; deterministic seeds; no paid API calls.", "", "## Experiments", f"Completed {len(self.experiments)} experiments and {len(self.runs)} runs.", "", "| Strategy | Round | Mean | Status |", "|---|---:|---:|---|"]
        for exp in self.experiments:
            h = next((h for h in self.hypotheses if h["id"] == exp.get("hypothesis_id")), {})
            lines.append(f"| {exp['strategy']} | {exp['round']} | {exp['mean']:.3f} | {h.get('status','')} |")
        if best:
            lines += ["", "## Best Hypothesis", f"**{best['id']}** — {best['statement']}", f"Confidence: {best['confidence']:.3f}; status: {best['status']}"]
        lines += ["", "## Limitations", "Synthetic data and a transparent heuristic do not establish a universal scientific truth.", "", "## Next Work", "Repeat with an approved external provider and a preregistered dataset."]
        return "\n".join(lines) + "\n"

    def stop(self) -> None:
        self._stop = True
        self.project["status"] = "Stopped"

    def approve(self) -> None:
        self.approval_mode = "Autonomous Sandbox"

    def set_approval_mode(self, mode: str) -> None:
        self.approval_mode = mode

    def snapshot(self) -> dict[str, Any]:
        best = max(self.hypotheses, key=lambda h: h["confidence"], default=None)
        next_strategy = self.STRATEGIES[len(self.experiments)] if len(self.experiments) < 5 else (best or {}).get("predictions", {}).get("strategy", "")
        return {"project": self.project, "hypotheses": self.hypotheses, "experiments": self.experiments, "runs": self.runs, "queue": self.queue, "evidence": self.evidence, "journal": self.journal, "best_hypothesis": best, "next_experiment": {"strategy": next_strategy, "round": 1 if len(self.experiments) < 5 else 2}, "approval_mode": self.approval_mode}

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | os.PathLike[str]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, val in data.items():
            if key != "best_hypothesis":
                setattr(self, key, val)


# ---------------------------------------------------------------------------
# Tkinter dashboard


class DashboardApp:
    """Desktop dashboard.  Construct lazily with :func:`create_app`."""

    def __init__(self, root: Any, engine: Any | None = None):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is unavailable in this Python installation")
        self.root = root
        self.adapter = engine if isinstance(engine, EngineAdapter) else EngineAdapter(engine if engine is not None else load_default_engine())
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.snapshot_data: dict[str, Any] = {}
        self.report_text = ""
        self._build_style()
        self._build_layout()
        self.refresh()
        self.root.after(200, self._poll_worker)

    # -- construction -----------------------------------------------------
    def _build_style(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("1380x860")
        self.root.minsize(980, 620)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Subtle.TLabel", foreground="#667085")
        style.configure("Metric.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Treeview", rowheight=25)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="Rule-Based Scientist · Mock/Synthetic only", style="Subtle.TLabel").pack(side="left", padx=(15, 0))
        self.mode_var = tk.StringVar(value="Autonomous Sandbox")
        ttk.Label(top, text="Mode").pack(side="right", padx=(8, 3))
        mode = ttk.Combobox(top, textvariable=self.mode_var, state="readonly", width=20, values=("Autonomous Sandbox", "Human Approval"))
        mode.pack(side="right")
        mode.bind("<<ComboboxSelected>>", lambda _e: self._set_mode())

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Research question").pack(side="left")
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(controls, textvariable=self.question_var)
        self.question_entry.pack(side="left", fill="x", expand=True, padx=8)
        self._button(controls, "New Project", self.new_project).pack(side="left", padx=2)
        self._button(controls, "Run Demo", self.run_demo).pack(side="left", padx=2)
        self._button(controls, "Run Next", self.run_next).pack(side="left", padx=2)
        self._button(controls, "Approve", self.approve, accent=True).pack(side="left", padx=2)
        self._button(controls, "Stop", self.stop, danger=True).pack(side="left", padx=2)
        self._button(controls, "Report", self.generate_report).pack(side="left", padx=2)
        self._button(controls, "Save", self.save_state).pack(side="left", padx=2)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=1)
        body.add(right, weight=3)

        # Summary and budget cards.
        summary = ttk.LabelFrame(left, text="Project / Budget", padding=8)
        summary.pack(fill="x", pady=(0, 8))
        self.summary_text = tk.Text(summary, height=9, wrap="word", relief="flat", background="#f7f8fa")
        self.summary_text.pack(fill="x")
        self._readonly(self.summary_text)

        hyp_frame = ttk.LabelFrame(left, text="Active Hypotheses", padding=4)
        hyp_frame.pack(fill="both", expand=True)
        self.hyp_tree = self._tree(hyp_frame, ("id", "statement", "confidence", "status"), (70, 255, 80, 100))
        self.hyp_tree.pack(fill="both", expand=True)
        self.hyp_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected_hypothesis())

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self.experiment_tab = ttk.Frame(self.tabs, padding=4)
        self.queue_tab = ttk.Frame(self.tabs, padding=4)
        self.evidence_tab = ttk.Frame(self.tabs, padding=4)
        self.journal_tab = ttk.Frame(self.tabs, padding=4)
        self.chart_tab = ttk.Frame(self.tabs, padding=4)
        self.graph_tab = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(self.experiment_tab, text="Experiments")
        self.tabs.add(self.queue_tab, text="Queue")
        self.tabs.add(self.evidence_tab, text="Evidence")
        self.tabs.add(self.journal_tab, text="Journal")
        self.tabs.add(self.chart_tab, text="Charts")
        self.tabs.add(self.graph_tab, text="Hypothesis Graph")

        self.exp_tree = self._tree(self.experiment_tab, ("id", "name", "round", "status", "mean", "runs"), (90, 250, 65, 100, 90, 65))
        self.exp_tree.pack(fill="both", expand=True)
        self.queue_tree = self._tree(self.queue_tab, ("id", "status", "type", "reason"), (120, 100, 150, 500))
        self.queue_tree.pack(fill="both", expand=True)
        self.evidence_tree = self._tree(self.evidence_tab, ("id", "hypothesis", "experiment", "result", "rule"), (90, 100, 110, 90, 550))
        self.evidence_tree.pack(fill="both", expand=True)
        self.journal_text = tk.Text(self.journal_tab, wrap="word", relief="flat")
        self.journal_text.pack(fill="both", expand=True)
        self._readonly(self.journal_text)
        self.chart_canvas = tk.Canvas(self.chart_tab, background="white", highlightthickness=0)
        self.chart_canvas.pack(fill="both", expand=True)
        self.chart_canvas.bind("<Configure>", lambda _e: self._draw_chart())
        self.graph_canvas = tk.Canvas(self.graph_tab, background="white", highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True)
        self.graph_canvas.bind("<Configure>", lambda _e: self._draw_graph())

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, style="Subtle.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    @staticmethod
    def _button(parent: Any, text: str, command: Callable[[], None], accent: bool = False, danger: bool = False) -> Any:
        b = ttk.Button(parent, text=text, command=command)
        return b

    @staticmethod
    def _readonly(widget: Any) -> None:
        widget.configure(state="disabled")

    @staticmethod
    def _tree(parent: Any, columns: tuple[str, ...], widths: tuple[int, ...]) -> Any:
        # Return a widget whose parent is the caller's frame.  Earlier
        # versions wrapped the tree in an un-packed frame, making all tables
        # silently disappear; the dashboard intentionally keeps the simple
        # direct-parent contract so callers can use ``tree.pack(...)``.
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        for col, width in zip(columns, widths):
            tree.heading(col, text=col.replace("_", " ").title())
            tree.column(col, width=width, minwidth=45, anchor="w")
        return tree

    # -- state and rendering ---------------------------------------------
    def refresh(self) -> None:
        self.snapshot_data = self.adapter.snapshot()
        project = self.snapshot_data.get("project") if isinstance(self.snapshot_data.get("project"), Mapping) else self.snapshot_data
        if not isinstance(project, Mapping):
            project = {}
        q = _first(project, "research_question", "question", default="")
        self.question_var.set(str(q or ""))
        self._render_summary(project)
        self._fill_hypotheses(_records(self.snapshot_data.get("hypotheses")))
        self._fill_experiments(_records(self.snapshot_data.get("experiments")))
        self._fill_queue(_records(self.snapshot_data.get("queue")))
        self._fill_evidence(_records(self.snapshot_data.get("evidence")))
        self._render_journal(_records(self.snapshot_data.get("journal")))
        self._draw_chart()
        self._draw_graph()

    def _render_summary(self, project: Mapping[str, Any]) -> None:
        budget = _first(project, "budget", default={})
        if not isinstance(budget, Mapping):
            budget = {}
        best = self.snapshot_data.get("best_hypothesis") or {}
        if not isinstance(best, Mapping):
            best = {}
        nxt = self.snapshot_data.get("next_experiment") or {}
        if not isinstance(nxt, Mapping):
            nxt = {}
        lines = [
            f"Question: {_first(project, 'research_question', 'question', default='—')}",
            f"Status: {_first(project, 'status', default='—')}",
            f"Experiments: {_first(budget, 'experiments_used', 'used_experiments', default=len(_records(self.snapshot_data.get('experiments'))))} / {_first(budget, 'max_experiments', default='—')}",
            f"Runs: {_first(budget, 'runs_used', 'used_runs', default=len(_records(self.snapshot_data.get('runs'))))} / {_first(budget, 'max_runs', default='—')}",
            f"Runtime: {_first(budget, 'runtime_used', 'runtime_seconds', default=0)} / {_first(budget, 'max_runtime', 'max_runtime_seconds', default='—')} s",
            f"Estimated API cost: ${_first(budget, 'cost_used', 'spent_api_cost', default=0)} / ${_first(budget, 'estimated_api_cost', default=0)}",
            f"Best: {_first(best, 'id', default='—')} {_first(best, 'status', default='')}",
            f"Next: {_first(nxt, 'strategy', 'name', default='—')} (round {_first(nxt, 'round', default='—')})",
        ]
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self._readonly(self.summary_text)

    def _clear_tree(self, tree: Any) -> None:
        for item in tree.get_children(""):
            tree.delete(item)

    def _fill_hypotheses(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.hyp_tree)
        for row in rows:
            conf = _first(row, "confidence", default="")
            try:
                conf = f"{float(conf):.3f}"
            except (ValueError, TypeError):
                pass
            self.hyp_tree.insert("", "end", values=(_first(row, "id", default=""), _first(row, "statement", "name", default=""), conf, _first(row, "status", default="")))

    def _fill_experiments(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.exp_tree)
        for row in rows:
            self.exp_tree.insert("", "end", values=(_first(row, "id", default=""), _first(row, "name", "strategy", default=""), _first(row, "round", default=""), _first(row, "status", default=""), _first(row, "mean", "success_rate", default=""), _first(row, "runs", default="")))

    def _fill_queue(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.queue_tree)
        for row in rows:
            self.queue_tree.insert("", "end", values=(_first(row, "id", default=""), _first(row, "status", default=""), _first(row, "type", "experiment_type", default=""), _first(row, "reason", "error", default="")))

    def _fill_evidence(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.evidence_tree)
        for row in rows:
            self.evidence_tree.insert("", "end", values=(_first(row, "id", default=""), _first(row, "hypothesis_id", "hypothesis", default=""), _first(row, "experiment_id", "experiment", default=""), _first(row, "result", "value", default=""), _first(row, "rule", "reason", default="")))

    def _render_journal(self, rows: list[dict[str, Any]]) -> None:
        self.journal_text.configure(state="normal")
        self.journal_text.delete("1.0", "end")
        for row in rows:
            stamp = _first(row, "time", "timestamp", "created_at", default="")
            event = _first(row, "event", "title", default="Journal")
            summary = _first(row, "summary", "message", "text", default="")
            self.journal_text.insert("end", f"[{stamp}] {event}\n{summary}\n\n")
        self._readonly(self.journal_text)

    def _draw_chart(self) -> None:
        if not hasattr(self, "chart_canvas"):
            return
        c = self.chart_canvas
        c.delete("all")
        rows = _records(self.snapshot_data.get("experiments"))
        grouped: dict[str, list[float]] = {}
        for row in rows:
            name = str(_first(row, "strategy", "name", default="Unknown"))
            try:
                val = float(_first(row, "mean", "success_rate", "value", default=0))
            except (TypeError, ValueError):
                val = 0.0
            grouped.setdefault(name, []).append(val)
        width = max(500, c.winfo_width())
        height = max(300, c.winfo_height())
        if not grouped:
            c.create_text(width / 2, height / 2, text="Run an experiment to plot evidence", fill="#667085", font=("Segoe UI", 13))
            return
        c.create_text(18, 16, anchor="w", text="Observed mean metric by strategy", fill="#1d2939", font=("Segoe UI", 12, "bold"))
        x0, y0, x1, y1 = 65, 48, width - 30, height - 55
        c.create_line(x0, y1, x1, y1, fill="#98a2b3")
        c.create_line(x0, y0, x0, y1, fill="#98a2b3")
        for tick in range(0, 6):
            y = y1 - (y1 - y0) * tick / 5
            c.create_line(x0, y, x1, y, fill="#eaecf0")
            c.create_text(x0 - 8, y, text=f"{tick/5:.1f}", anchor="e", fill="#667085")
        names = list(grouped)
        bw = max(24, (x1 - x0) / max(1, len(names)) * 0.58)
        for i, name in enumerate(names):
            val = statistics.mean(grouped[name])
            cx = x0 + (i + 0.5) * (x1 - x0) / len(names)
            top = y1 - (y1 - y0) * max(0.0, min(1.0, val))
            c.create_rectangle(cx - bw / 2, top, cx + bw / 2, y1, fill="#6172f3", outline="")
            c.create_text(cx, top - 8, text=f"{val:.2f}", fill="#344054")
            c.create_text(cx, y1 + 15, text=name, angle=25, anchor="w", fill="#475467")

    def _draw_graph(self) -> None:
        if not hasattr(self, "graph_canvas"):
            return
        c = self.graph_canvas
        c.delete("all")
        width, height = max(700, c.winfo_width()), max(360, c.winfo_height())
        question = _first(self.snapshot_data.get("project") if isinstance(self.snapshot_data.get("project"), Mapping) else {}, "research_question", "question", default="Question")
        hyps = _records(self.snapshot_data.get("hypotheses"))
        exps = _records(self.snapshot_data.get("experiments"))
        # Keep the graph readable: question -> up to 5 hypotheses -> latest
        # experiments/evidence.  Full details remain in the tables.
        nodes: list[tuple[str, str, float, float, str]] = [("Q", str(question), 100, height / 2, "#dbeafe")]
        for i, h in enumerate(hyps[:5]):
            nodes.append((str(_first(h, "id", default=f"H{i+1}")), str(_first(h, "statement", default="")), 290, 70 + i * max(55, (height - 130) / max(1, min(5, len(hyps)))), "#dcfce7"))
        for i, e in enumerate(exps[-5:]):
            nodes.append((str(_first(e, "id", default=f"E{i+1}")), f"{_first(e, 'strategy', 'name', default='Experiment')}\nmean {_first(e, 'mean', 'success_rate', default='')}", 560, 80 + i * max(50, (height - 120) / max(1, min(5, len(exps)))), "#fef3c7"))
        for idx in range(1, len(nodes)):
            parent_x, parent_y = nodes[0][2], nodes[0][3]
            if idx > 1:
                parent_x, parent_y = nodes[1 + min(4, (idx - 1) % max(1, min(5, len(hyps))))][2], nodes[1 + min(4, (idx - 1) % max(1, min(5, len(hyps))))][3]
            c.create_line(parent_x + 65, parent_y, nodes[idx][2] - 65, nodes[idx][3], fill="#98a2b3", arrow="last")
        for ident, label, x, y, color in nodes:
            c.create_oval(x - 65, y - 25, x + 65, y + 25, fill=color, outline="#98a2b3")
            short = label if len(label) <= 28 else label[:26] + "…"
            c.create_text(x, y, text=f"{ident}\n{short}", width=120, justify="center", fill="#1d2939")

    def _show_selected_hypothesis(self) -> None:
        # Keep this hook intentionally lightweight; selection can later drive a
        # details pane without changing the core dashboard contract.
        return None

    # -- actions ----------------------------------------------------------
    def _set_mode(self) -> None:
        mode = self.mode_var.get()
        self.adapter.set_approval_mode(mode)
        self.status_var.set(f"Mode set to {mode}")

    def new_project(self) -> None:
        question = self.question_var.get().strip()
        if not question:
            self.status_var.set("Enter a research question first")
            return
        self.adapter.create_project(research_question=question, scope="User-defined scope", constraints="No paid APIs; bounded synthetic experiments", metrics="success_rate", max_experiments=10, budget={"max_experiments": 10, "max_runs": 40, "max_runtime": 60, "estimated_api_cost": 0})
        self.refresh()
        self.status_var.set("New research project created")

    def run_demo(self) -> None:
        self._start_worker("Running two-round demo study…", self.adapter.run_demo)

    def run_next(self) -> None:
        self._start_worker("Running next experiment…", self.adapter.run_next)

    def approve(self) -> None:
        self.adapter.approve()
        self.status_var.set("Pending external action approved (Mock/Synthetic remains local)")

    def stop(self) -> None:
        self.stop_event.set()
        self.adapter.stop()
        self.status_var.set("Stop requested")

    def generate_report(self) -> None:
        try:
            report = self.adapter.generate_report()
            if isinstance(report, Mapping):
                report = _first(report, "markdown", "text", "report", default=json.dumps(report, ensure_ascii=False, indent=2))
            if report is None:
                report = "No report returned by the engine."
            self.report_text = str(report)
            self._show_report_window()
            self.status_var.set("Report generated")
        except Exception as exc:
            self.status_var.set(f"Report failed: {exc}")

    def _show_report_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Research Report")
        win.geometry("900x700")
        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("1.0", self.report_text)
        ttk.Button(win, text="Save Markdown…", command=lambda: self._save_report(text.get("1.0", "end"))).pack(pady=(0, 8))

    def _save_report(self, report: str) -> None:
        if filedialog is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=(("Markdown", "*.md"), ("Text", "*.txt")))
        if path:
            Path(path).write_text(report, encoding="utf-8")
            self.status_var.set(f"Report saved: {path}")

    def save_state(self) -> None:
        if filedialog is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if path:
            try:
                self.adapter.save(path)
                self.status_var.set(f"State saved: {path}")
            except Exception as exc:
                self.status_var.set(f"Save failed: {exc}")

    def _start_worker(self, label: str, fn: Callable[[], Any]) -> None:
        if self.worker and self.worker.is_alive():
            self.status_var.set("An operation is already running")
            return
        self.stop_event.clear()
        self.status_var.set(label)
        self.progress.start(10)

        def run() -> None:
            try:
                result = fn()
                self.ui_queue.put(("done", result))
            except Exception:
                self.ui_queue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=run, name="aiscientist-ui-worker", daemon=True)
        self.worker.start()

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                self.progress.stop()
                if kind == "done":
                    self.refresh()
                    self.status_var.set("Operation complete")
                else:
                    self.status_var.set("Operation failed; see console for details")
                    print(payload, file=sys.stderr)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_worker)


def create_app(root: Any | None = None, engine: Any | None = None) -> DashboardApp:
    """Create and return a dashboard without entering ``mainloop``."""
    if tk is None:
        raise RuntimeError("Tkinter is unavailable")
    root = root or tk.Tk()
    return DashboardApp(root, engine if engine is not None else load_default_engine())


def headless_snapshot(engine: Any | None = None, run_demo: bool = True) -> dict[str, Any]:
    """Run a deterministic smoke cycle without a display server."""
    adapter = engine if isinstance(engine, EngineAdapter) else EngineAdapter(engine if engine is not None else load_default_engine())
    if run_demo:
        adapter.run_demo()
    return adapter.snapshot()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AI Scientist Mini dashboard")
    parser.add_argument("--headless", action="store_true", help="run the local synthetic smoke cycle and print JSON")
    parser.add_argument("--no-demo", action="store_true", help="do not run demo in headless mode")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.headless:
        print(json.dumps(headless_snapshot(run_demo=not args.no_demo), ensure_ascii=False, indent=2))
        return 0
    if tk is None:
        print("Tkinter is unavailable; use --headless for a smoke run.", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"Cannot open dashboard ({exc}); use --headless for a smoke run.", file=sys.stderr)
        return 2
    create_app(root, load_default_engine())
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
