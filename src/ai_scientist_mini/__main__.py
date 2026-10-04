"""Unified local CLI for AI Scientist Mini.

The command line is intentionally small and offline-first.  It exposes the
same bounded scientific loop used by the desktop dashboard and keeps the
integration boundary under ``ai_scientist_mini.integrations``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .demo_study import (
    DemoConfiguration,
    DemoExperiment,
    DemoHypothesis,
    DemoRun,
    DemoStudy,
    run_demo_study,
)
from .engine import ScientistEngine
from .models import ApprovalMode, ResearchBudget, to_dict
from .providers import MockLLMExperimentProvider, SyntheticProvider
from .reporting import generate_demo_artifacts, write_html_report, write_report


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _demo_from_dict(value: dict[str, Any]) -> DemoStudy:
    """Rehydrate a persisted demo JSON file without a database dependency."""

    def make(cls: type[Any], item: dict[str, Any]) -> Any:
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: val for key, val in item.items() if key in allowed})

    return DemoStudy(
        **{
            key: value[key]
            for key in (
                "id",
                "name",
                "research_question",
                "background",
                "scope",
                "constraints",
                "available_resources",
                "metrics",
                "budget",
                "created_at",
                "completed_at",
                "code_version",
            )
            if key in value
        },
        hypotheses=[make(DemoHypothesis, item) for item in value.get("hypotheses", [])],
        configurations=[make(DemoConfiguration, item) for item in value.get("configurations", [])],
        experiments=[make(DemoExperiment, item) for item in value.get("experiments", [])],
        runs=[make(DemoRun, item) for item in value.get("runs", [])],
        decisions=list(value.get("decisions", [])),
        journal=list(value.get("journal", [])),
    )


def _load_demo(path: Path) -> DemoStudy:
    candidate = path
    if candidate.is_dir():
        candidate = candidate / "memory_strategy_study.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Demo study file not found: {candidate}")
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "hypotheses" not in value or "configurations" not in value:
        raise ValueError(f"Not a persisted Memory Strategy Study: {candidate}")
    return _demo_from_dict(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiscientist-mini",
        description="AI Scientist Mini — local, auditable, budget-controlled experiments",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run a deterministic local demo study")
    demo.add_argument("study", nargs="?", default="memory-strategy", choices=("memory-strategy",))
    demo.add_argument("--data-dir", "--output", dest="data_dir", type=Path, default=Path("artifacts/memory_strategy"))
    demo.add_argument("--seed", type=int, default=20261004)

    smoke = sub.add_parser("smoke-test", help="run the no-window dashboard smoke test")
    smoke.add_argument("--json", action="store_true", dest="as_json")

    serve = sub.add_parser("serve", help="open the local desktop dashboard")
    serve.add_argument("--demo", action="store_true", help="start with the Memory Strategy Study")
    serve.add_argument("--state", type=Path, help="optional JSON state path")

    create = sub.add_parser("create", help="create and persist a research project")
    create.add_argument("research_question")
    create.add_argument("--state", type=Path, default=Path("artifacts/project.json"))
    create.add_argument("--name", default="AI Scientist Study")
    create.add_argument("--scope", default="")
    create.add_argument("--max-experiments", type=int, default=20)
    create.add_argument("--max-runs", type=int, default=100)
    create.add_argument("--max-runtime", type=float, default=3600.0)
    create.add_argument("--estimated-cost", type=float, default=0.0)
    create.add_argument("--approval-mode", choices=[mode.value for mode in ApprovalMode], default=ApprovalMode.AUTONOMOUS_SANDBOX.value)

    run = sub.add_parser("run", help="continue a persisted project through the scientific loop")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--rounds", type=int, default=2)
    run.add_argument("--runs-per-experiment", type=int, default=5)
    run.add_argument("--provider", choices=("synthetic", "mock-llm"), default="synthetic")
    run.add_argument("--approved", action="store_true", help="approve an external provider call (local providers remain zero-cost)")

    show = sub.add_parser("show", help="print a persisted project state")
    show.add_argument("state", type=Path)

    report = sub.add_parser("report", help="generate a report from a demo study")
    report.add_argument("--project", type=Path, help="demo JSON file or directory")
    report.add_argument("--out", type=Path, default=Path("artifacts/report"))
    report.add_argument("--seed", type=int, default=20261004)

    integrations = sub.add_parser("integrations", help="invoke the versioned adapter boundary")
    integrations.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _command_demo(args: argparse.Namespace) -> int:
    output = args.data_dir
    paths = generate_demo_artifacts(output, seed=args.seed)
    study = run_demo_study(seed=args.seed)
    summary = {
        "study": study.name,
        "study_id": study.id,
        "hypotheses": len(study.hypotheses),
        "configurations": len(study.configurations),
        "experiments": len(study.experiments),
        "runs": len(study.runs),
        "rounds": sorted({experiment.round for experiment in study.experiments}),
        "estimated_api_cost": study.budget["estimated_api_cost"],
        "artifacts": paths,
    }
    print(_json(summary))
    return 0


def _command_smoke(args: argparse.Namespace) -> int:
    from .ui.app import run_smoke_test

    value = run_smoke_test()
    if args.as_json:
        print(_json(value))
    else:
        print("AI Scientist Mini UI smoke test:", "PASS" if value.get("passed") else "FAIL")
        for key, item in value.get("checks", {}).items():
            print(f"  {'✓' if item else '✗'} {key}")
    return 0 if value.get("passed") else 1


def _command_serve(args: argparse.Namespace) -> int:
    from .ui.app import main as ui_main

    argv: list[str] = []
    if args.demo:
        argv.append("--demo")
    if args.state:
        argv.extend(["--state", str(args.state)])
    return int(ui_main(argv))


def _command_create(args: argparse.Namespace) -> int:
    budget = ResearchBudget(
        maximum_experiments=max(0, args.max_experiments),
        maximum_runs=max(0, args.max_runs),
        maximum_runtime_seconds=max(0.0, args.max_runtime),
        estimated_api_cost=max(0.0, args.estimated_cost),
    )
    engine = ScientistEngine()
    project = engine.create_project(
        args.research_question,
        name=args.name,
        scope=args.scope,
        budget=budget,
        max_experiments=args.max_experiments,
        approval_mode=args.approval_mode,
    )
    path = engine.save_state(args.state)
    print(_json({"state": str(path), "project": to_dict(project)}))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    provider = MockLLMExperimentProvider() if args.provider == "mock-llm" else SyntheticProvider()
    engine = ScientistEngine.load_state(args.state, provider=provider)
    experiments = engine.run_scientific_cycle(
        rounds=max(1, args.rounds),
        runs_per_experiment=max(1, args.runs_per_experiment),
        approved=bool(args.approved),
    )
    path = engine.save_state(args.state)
    project = engine.require_project()
    print(_json({
        "state": str(path),
        "project_id": project.id,
        "status": project.status,
        "experiments_run": len(experiments),
        "hypotheses": len(project.hypotheses),
        "evidence": len(project.evidence),
        "runs": sum(len(experiment.runs) for experiment in project.experiments),
        "budget": to_dict(project.budget),
    }))
    return 0


def _command_show(args: argparse.Namespace) -> int:
    engine = ScientistEngine.load_state(args.state)
    print(_json(to_dict(engine.require_project())))
    return 0


def _command_report(args: argparse.Namespace) -> int:
    if args.project:
        study = _load_demo(args.project)
        args.out.mkdir(parents=True, exist_ok=True)
        md = write_report(study, args.out / "research_report.md")
        html = write_html_report(study, args.out / "research_report.html")
        print(_json({"report": str(md), "html_report": str(html)}))
        return 0
    paths = generate_demo_artifacts(args.out, seed=args.seed)
    print(_json(paths))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.command:
        _build_parser().print_help()
        return 0
    try:
        if args.command == "demo":
            return _command_demo(args)
        if args.command == "smoke-test":
            return _command_smoke(args)
        if args.command == "serve":
            return _command_serve(args)
        if args.command == "create":
            return _command_create(args)
        if args.command == "run":
            return _command_run(args)
        if args.command == "show":
            return _command_show(args)
        if args.command == "report":
            return _command_report(args)
        if args.command == "integrations":
            from .integrations.cli import main as integration_main

            return int(integration_main(args.args))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
