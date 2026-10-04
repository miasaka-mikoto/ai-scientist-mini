"""Built-in offline Memory Strategy Study.

The demo is intentionally a real invocation of the same engine used by the
UI: five hypotheses/configurations, four fixed-seed runs each in round one,
then two uncertainty-driven follow-ups in round two.  It therefore exercises
the complete queue → provider → analysis → hypothesis update → report path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import ScientificEngine
from .models import Budget, Project
from .providers import SyntheticProvider
from .store import ResearchStore


def run_memory_strategy_study(
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    first_round_runs: int = 4,
    second_round_runs: int = 3,
    second_round_count: int = 2,
) -> dict[str, Any]:
    """Run and return the bounded offline Memory Strategy Study.

    Parameters are deliberately explicit so tests and teaching notebooks can
    lower run counts while the default meets the product acceptance target of
    20+ runs and two selection rounds.  Supplying ``db_path`` makes the study
    resumable across process restarts.
    """

    store = ResearchStore(db_path)
    project = Project(
        research_question="Which memory strategy is more suitable for a long-term Agent?",
        scope="Offline synthetic benchmark only; no external model or network.",
        constraints=["No paid API calls", "Fixed seeds", "Public decision summaries only"],
        available_resources={"sample_size": 24, "provider": "SyntheticProvider"},
        metrics=["score", "retention", "latency"],
        budget=Budget(max_experiments=20, max_runs=100, max_runtime_seconds=300, estimated_api_cost=0.0),
    )
    # If a persistent database already contains a project, use its ID/state so
    # rerunning this helper resumes rather than duplicating all experiments.
    existing = store.list_projects()
    if existing:
        project = next((p for p in existing if p.research_question == project.research_question), project)
    engine = ScientificEngine(project, provider=SyntheticProvider(), store=store)
    report = engine.run_study(first_round_runs=first_round_runs, second_round_runs=second_round_runs, second_round_count=second_round_count)
    report["demo"] = {
        "name": "Memory Strategy Study",
        "target_hypotheses": 5,
        "target_configurations": 5,
        "target_runs": 20,
        "provider": "SyntheticProvider",
        "cost": 0.0,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "memory_strategy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "memory_strategy_snapshot.json").write_text(json.dumps(engine.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        # Reporting stays an optional outer layer of the core engine.  Import
        # locally to avoid a core→UI dependency and keep headless execution
        # free of chart servers or network access.
        from aiscientist.reporting import generate_report, save_hypothesis_graph, save_report
        from aiscientist.visualization import save_charts

        rendered = generate_report(report)
        report_context = rendered.data
        chart_paths = save_charts(
            report_context.get("analysis", {}),
            destination,
            metric="score",
            group_field="group",
            prefix="memory_strategy",
        )
        report_context["charts"] = chart_paths
        save_report(report_context, destination / "research_report.md", format="markdown")
        save_report(report_context, destination / "research_report.html", format="html")
        save_report(report_context, destination / "research_report.pdf", format="pdf")
        graph = report_context.get("graph", {})
        save_hypothesis_graph(graph, destination / "hypothesis_graph.mmd", format="mermaid")
        (destination / "analysis_summary.json").write_text(
            json.dumps(report_context.get("analysis", {}), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        journal = report_context.get("journal", {})
        (destination / "scientist_journal.json").write_text(
            json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["artifacts"] = {
            "report_markdown": str(destination / "research_report.md"),
            "report_html": str(destination / "research_report.html"),
            "report_pdf": str(destination / "research_report.pdf"),
            "charts": chart_paths,
            "hypothesis_graph": str(destination / "hypothesis_graph.mmd"),
            "journal": str(destination / "scientist_journal.json"),
        }
        # Rewrite the structured report after artifact paths are known.  The
        # first write above is useful as an early checkpoint, while this final
        # write makes the on-disk JSON self-describing for reopen/inspection.
        (destination / "memory_strategy_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # The helper returns JSON-compatible data, not a live engine.  Close the
    # SQLite connection so file-backed demos checkpoint cleanly and can be
    # reopened by a second process without a lingering WAL writer.
    store.close()
    return report


run_demo_study = run_memory_strategy_study


__all__ = ["run_memory_strategy_study", "run_demo_study"]
