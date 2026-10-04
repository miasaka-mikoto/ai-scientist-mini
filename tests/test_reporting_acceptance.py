from __future__ import annotations

import json
from pathlib import Path

from aiscientist.core.models import ExperimentConfig, Hypothesis, Project
from aiscientist.reporting import (
    ScientistJournal,
    build_demo_report,
    build_hypothesis_graph,
    generate_report,
    hypothesis_graph_to_mermaid,
    save_report,
)


def test_builtin_demo_report_has_five_hypotheses_and_at_least_20_runs() -> None:
    demo = build_demo_report()
    assert len(demo["hypotheses"]) >= 5
    assert len(demo["experiments"]) >= 5
    assert len(demo["results"]) >= 20
    assert demo["analysis"]["groups"]
    assert demo["graph"]["nodes"] and demo["graph"]["edges"]


def test_hypothesis_graph_contains_full_scientific_trace() -> None:
    project = Project("Which strategy is best?")
    h = Hypothesis("Episodic wins")
    e = ExperimentConfig("Episodic", project_id=project.id, hypothesis_id=h.id)
    result = {
        "run_id": "run-1",
        "experiment_id": e.id,
        "metrics": {"score": 0.9},
        "success": True,
    }
    graph = build_hypothesis_graph(project, [h], [e], [result])
    types = {node["type"] for node in graph["nodes"]}
    assert {"question", "hypothesis", "experiment", "result", "evidence"} <= types
    mermaid = hypothesis_graph_to_mermaid(graph)
    assert "flowchart" in mermaid
    assert "proposes" in mermaid and "produces" in mermaid


def test_report_is_public_and_excludes_private_chain_of_thought(tmp_path: Path) -> None:
    project = Project("Does summary improve retention?")
    h = Hypothesis("Summary improves retention", confidence=0.75)
    e = ExperimentConfig("Summary", project_id=project.id, hypothesis_id=h.id, seed=4)
    journal = ScientistJournal(study_id=project.id)
    journal.append(
        "decision",
        "Selected summary because it reduced uncertainty.",
        details={"chain_of_thought": "private hidden reasoning", "evidence_run": "r1"},
        rule="uncertainty_first",
    )
    report = generate_report(
        project=project,
        hypotheses=[h],
        experiments=[e],
        results=[{"run_id": "r1", "experiment_id": e.id, "metrics": {"score": 0.8}, "success": True}],
        journal=journal,
    )
    assert "Research Question" in report.markdown
    assert "Methods" in report.markdown
    assert "Limitations" in report.markdown
    assert "private hidden reasoning" not in report.markdown
    assert "chain_of_thought" not in json.dumps(report.data)
    md_path = save_report(report, tmp_path / "research_report.md")
    assert md_path.exists()
    assert md_path.read_text(encoding="utf-8").startswith("#")
