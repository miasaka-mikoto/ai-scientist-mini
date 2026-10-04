from pathlib import Path

from ai_scientist_mini.charts import write_charts
from ai_scientist_mini.demo_study import STRATEGIES, run_demo_study, summarize_by_strategy
from ai_scientist_mini.reporting import build_report, generate_demo_artifacts


def test_memory_strategy_demo_has_two_rounds_and_reproducible_runs():
    first = run_demo_study(seed=123)
    second = run_demo_study(seed=123)
    assert len(first.hypotheses) == 5
    assert len(first.configurations) == 5
    assert len(first.runs) >= 20
    assert {experiment.round for experiment in first.experiments} == {1, 2}
    assert first.to_json() == second.to_json()
    assert all(run.data_hash for run in first.runs)


def test_demo_statistics_and_audit_trail_are_transparent():
    study = run_demo_study(seed=456)
    summaries = summarize_by_strategy(study.runs)
    assert set(summaries) == set(STRATEGIES)
    assert all(summary["n"] > 0 for summary in summaries.values())
    assert all(0 <= summary["success_rate"] <= 1 for summary in summaries.values())
    assert any(decision["type"] == "hypothesis_update" for decision in study.decisions)
    assert any(decision["type"] == "experiment_selection" for decision in study.decisions)
    assert all("rule" in decision for decision in study.decisions)


def test_report_and_charts_are_written(tmp_path: Path):
    study = run_demo_study(seed=789)
    chart_paths = write_charts(study, tmp_path / "charts")
    report = build_report(study, chart_paths=chart_paths)
    assert "Research Question" in report
    assert "Scientific Audit Trail" in report
    assert "Figures" in report
    assert (tmp_path / "charts" / "mean_score_by_strategy.svg").exists()
    assert (tmp_path / "charts" / "success_rate_by_strategy.svg").exists()


def test_generate_demo_artifacts_contains_database_report_and_manifest(tmp_path: Path):
    paths = generate_demo_artifacts(tmp_path, seed=42)
    for key in ("database", "report", "html_report", "mean_score", "success_rate"):
        assert Path(paths[key]).exists(), key
    assert '"run_count": 26' in (tmp_path / "demo_manifest.json").read_text()

