"""End-to-end acceptance checks for the bounded scientific loop.

These tests intentionally exercise the public engine rather than private
implementation details.  They are the regression guard for the deliverable's
most important promise: a restartable, auditable two-round synthetic study.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiscientist.core.engine import ApprovalRequired, ScientificEngine, StopRequested
from aiscientist.core.models import Budget, BudgetError, ExperimentConfig, ExperimentStatus, Project, ProviderResult
from aiscientist.core.providers import ExperimentProvider, PythonFunctionProvider, SyntheticProvider
from aiscientist.core.store import ResearchStore


def _memory_project() -> Project:
    return Project(
        research_question="Which Memory strategy is best for a long-term Agent?",
        scope="offline synthetic benchmark",
        constraints=["no paid API", "bounded runs"],
        metrics=["score", "retention", "latency"],
        budget=Budget(max_experiments=12, max_runs=40, max_runtime_seconds=60, estimated_api_cost=0.0),
    )


def test_complete_two_round_memory_study_is_bounded_and_auditable(tmp_path: Path) -> None:
    db = tmp_path / "memory-study.sqlite3"
    project = _memory_project()
    with ResearchStore(db) as store:
        engine = ScientificEngine(project, provider=SyntheticProvider(), store=store)
        report = engine.run_study(first_round_runs=2, second_round_runs=2, second_round_count=2)

        assert len(report["hypotheses"]) >= 5
        assert len(report["experiments"]) >= 5
        assert len(report["runs"]) >= 14  # five first-round + two follow-up configs
        assert report["rounds"]["first"] >= 5
        assert report["rounds"]["second"] >= 1
        assert report["budget"]["spent_runs"] <= 40
        assert report["budget"]["spent_api_cost"] == 0.0
        assert report["best_hypothesis"] is not None

        # Every conclusion is traceable to experiment evidence and a rule.
        updates = [a for a in report["audit"] if a["event_type"] == "HypothesisUpdated"]
        assert updates
        assert all(a["rule"] for a in updates)
        assert any(a["event_type"] == "ExperimentSelected" for a in report["audit"])
        for hypothesis in report["hypotheses"]:
            if hypothesis["evidence"]:
                evidence = hypothesis["evidence"][-1]
                assert evidence["experiment_id"]
                assert evidence["rule"]
        for run in report["runs"]:
            assert run["result_id"]
            assert run["config_hash"]
            assert run["code_version"]
            assert run["environment"]


def test_reopen_continues_persisted_study_state(tmp_path: Path) -> None:
    db = tmp_path / "resume.sqlite3"
    project = _memory_project()
    with ResearchStore(db) as store:
        engine = ScientificEngine(project, store=store)
        engine.generate_hypotheses()
        configs = engine.design_experiments()
        execution = engine.run_experiment(configs[0], runs=2)
        assert len(execution.runs) == 2
        project_id = engine.project.id

    with ResearchStore(db) as reopened:
        resumed = ScientificEngine(Project.from_dict(reopened.get_project(project_id).to_dict()), store=reopened)
        assert len(resumed.hypotheses) >= 5
        assert len(resumed.experiments) >= 5
        assert len(reopened.list_runs()) == 2
        # Repeating an already-complete config is a no-op and is explicitly
        # marked duplicate rather than consuming more budget.
        duplicate = resumed.run_experiment(resumed.experiments[0], runs=2)
        assert duplicate.duplicate is True
        assert len(reopened.list_runs()) == 2


def test_budget_limit_and_stop_button_are_hard_guards(tmp_path: Path) -> None:
    project = Project(
        "A bounded question",
        budget=Budget(max_experiments=1, max_runs=1, max_runtime_seconds=60),
    )
    with ResearchStore(tmp_path / "budget.sqlite3") as store:
        engine = ScientificEngine(project, store=store)
        engine.generate_hypotheses()
        config = engine.design_experiments()[0]
        with pytest.raises(BudgetError):
            engine.run_experiment(config, runs=2)

        engine.request_stop()
        with pytest.raises(StopRequested):
            engine.run_experiment(config, runs=1)
        assert engine.stop_requested is True


def test_human_approval_blocks_nonlocal_provider_before_execution(tmp_path: Path) -> None:
    # A provider adapter owned by an external integration must be blocked
    # before its callable is invoked.  Built-in synthetic/mock providers remain
    # safe to run automatically in Human Approval mode.
    called = {"value": False}

    class ExternalProvider(ExperimentProvider):
        name = "ExternalProvider"

        def run(self, _config, _seed=None):
            called["value"] = True
            return {"score": 0.9}

    project = Project("Approval question", budget=Budget(max_experiments=2, max_runs=2))
    with ResearchStore(tmp_path / "approval.sqlite3") as store:
        engine = ScientificEngine(
            project,
            provider=ExternalProvider(),
            store=store,
            approval_mode="Human Approval",
        )
        config = ExperimentConfig("external", project_id=project.id, seed=1)
        with pytest.raises(ApprovalRequired):
            engine.run_experiment(config)
        assert called["value"] is False


def test_failed_provider_run_is_not_a_negative_hypothesis_result(tmp_path: Path) -> None:
    def broken(_config, _seed):
        raise RuntimeError("simulated infrastructure outage")

    project = Project("Failure taxonomy question", budget=Budget(max_experiments=2, max_runs=2))
    with ResearchStore(tmp_path / "failure.sqlite3") as store:
        engine = ScientificEngine(project, provider=PythonFunctionProvider(broken), store=store)
        config = ExperimentConfig("broken", project_id=project.id, seed=3)
        execution = engine.run_experiment(config, runs=1)
        assert execution.runs[0].status is ExperimentStatus.FAILED
        assert execution.runs[0].failure_type.value == "Infrastructure Failure"
        assert execution.results[0].success is False
        assert execution.results[0].failure_type.value == "Infrastructure Failure"


def test_provider_cost_estimate_is_blocked_before_execution(tmp_path: Path) -> None:
    called = {"value": False}

    class MeteredProvider(ExperimentProvider):
        name = "MeteredProvider"
        estimated_cost_per_run = 0.75

        def run(self, _config, _seed=None):
            called["value"] = True
            return ProviderResult(metrics={"score": 0.9}, estimated_cost=0.75)

    project = Project("Cost guard", budget=Budget(max_experiments=2, max_runs=2, estimated_api_cost=0.50))
    with ResearchStore(tmp_path / "cost.sqlite3") as store:
        engine = ScientificEngine(project, provider=MeteredProvider(), store=store)
        config = ExperimentConfig("metered", project_id=project.id, seed=1)
        with pytest.raises(BudgetError):
            engine.run_experiment(config, runs=1)
        assert called["value"] is False
