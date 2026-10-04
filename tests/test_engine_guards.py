from pathlib import Path

import pytest

from ai_scientist_mini.engine import DuplicateExperiment, ScientistEngine
from ai_scientist_mini.models import (
    ApprovalMode,
    BudgetExceeded,
    FailureType,
    QueueStatus,
    ResearchBudget,
)
from ai_scientist_mini.providers import ExperimentProvider, ProviderError, ProviderResult


def _engine(*, provider=None, max_runs=10, max_experiments=5):
    engine = ScientistEngine(provider=provider)
    engine.create_project(
        "Which memory strategy is best for a long-term agent?",
        budget=ResearchBudget(maximum_experiments=max_experiments, maximum_runs=max_runs),
    )
    engine.propose_hypotheses()
    return engine


def test_budget_and_duplicate_guards_account_completed_runs():
    engine = _engine(max_runs=1)
    design = engine.design_experiment(engine.project.hypotheses[0], seed=7, sample_size=1)
    experiment = engine.queue_experiment(design)
    engine.run_experiment(experiment, runs=1)
    assert engine.project.budget.experiments_used == 1
    assert engine.project.budget.runs_used == 1
    with pytest.raises(DuplicateExperiment):
        engine.queue_experiment(design)
    other_design = engine.design_experiment(engine.project.hypotheses[1], seed=8, sample_size=1)
    other = engine.queue_experiment(other_design)
    with pytest.raises(BudgetExceeded):
        engine.run_experiment(other, runs=1)


class FailingProvider(ExperimentProvider):
    def execute(self, design, *, seed, run_index=0):
        raise ProviderError("bad synthetic design", FailureType.INVALID_DESIGN)


def test_failure_type_is_preserved_and_is_not_a_rejected_hypothesis():
    engine = _engine(provider=FailingProvider())
    design = engine.design_experiment(engine.project.hypotheses[0], seed=2, sample_size=1)
    experiment = engine.queue_experiment(design)
    result = engine.run_experiment(experiment, runs=1)
    assert result.status == QueueStatus.FAILED
    assert result.failure_type == FailureType.INVALID_DESIGN
    assert result.runs[0].failure_type == FailureType.INVALID_DESIGN
    assert engine.project.hypotheses[0].status.value == "Untested"


class StopAfterFirstProvider(ExperimentProvider):
    def __init__(self):
        self.engine = None

    def execute(self, design, *, seed, run_index=0):
        if run_index == 0:
            self.engine.request_stop()
        return ProviderResult(metrics={"score": 0.7}, runtime_seconds=0.0)


def test_stop_button_cancels_in_progress_experiment_and_restart_persists(tmp_path: Path):
    provider = StopAfterFirstProvider()
    engine = _engine(provider=provider, max_runs=10)
    provider.engine = engine
    design = engine.design_experiment(engine.project.hypotheses[0], seed=3, sample_size=3)
    experiment = engine.queue_experiment(design)
    result = engine.run_experiment(experiment, runs=3)
    assert result.status == QueueStatus.CANCELLED
    assert len(result.runs) == 1
    assert engine.project.memory.previous_experiments == []  # cancelled runs are not results
    path = engine.save_state(tmp_path / "state.json")
    restored = ScientistEngine.load_state(path, provider=provider)
    assert restored.project.experiment(experiment.id).status == QueueStatus.CANCELLED
    assert restored.project.budget.experiments_used == 1


def test_successful_run_is_added_to_research_memory():
    engine = _engine(max_runs=2)
    design = engine.design_experiment(engine.project.hypotheses[0], seed=4, sample_size=1)
    experiment = engine.queue_experiment(design)
    engine.run_experiment(experiment, runs=1)
    assert experiment.id in engine.project.memory.previous_experiments
    assert engine.project.memory.results[-1]["experiment_id"] == experiment.id


def test_human_approval_denial_keeps_queue_item_resumable():
    engine = _engine(max_runs=4)
    engine.project.approval_mode = ApprovalMode.HUMAN_APPROVAL
    design = engine.design_experiment(engine.project.hypotheses[0], seed=9, sample_size=1)
    experiment = engine.queue_experiment(design)
    with pytest.raises(PermissionError):
        engine.run_experiment(experiment, runs=1, approved=False)
    assert experiment.status == QueueStatus.QUEUED
    assert experiment.runs == []
