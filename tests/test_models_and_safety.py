from __future__ import annotations

import json

import pytest

from aiscientist.core.models import (
    Budget,
    BudgetError,
    ExperimentConfig,
    ExperimentResult,
    FailureType,
    Hypothesis,
    HypothesisStatus,
    Project,
)


def test_domain_models_round_trip_and_enum_values() -> None:
    project = Project(
        research_question="Which memory strategy is best for a long-term agent?",
        scope="offline synthetic benchmark",
        constraints=["no paid API"],
        metrics=["score"],
        budget=Budget(max_experiments=5, max_runs=10),
    )
    restored = Project.from_dict(json.loads(project.to_json()))
    assert restored.research_question == project.research_question
    assert restored.budget.max_runs == 10
    assert restored.status.value == "Planning"

    h = Hypothesis(
        statement="Episodic memory improves long-term score",
        confidence=0.9,
        status=HypothesisStatus.SUPPORTED,
    )
    assert Hypothesis.from_dict(h.to_dict()).status is HypothesisStatus.SUPPORTED


def test_budget_is_a_hard_limit_and_never_goes_negative() -> None:
    budget = Budget(max_experiments=1, max_runs=2)
    assert budget.can_schedule(experiments=1, runs=2)
    budget.consume(experiments=1, runs=2, cost=0.0)
    assert budget.remaining_experiments == 0
    assert budget.remaining_runs == 0
    with pytest.raises(BudgetError):
        budget.consume(experiments=1)


def test_budget_also_limits_runtime_and_api_cost() -> None:
    budget = Budget(max_experiments=2, max_runs=2, max_runtime_seconds=1.0, estimated_api_cost=0.50)
    assert budget.can_schedule(runs=1, runtime_seconds=0.4, cost=0.25)
    budget.consume(runs=1, runtime_seconds=0.4, cost=0.25)
    assert budget.remaining_runtime_seconds == pytest.approx(0.6)
    assert budget.remaining_api_cost == pytest.approx(0.25)
    assert not budget.can_schedule(runtime_seconds=0.7)
    assert not budget.can_schedule(cost=0.3)
    with pytest.raises(BudgetError):
        budget.consume(cost=0.30)


def test_config_hash_is_stable_and_seed_sensitive() -> None:
    a = ExperimentConfig(independent_variable="summary", sample_size=8, seed=11)
    b = ExperimentConfig(independent_variable="summary", sample_size=8, seed=11)
    c = ExperimentConfig(independent_variable="summary", sample_size=8, seed=12)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
    assert a.config_hash(include_seed=False) == c.config_hash(include_seed=False)


def test_failure_taxonomy_is_serializable_and_not_hypothesis_rejection() -> None:
    result = ExperimentResult(
        run_id="run-1",
        success=False,
        failure_type=FailureType.INFRASTRUCTURE,
        error="provider unavailable",
    )
    restored = ExperimentResult.from_dict(result.to_dict())
    assert restored.failure_type is FailureType.INFRASTRUCTURE
    assert restored.success is False
    # A failed run must not silently become a negative scientific result.
    assert restored.failure_type is not FailureType.NEGATIVE_RESULT
