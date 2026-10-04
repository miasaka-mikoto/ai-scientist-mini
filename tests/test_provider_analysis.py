from __future__ import annotations

import pytest

from aiscientist.core.analysis import AnalysisSummary
from aiscientist.core.models import ExperimentConfig, FailureType, ProviderResult
from aiscientist.core.providers import (
    MockLLMExperimentProvider,
    PythonFunctionProvider,
    SyntheticProvider,
)


def test_synthetic_provider_is_reproducible_and_free() -> None:
    config = ExperimentConfig("Episodic", sample_size=12, seed=123)
    provider = SyntheticProvider()
    first = provider.run(config, seed=123)
    second = provider.run(config, seed=123)
    assert first.metrics == second.metrics
    assert first.raw_output == second.raw_output
    assert first.estimated_cost == 0.0
    assert first.success is True


def test_mock_llm_never_requires_an_api_and_is_deterministic() -> None:
    config = ExperimentConfig(
        "arbitrary treatment",
        procedure="Answer a fixed synthetic prompt",
        parameters={"prompt": "hello"},
        seed=99,
    )
    provider = MockLLMExperimentProvider()
    a = provider.run(config, seed=99)
    b = provider.run(config, seed=99)
    assert a.metrics == b.metrics
    assert a.estimated_cost == 0.0
    assert "no external model" in " ".join(a.observations).lower()


def test_python_function_provider_classifies_infrastructure_failure() -> None:
    def broken(_config, _seed):
        raise RuntimeError("synthetic failure")

    result = PythonFunctionProvider(broken).run(ExperimentConfig("x"), seed=4)
    assert isinstance(result, ProviderResult)
    assert not result.success
    assert result.failure_type is FailureType.INFRASTRUCTURE
    assert "synthetic failure" in (result.error or "")


def test_result_analyzer_reports_required_statistics_and_traceable_values() -> None:
    records = [
        {"run_id": "r1", "metrics": {"score": 0.2}, "success": False, "group": "No Memory"},
        {"run_id": "r2", "metrics": {"score": 0.8}, "success": True, "group": "Episodic"},
        {"run_id": "r3", "metrics": {"score": 0.9}, "success": True, "group": "Episodic"},
    ]
    values = [row["metrics"]["score"] for row in records]
    outcomes = [row["success"] for row in records]
    summary = AnalysisSummary.from_values(values, baseline=[0.2], outcomes=outcomes)
    assert summary.n == 3
    assert summary.mean == pytest.approx(0.6333333333333333)
    assert summary.median == 0.8
    assert summary.std is not None
    assert summary.ci_lower < summary.mean < summary.ci_upper
    assert summary.effect_size is not None
    assert summary.success_rate == pytest.approx(2 / 3)
