from __future__ import annotations

import json

import pytest

from aiscientist.adapters import (
    AdapterError,
    LLMLabAdapter,
    LocalJsonAdapter,
)
from aiscientist.reporting import generate_report


def test_local_json_adapter_is_deterministic_and_offline() -> None:
    request = {
        "provider": "synthetic",
        "seed": 7,
        "metadata": {"source": "test"},
        "config": {
            "independent_variable": "Summary",
            "metric": "score",
            "sample_size": 8,
        },
    }
    adapter = LocalJsonAdapter()
    first = json.loads(adapter.execute_json(json.dumps(request)))
    second = adapter.execute_dict(request)
    # Runtime is observational and naturally differs by a few microseconds;
    # the deterministic metrics/raw samples and protocol envelope must match.
    assert first["protocol_version"] == second["protocol_version"]
    assert first["provider"] == second["provider"]
    assert first["result"]["metrics"] == second["result"]["metrics"]
    assert first["result"]["raw_output"] == second["result"]["raw_output"]
    assert first["result"]["estimated_cost"] == 0.0
    assert first["metadata"] == {"source": "test"}


def test_adapter_rejects_unknown_provider_and_external_placeholder() -> None:
    with pytest.raises(AdapterError):
        LocalJsonAdapter().execute_dict({"provider": "paid-api", "config": {"independent_variable": "x"}})
    with pytest.raises(NotImplementedError):
        LLMLabAdapter().execute(None)  # type: ignore[arg-type]


def test_report_aggregates_repeated_follow_up_groups() -> None:
    experiments = [
        {"id": "e1", "hypothesis_id": "h1", "independent_variable": "Summary"},
        {"id": "e2", "hypothesis_id": "h1", "independent_variable": "Summary"},
    ]
    runs = [
        {"id": "run1", "experiment_id": "e1", "result_id": "r1", "status": "Complete"},
        {"id": "run2", "experiment_id": "e2", "result_id": "r2", "status": "Complete"},
    ]
    results = [
        {"id": "r1", "experiment_id": "e1", "metrics": {"score": 0.6}, "success": True},
        {"id": "r2", "experiment_id": "e2", "metrics": {"score": 0.8}, "success": True},
    ]
    report = generate_report(
        project={"research_question": "Does summary help?", "metrics": ["score"]},
        hypotheses=[{"id": "h1", "statement": "Summary helps", "status": "Untested"}],
        experiments=experiments,
        runs=runs,
        results=results,
    )
    assert report.data["analysis"]["groups"]["Summary"]["n"] == 2
    assert "| Summary | 2 | 0.7000" in report.markdown
    assert "| Complete |" in report.markdown


def test_report_uses_metric_success_rate_over_execution_success() -> None:
    report = generate_report(
        project={"research_question": "criterion", "metrics": ["score"]},
        hypotheses=[],
        experiments=[],
        results=[
            {"group": "control", "metrics": {"score": 0.4, "success_rate": 0.0}, "success": True},
            {"group": "control", "metrics": {"score": 0.8, "success_rate": 1.0}, "success": True},
        ],
    )
    assert report.data["analysis"]["groups"]["control"]["success_rate"] == 0.5
