"""Deterministic end-to-end demo study for AI Scientist Mini.

The demo deliberately uses a local synthetic benchmark.  It is useful both as
an acceptance fixture and as an example of how a real provider can be plugged
into the application later.  No network access or model/API key is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STUDY_NAME = "Memory Strategy Study"
STRATEGIES: tuple[str, ...] = (
    "No Memory",
    "Sliding Window",
    "Summary",
    "Episodic",
    "Vector",
)

# These are parameters of a transparent synthetic world, not claims about
# actual memory systems.  The values create a non-trivial but stable demo.
STRATEGY_PROFILES: dict[str, dict[str, float]] = {
    "No Memory": {"base": 0.47, "retention": 0.12, "latency": 0.98},
    "Sliding Window": {"base": 0.60, "retention": 0.38, "latency": 0.92},
    "Summary": {"base": 0.72, "retention": 0.68, "latency": 0.84},
    "Episodic": {"base": 0.77, "retention": 0.76, "latency": 0.79},
    "Vector": {"base": 0.74, "retention": 0.72, "latency": 0.73},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


@dataclass(slots=True)
class DemoHypothesis:
    id: str
    statement: str
    rationale: str
    predictions: list[str]
    test_method: str
    confidence: float = 0.5
    status: str = "Untested"
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DemoConfiguration:
    id: str
    strategy: str
    independent_variable: str
    dependent_variable: str
    control: str
    dataset: str
    sample_size: int
    seed: int
    metric: str
    procedure: list[str]
    success_criteria: str
    hypothesis_id: str


@dataclass(slots=True)
class DemoRun:
    id: str
    experiment_id: str
    configuration_id: str
    hypothesis_id: str
    strategy: str
    round: int
    run_index: int
    seed: int
    status: str
    score: float | None
    success: bool | None
    latency_ms: float | None
    observations: list[str]
    error_class: str | None = None
    started_at: str = ""
    finished_at: str = ""
    data_hash: str = ""


@dataclass(slots=True)
class DemoExperiment:
    id: str
    configuration_id: str
    hypothesis_id: str
    round: int
    status: str
    why: str
    run_ids: list[str] = field(default_factory=list)
    result_summary: dict[str, Any] = field(default_factory=dict)
    selection_rule: str = ""
    selected_at: str = ""


@dataclass(slots=True)
class DemoStudy:
    id: str
    name: str
    research_question: str
    background: str
    scope: list[str]
    constraints: list[str]
    available_resources: list[str]
    metrics: list[str]
    budget: dict[str, Any]
    hypotheses: list[DemoHypothesis]
    configurations: list[DemoConfiguration]
    experiments: list[DemoExperiment]
    runs: list[DemoRun]
    decisions: list[dict[str, Any]]
    journal: list[dict[str, Any]]
    created_at: str
    completed_at: str = ""
    code_version: str = "demo-1.0"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation suitable for persistence."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)


def _hypotheses() -> list[DemoHypothesis]:
    rationale = {
        "No Memory": "A no-state baseline may be fastest but should forget earlier facts.",
        "Sliding Window": "Keeping the latest turns may preserve nearby context at bounded cost.",
        "Summary": "A compressed summary may preserve durable facts while limiting context growth.",
        "Episodic": "Retrieving task episodes may preserve relevant experiences over long gaps.",
        "Vector": "Similarity retrieval may find relevant facts, with possible retrieval noise.",
    }
    predictions = {
        "No Memory": "Lowest long-horizon retention; useful baseline for comparison.",
        "Sliding Window": "Better than no memory on short gaps, weaker on long gaps.",
        "Summary": "High retention with moderate latency and stable success rate.",
        "Episodic": "High retention and the strongest overall success rate in this synthetic task.",
        "Vector": "High retention but occasional retrieval noise and higher variance.",
    }
    result: list[DemoHypothesis] = []
    for strategy in STRATEGIES:
        hid = _stable_id("H", strategy)
        result.append(
            DemoHypothesis(
                id=hid,
                statement=f"{strategy} is a suitable strategy for long-term agent memory.",
                rationale=rationale[strategy],
                predictions=[predictions[strategy]],
                test_method="Run the deterministic long-horizon synthetic memory benchmark and compare success rate and score.",
            )
        )
    return result


def _configurations(hypotheses: Sequence[DemoHypothesis], seed: int) -> list[DemoConfiguration]:
    result: list[DemoConfiguration] = []
    for index, hypothesis in enumerate(hypotheses):
        strategy = STRATEGIES[index]
        cid = _stable_id("CFG", f"{strategy}:{seed}")
        result.append(
            DemoConfiguration(
                id=cid,
                strategy=strategy,
                independent_variable="memory_strategy",
                dependent_variable="long_horizon_recall_score",
                control="No Memory",
                dataset="synthetic-long-horizon-memory-v1",
                sample_size=24,
                seed=seed + index * 1009,
                metric="recall_score",
                procedure=[
                    "Generate a fixed sequence of facts and distractors.",
                    "Apply the selected memory strategy between observation and query.",
                    "Ask delayed recall questions and score exact/partial recall.",
                    "Record score, success, latency, and reproducibility metadata.",
                ],
                success_criteria="score >= 0.65",
                hypothesis_id=hypothesis.id,
            )
        )
    return result


def create_demo_study(*, seed: int = 20261004) -> DemoStudy:
    """Construct an unexecuted study with five hypotheses/configurations."""
    hypotheses = _hypotheses()
    configurations = _configurations(hypotheses, seed)
    return DemoStudy(
        id=_stable_id("study", f"{STUDY_NAME}:{seed}"),
        name=STUDY_NAME,
        research_question="Which memory strategy is most suitable for a long-term agent under a bounded synthetic research budget?",
        background=(
            "Long-term agents must retain useful information across distractors and time gaps. "
            "This demo compares five strategies in a controlled synthetic benchmark; it is not a claim about real-world agents."
        ),
        scope=["Synthetic memory benchmark", "Five candidate strategies", "Two experiment-selection rounds"],
        constraints=[
            "No network access or paid model API",
            "Deterministic seed and fixed synthetic data generator",
            "Results are evidence for this benchmark only, not scientific truth",
        ],
        available_resources=["RuleBasedScientist", "SyntheticProvider", "Local Python runtime"],
        metrics=["mean", "median", "std", "confidence_interval_95", "effect_size_vs_control", "success_rate"],
        budget={
            "maximum_experiments": 10,
            "maximum_runs": 40,
            "maximum_runtime_seconds": 30,
            "estimated_api_cost": 0.0,
            "currency": "USD",
        },
        hypotheses=hypotheses,
        configurations=configurations,
        experiments=[],
        runs=[],
        decisions=[],
        journal=[],
        created_at=_utc_now(),
    )


def _synthetic_observation(strategy: str, seed: int, *, round_number: int, run_index: int) -> tuple[float, bool, float, list[str]]:
    """Generate one stable benchmark result from explicit rules.

    A local ``Random`` instance makes runs independent and reproducible.  The
    round effect models a small follow-up improvement without changing the
    underlying ranking.
    """
    profile = STRATEGY_PROFILES[strategy]
    rng = random.Random(seed)
    gap = rng.choice((0.0, 0.02, 0.04, 0.07, 0.11))
    distractor_penalty = rng.uniform(0.0, 0.06)
    round_bonus = 0.012 if round_number == 2 else 0.0
    score = profile["base"] + profile["retention"] * (0.25 - gap) - distractor_penalty + round_bonus
    score += rng.gauss(0.0, 0.035 if strategy != "Vector" else 0.05)
    score = max(0.0, min(1.0, score))
    latency = 40.0 + (1.0 - profile["latency"]) * 160.0 + rng.uniform(-5.0, 5.0)
    success = score >= 0.65
    observations = [
        f"gap={gap:.2f}",
        f"distractor_penalty={distractor_penalty:.3f}",
        "success criterion met" if success else "success criterion not met",
    ]
    return round(score, 6), success, round(latency, 3), observations


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_experiment(study: DemoStudy, config: DemoConfiguration, *, round_number: int, count: int, selection_rule: str, why: str) -> DemoExperiment:
    experiment = DemoExperiment(
        id=_stable_id("EXP", f"{study.id}:{config.id}:round-{round_number}"),
        configuration_id=config.id,
        hypothesis_id=config.hypothesis_id,
        round=round_number,
        status="Running",
        why=why,
        selection_rule=selection_rule,
        selected_at=_utc_now(),
    )
    study.experiments.append(experiment)
    started = _utc_now()
    for run_index in range(count):
        run_seed = config.seed + round_number * 100_000 + run_index
        run_id = _stable_id("RUN", f"{experiment.id}:{run_index}:{run_seed}")
        score, success, latency, observations = _synthetic_observation(config.strategy, run_seed, round_number=round_number, run_index=run_index)
        run = DemoRun(
            id=run_id,
            experiment_id=experiment.id,
            configuration_id=config.id,
            hypothesis_id=config.hypothesis_id,
            strategy=config.strategy,
            round=round_number,
            run_index=run_index,
            seed=run_seed,
            status="Complete",
            score=score,
            success=success,
            latency_ms=latency,
            observations=observations,
            started_at=started,
            finished_at=_utc_now(),
            data_hash=_hash_payload({"strategy": config.strategy, "seed": run_seed, "score": score, "success": success}),
        )
        study.runs.append(run)
        experiment.run_ids.append(run.id)
    experiment.status = "Complete"
    experiment.result_summary = summarize_runs([run for run in study.runs if run.experiment_id == experiment.id])
    return experiment


def _update_hypotheses(study: DemoStudy, *, round_number: int) -> None:
    """Apply transparent evidence rules and append an audit decision."""
    summaries = summarize_by_strategy(study.runs)
    for hypothesis in study.hypotheses:
        strategy = next((cfg.strategy for cfg in study.configurations if cfg.hypothesis_id == hypothesis.id), "")
        summary = summaries.get(strategy)
        if not summary or summary["n"] == 0:
            continue
        previous = hypothesis.confidence
        success_rate = float(summary["success_rate"])
        mean = float(summary["mean"])
        # Explicit rule: confidence is weighted by benchmark score and success.
        confidence = max(0.01, min(0.99, 0.55 * mean + 0.45 * success_rate))
        hypothesis.confidence = round(confidence, 4)
        if summary["n"] < 3:
            hypothesis.status = "Inconclusive"
        elif confidence >= 0.68:
            hypothesis.status = "Supported"
        elif confidence <= 0.48:
            hypothesis.status = "Rejected"
        else:
            hypothesis.status = "Inconclusive"
        evidence_id = _stable_id("EVD", f"{study.id}:{strategy}:round-{round_number}")
        if evidence_id not in hypothesis.evidence:
            hypothesis.evidence.append(evidence_id)
        study.decisions.append(
            {
                "id": _stable_id("DEC", f"{hypothesis.id}:{round_number}"),
                "type": "hypothesis_update",
                "round": round_number,
                "hypothesis_id": hypothesis.id,
                "evidence_id": evidence_id,
                "previous_confidence": round(previous, 4),
                "new_confidence": hypothesis.confidence,
                "status": hypothesis.status,
                "rule": "confidence = clamp(0.55 * mean_score + 0.45 * success_rate, 0.01, 0.99); status thresholds 0.68/0.48",
                "source_run_ids": [r.id for r in study.runs if r.strategy == strategy],
            }
        )


def _select_second_round(study: DemoStudy, *, seed: int) -> list[DemoConfiguration]:
    summaries = summarize_by_strategy(study.runs)
    # Best expected information: take one high-performing and one uncertain
    # strategy, while ensuring deterministic tie-breaking by strategy name.
    ranked = sorted(
        study.configurations,
        key=lambda cfg: (-float(summaries.get(cfg.strategy, {}).get("mean", 0.0)), cfg.strategy),
    )
    best = ranked[0]
    uncertain = min(
        study.configurations,
        key=lambda cfg: (abs(float(summaries.get(cfg.strategy, {}).get("success_rate", 0.0)) - 0.5), cfg.strategy),
    )
    selected = [best] if best.id == uncertain.id else [best, uncertain]
    study.decisions.append(
        {
            "id": _stable_id("DEC", f"selection:round-2:{seed}"),
            "type": "experiment_selection",
            "round": 2,
            "rule": "Best Expected Information (best mean) + Uncertainty First (closest success rate to 0.5)",
            "selected_configuration_ids": [cfg.id for cfg in selected],
            "candidate_summaries": summaries,
        }
    )
    return selected


def run_demo_study(*, seed: int = 20261004, output_dir: str | os.PathLike[str] | None = None) -> DemoStudy:
    """Run the complete two-round Memory Strategy Study.

    Round one executes 5 configurations × 4 runs (20 runs).  Round two
    executes the selected configurations × 3 runs, yielding at least 23 total
    runs.  The queue and audit semantics are represented in the returned study
    and persisted when ``output_dir`` is supplied.
    """
    study = create_demo_study(seed=seed)
    study.journal.append(
        {
            "timestamp": _utc_now(),
            "event": "study_created",
            "public_summary": "Created five candidate memory hypotheses and deterministic configurations.",
        }
    )
    for config in study.configurations:
        _run_experiment(
            study,
            config,
            round_number=1,
            count=4,
            selection_rule="Random (coverage pass)",
            why="Initial coverage: execute one equal-sized batch for every candidate.",
        )
    _update_hypotheses(study, round_number=1)
    study.journal.append(
        {
            "timestamp": _utc_now(),
            "event": "round_complete",
            "round": 1,
            "public_summary": "Initial coverage completed; confidence updated from observed scores and success rates.",
        }
    )
    selected = _select_second_round(study, seed=seed)
    study.journal.append(
        {
            "timestamp": _utc_now(),
            "event": "next_experiment_selected",
            "round": 2,
            "selected_strategies": [cfg.strategy for cfg in selected],
            "public_summary": "Selected a best-performing and an uncertain strategy for follow-up evidence.",
        }
    )
    for config in selected:
        _run_experiment(
            study,
            config,
            round_number=2,
            count=3,
            selection_rule="Best Expected Information + Uncertainty First",
            why="Follow-up evidence for the strongest current candidate and the most uncertain candidate.",
        )
    _update_hypotheses(study, round_number=2)
    study.completed_at = _utc_now()
    study.journal.append(
        {
            "timestamp": _utc_now(),
            "event": "study_complete",
            "public_summary": "Two-round synthetic study completed; report and charts can be generated from the audit trail.",
        }
    )
    if output_dir is not None:
        persist_demo_study(study, output_dir)
    return study


def persist_demo_study(study: DemoStudy, output_dir: str | os.PathLike[str]) -> Path:
    """Persist the complete study database and a compact reproducibility file."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    database_path = root / "memory_strategy_study.json"
    database_path.write_text(study.to_json() + "\n", encoding="utf-8")
    manifest = {
        "study_id": study.id,
        "study_name": study.name,
        "seed": study.configurations[0].seed if study.configurations else None,
        "run_count": len(study.runs),
        "experiment_count": len(study.experiments),
        "code_version": study.code_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_hash": hashlib.sha256(database_path.read_bytes()).hexdigest(),
    }
    (root / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return database_path


def summarize_runs(runs: Iterable[DemoRun]) -> dict[str, Any]:
    completed = [r for r in runs if r.status == "Complete" and r.score is not None]
    scores = [float(r.score) for r in completed]
    successes = [bool(r.success) for r in completed]
    if not scores:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "ci95": [0.0, 0.0], "success_rate": 0.0, "effect_size": 0.0}
    mean = statistics.fmean(scores)
    median = statistics.median(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    margin = 1.96 * std / (len(scores) ** 0.5)
    return {
        "n": len(scores),
        "mean": round(mean, 6),
        "median": round(median, 6),
        "std": round(std, 6),
        "ci95": [round(max(0.0, mean - margin), 6), round(min(1.0, mean + margin), 6)],
        "success_rate": round(sum(successes) / len(successes), 6),
        "effect_size": 0.0,
    }


def summarize_by_strategy(runs: Iterable[DemoRun]) -> dict[str, dict[str, Any]]:
    run_list = list(runs)
    result = {strategy: summarize_runs([run for run in run_list if run.strategy == strategy]) for strategy in STRATEGIES}
    control = result.get("No Memory", {})
    control_n = int(control.get("n", 0))
    control_std = float(control.get("std", 0.0))
    control_mean = float(control.get("mean", 0.0))
    for strategy, summary in result.items():
        n = int(summary.get("n", 0))
        sd = float(summary.get("std", 0.0))
        pooled_denominator = (((max(control_n - 1, 0) * control_std**2) + (max(n - 1, 0) * sd**2)) / max(control_n + n - 2, 1)) ** 0.5
        summary["effect_size"] = round((float(summary.get("mean", 0.0)) - control_mean) / pooled_denominator, 6) if pooled_denominator else 0.0
    return result


if __name__ == "__main__":  # pragma: no cover - convenience for manual QA
    import argparse

    parser = argparse.ArgumentParser(description="Run the local AI Scientist Mini demo study")
    parser.add_argument("--output", default="artifacts/demo_study", help="output directory")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    study = run_demo_study(seed=args.seed, output_dir=args.output)
    print(json.dumps({"study_id": study.id, "experiments": len(study.experiments), "runs": len(study.runs)}, indent=2))
