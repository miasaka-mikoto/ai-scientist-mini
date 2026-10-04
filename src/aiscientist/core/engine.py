"""Rule-based scientific loop and experiment orchestration.

This module is intentionally transparent: every confidence update and every
selection decision emits an :class:`AuditEvent` with the rule that caused it.
It is a bounded offline research sandbox, not a claim of autonomous scientific
discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .analysis import AnalysisSummary, analyze_values, compare_groups
from .models import (
    AuditEvent,
    Budget,
    BudgetError,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRun,
    ExperimentStatus,
    FailureType,
    Hypothesis,
    HypothesisStatus,
    JournalEntry,
    MemoryRecord,
    Project,
    ProjectStatus,
    QueueItem,
    SelectionStrategy,
    environment_snapshot,
    new_id,
    utc_now,
)
from .providers import ExperimentProvider, MockLLMExperimentProvider, SyntheticProvider
from .store import ResearchStore


class ApprovalRequired(RuntimeError):
    """Raised when Human Approval mode blocks an external operation."""


class StopRequested(RuntimeError):
    """Raised when the user pressed the stop button."""


@dataclass
class ExperimentExecution:
    """Return value for :meth:`ScientificEngine.run_experiment`.

    It behaves like a read-only sequence of results for simple callers while
    retaining configuration, runs and aggregate analysis for the dashboard.
    """

    config: ExperimentConfig
    runs: list[ExperimentRun] = field(default_factory=list)
    results: list[ExperimentResult] = field(default_factory=list)
    analysis: AnalysisSummary | None = None
    duplicate: bool = False

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> ExperimentResult:
        return self.results[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "results": [r.to_dict() for r in self.results],
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "duplicate": self.duplicate,
        }


class RuleBasedScientist:
    """Transparent hypothesis and design heuristics.

    The class has no network/model dependency and can be used independently of
    :class:`ScientificEngine` for teaching or unit tests.
    """

    MEMORY_STRATEGIES = ("No Memory", "Sliding Window", "Summary", "Episodic", "Vector")

    def generate_hypotheses(self, research_question: str, *, project_id: str | None = None) -> list[Hypothesis]:
        question = (research_question or "").strip()
        lower = question.lower()
        if any(term in lower for term in ("memory", "长期", "long-term", "agent")):
            statements = [
                ("No Memory is a sufficient baseline for long-term Agent tasks.", "A stateless agent may perform adequately when tasks are short or self-contained.", "No Memory"),
                ("Sliding Window memory improves long-term Agent performance over No Memory.", "Recent context should preserve local continuity at modest cost.", "Sliding Window"),
                ("Summary memory improves long-term Agent performance while controlling context cost.", "Compressed history may retain salient facts and reduce irrelevant context.", "Summary"),
                ("Episodic memory provides the best long-term Agent performance.", "Retrievable episodes preserve durable experiences across tasks.", "Episodic"),
                ("Vector memory provides the best quality-cost trade-off for long-term Agent tasks.", "Semantic retrieval should recall relevant history, though indexing adds latency.", "Vector"),
            ]
            return [
                Hypothesis(
                    statement=statement,
                    rationale=rationale,
                    predictions=[f"{strategy} score exceeds the No Memory baseline" if strategy != "No Memory" else "Baseline score is below richer memory strategies"],
                    test_method="Run the synthetic memory benchmark with fixed seeds and compare mean score, retention, and latency.",
                    confidence=0.5,
                    project_id=project_id,
                    tags=[strategy, "memory-study"],
                )
                for statement, rationale, strategy in statements
            ]
        # Generic questions still receive explicit, falsifiable candidates.
        return [
            Hypothesis(
                statement=f"Treatment A improves the primary metric for: {question or 'the research question'}.",
                rationale="A simple intervention may increase the measured outcome.",
                predictions=["Mean score under Treatment A is higher than control."],
                test_method="Compare fixed-seed treatment and control runs.",
                project_id=project_id,
                tags=["treatment-a", "generic"],
            ),
            Hypothesis(
                statement=f"Treatment B improves the primary metric for: {question or 'the research question'}.",
                rationale="A distinct mechanism may outperform Treatment A.",
                predictions=["Treatment B has the highest mean score."],
                test_method="Compare treatment groups using the configured metric.",
                project_id=project_id,
                tags=["treatment-b", "generic"],
            ),
            Hypothesis(
                statement=f"The control is competitive for: {question or 'the research question'}.",
                rationale="The intervention may not justify its cost or complexity.",
                predictions=["Control is within the practical equivalence margin."],
                test_method="Estimate effect size and uncertainty against control.",
                project_id=project_id,
                tags=["control", "generic"],
            ),
        ]

    def design_experiment(
        self,
        hypothesis: Hypothesis,
        *,
        project: Project | None = None,
        treatment: str | None = None,
        seed: int = 42,
        sample_size: int | None = None,
    ) -> ExperimentConfig:
        strategy = treatment or self._treatment_for(hypothesis)
        sample = sample_size or int((project.available_resources if project else {}).get("sample_size", 20))
        metric = (project.metrics[0] if project and project.metrics else "score")
        return ExperimentConfig(
            independent_variable=strategy,
            dependent_variable=metric,
            control="No Memory" if "memory" in " ".join(hypothesis.tags).lower() else "control",
            dataset="synthetic_memory_benchmark" if "memory" in " ".join(hypothesis.tags).lower() else "synthetic",
            sample_size=sample,
            seed=seed,
            metric=metric,
            procedure=["Generate deterministic synthetic tasks", "Evaluate treatment", "Aggregate primary metric"],
            success_criteria={"min_score": 0.65, "confidence_level": 0.95},
            hypothesis_id=hypothesis.id,
            project_id=project.id if project else hypothesis.project_id,
            name=f"{strategy} benchmark",
            parameters={"strategy": strategy},
            provider="synthetic",
            runs_per_config=1,
        )

    def update_hypothesis_from_analysis(
        self,
        hypothesis: Hypothesis,
        analysis: AnalysisSummary,
        *,
        baseline_mean: float | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> tuple[HypothesisStatus, float, str]:
        """Apply an explicit, conservative confidence rule.

        Support requires a usable sample and either the configured practical
        threshold or a positive effect against a known baseline.  Confidence
        moves by a bounded amount; no hidden chain-of-thought is generated.
        """

        if analysis.n < 2:
            return HypothesisStatus.INCONCLUSIVE, max(0.25, hypothesis.confidence * 0.95), "n<2 => inconclusive"
        if analysis.std is None or math.isnan(analysis.std):
            return HypothesisStatus.INCONCLUSIVE, hypothesis.confidence, "missing variance => inconclusive"
        mean = analysis.mean
        lower = analysis.confidence_interval[0] if analysis.confidence_interval else mean
        upper = analysis.confidence_interval[1] if analysis.confidence_interval else mean
        threshold = 0.65
        if baseline_mean is not None:
            effect = mean - baseline_mean
            # Positive effect whose interval excludes zero supports; clearly
            # negative effect rejects.  Otherwise retain uncertainty.
            margin = max(0.01, (upper - lower) / 2)
            if effect > margin:
                status = HypothesisStatus.SUPPORTED
                rule = f"effect={effect:.3f} > CI half-width={margin:.3f}"
            elif effect < -margin:
                status = HypothesisStatus.REJECTED
                rule = f"effect={effect:.3f} < -CI half-width={margin:.3f}"
            else:
                status = HypothesisStatus.INCONCLUSIVE
                rule = f"|effect|={abs(effect):.3f} <= CI half-width={margin:.3f}"
        elif lower >= threshold:
            status = HypothesisStatus.SUPPORTED
            rule = f"CI lower={lower:.3f} >= threshold={threshold:.2f}"
        elif upper < threshold * 0.9:
            status = HypothesisStatus.REJECTED
            rule = f"CI upper={upper:.3f} < rejection threshold={threshold * 0.9:.2f}"
        else:
            status = HypothesisStatus.INCONCLUSIVE
            rule = f"CI overlaps practical threshold={threshold:.2f}"
        # Confidence is a bounded transform of distance from the decision
        # boundary and sample size.  Existing confidence is retained as prior.
        certainty = min(1.0, analysis.n / 20.0) * min(1.0, abs(mean - threshold) / 0.35 + 0.15)
        target = {HypothesisStatus.SUPPORTED: 0.80, HypothesisStatus.REJECTED: 0.20, HypothesisStatus.INCONCLUSIVE: 0.50}[status]
        confidence = min(0.99, max(0.01, hypothesis.confidence * (1 - certainty) + target * certainty))
        return status, confidence, rule

    def select_candidate(
        self,
        hypotheses: Sequence[Hypothesis],
        configs: Sequence[ExperimentConfig],
        *,
        strategy: SelectionStrategy = SelectionStrategy.BEST_EXPECTED_INFORMATION,
        completed_experiment_ids: set[str] | None = None,
        seed: int = 0,
    ) -> ExperimentConfig | None:
        completed = completed_experiment_ids or set()
        available = [c for c in configs if c.id not in completed]
        if not available:
            return None
        if isinstance(strategy, str):
            strategy = SelectionStrategy(strategy)
        by_hyp = {h.id: h for h in hypotheses}
        if strategy == SelectionStrategy.RANDOM:
            return random.Random(seed).choice(available)
        if strategy == SelectionStrategy.UNCERTAINTY_FIRST:
            return min(available, key=lambda c: abs(by_hyp.get(c.hypothesis_id, Hypothesis("")).confidence - 0.5))
        if strategy == SelectionStrategy.FOLLOW_UP_FAILED_RESULT:
            # Caller marks failed configs in metadata; prioritize those first.
            failed = [c for c in available if c.metadata.get("follow_up_failed")]
            if failed:
                return failed[0]
        # Expected information is greatest near confidence 0.5 and for unseen
        # treatments.  Stable ID tie-breakers make resume deterministic.
        return max(
            available,
            key=lambda c: (
                1.0 - abs(by_hyp.get(c.hypothesis_id, Hypothesis("")).confidence - 0.5),
                -len(c.metadata.get("observations", [])),
                c.id,
            ),
        )

    @staticmethod
    def _treatment_for(hypothesis: Hypothesis) -> str:
        # Tags are generated by the rule-based proposer and are the
        # unambiguous treatment identity.  Checking statement substrings first
        # incorrectly mapped “Sliding Window ... over No Memory” to the
        # baseline because the baseline phrase appeared later in the sentence.
        tags = {str(tag).strip().casefold() for tag in hypothesis.tags}
        for strategy in RuleBasedScientist.MEMORY_STRATEGIES:
            if strategy.casefold() in tags:
                return strategy
        statement = hypothesis.statement.casefold()
        for strategy in RuleBasedScientist.MEMORY_STRATEGIES:
            if strategy.casefold() in statement:
                return strategy
        return hypothesis.tags[0] if hypothesis.tags else "Treatment A"


class ScientificEngine(RuleBasedScientist):
    """Bounded, resumable implementation of the complete scientific loop."""

    def __init__(
        self,
        project: Project | None = None,
        *,
        provider: ExperimentProvider | None = None,
        store: ResearchStore | None = None,
        approval_mode: str = "Autonomous Sandbox",
        code_version: str = "aiscientist-mini-0.1",
    ):
        self.store = store or ResearchStore()
        self.project = project or Project(research_question="")
        persisted = self.store.get_project(self.project.id)
        if persisted is not None:
            self.project = persisted
        else:
            self.store.save_project(self.project)
        self.provider = provider or SyntheticProvider()
        self.approval_mode = approval_mode
        self.code_version = code_version
        self._stop_requested = False
        self.hypotheses: list[Hypothesis] = self.store.list_hypotheses(self.project.id)
        self.experiments: list[ExperimentConfig] = self.store.list_experiments(self.project.id)

    # ---------- setup ----------
    def create_project(self, research_question: str, **kwargs: Any) -> Project:
        self.project = Project(research_question=research_question, **kwargs)
        self.store.save_project(self.project)
        self.hypotheses, self.experiments = [], []
        self._audit("ProjectCreated", "Created research project", entity_type="project", entity_id=self.project.id)
        return self.project

    def generate_hypotheses(self, research_question: str | None = None) -> list[Hypothesis]:  # type: ignore[override]
        question = research_question if research_question is not None else self.project.research_question
        if self.hypotheses:
            return self.hypotheses
        self.hypotheses = super().generate_hypotheses(question, project_id=self.project.id)
        for hypothesis in self.hypotheses:
            self.store.save_hypothesis(hypothesis)
            self._audit("HypothesisProposed", hypothesis.statement, entity_type="hypothesis", entity_id=hypothesis.id, details={"rationale": hypothesis.rationale})
        self.project.status = ProjectStatus.RUNNING
        self.project.touch()
        self.store.save_project(self.project)
        return self.hypotheses

    def design_experiments(self, *, seed: int = 42, sample_size: int | None = None, force: bool = False) -> list[ExperimentConfig]:
        if not self.hypotheses:
            self.generate_hypotheses()
        if self.experiments and not force:
            return self.experiments
        configs: list[ExperimentConfig] = []
        for index, hypothesis in enumerate(self.hypotheses):
            treatment = self._treatment_for(hypothesis)
            config = super().design_experiment(hypothesis, project=self.project, treatment=treatment, seed=seed + index, sample_size=sample_size)
            # Keep a design hash in metadata for audit/dedup diagnostics.
            config.metadata["design_hash"] = config.config_hash(include_seed=False)
            if self.store.find_duplicate(config, include_seed=False):
                continue
            self.store.save_experiment(config)
            configs.append(config)
            self._audit("ExperimentDesigned", f"Designed {config.name}", entity_type="experiment", entity_id=config.id, details={"hypothesis_id": hypothesis.id, "config": config.to_dict()})
        self.experiments.extend(configs)
        return self.experiments

    # ---------- queue and execution ----------
    def enqueue_experiment(self, config: ExperimentConfig, *, priority: int = 0) -> QueueItem:
        if self._stop_requested:
            raise StopRequested("Stop button is active")
        item = QueueItem(experiment_id=config.id, priority=priority)
        self.store.enqueue(item)
        self._audit("ExperimentQueued", f"Queued {config.name}", entity_type="experiment", entity_id=config.id)
        return item

    def request_stop(self) -> None:
        self._stop_requested = True
        self._audit("StopRequested", "User requested stop button")

    def clear_stop(self) -> None:
        self._stop_requested = False
        self._audit("StopCleared", "User resumed experiment execution")

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def run_experiment(
        self,
        config: ExperimentConfig | str,
        *,
        runs: int | None = None,
        force: bool = False,
        approval: bool = False,
    ) -> ExperimentExecution:
        if self._stop_requested:
            raise StopRequested("Stop button is active")
        if isinstance(config, str):
            found = self.store.get_experiment(config)
            if found is None:
                raise KeyError(f"Unknown experiment: {config}")
            config = found
        requested_runs = max(1, int(runs or config.runs_per_config))
        duplicate = self.store.find_duplicate(config, include_seed=True)
        existing_runs = self.store.list_runs(duplicate.id) if duplicate else []
        if duplicate is not None and not force and len(existing_runs) >= requested_runs:
            results = [self.store.get_result(r.result_id) for r in existing_runs if r.result_id]
            results = [r for r in results if r is not None]
            return ExperimentExecution(duplicate, existing_runs, results, self.analyze_experiment(duplicate), duplicate=True)
        if self.approval_mode.lower().startswith("human") and not approval and not self._is_local_provider():
            raise ApprovalRequired("Human Approval mode requires approval before external resource use")
        # A new treatment consumes one experiment slot.  Resume runs consume
        # only their additional run slots.  ``design_experiments`` persists
        # planned configs before execution, so a config that exists but has
        # zero runs still consumes its first experiment slot.  Only a config
        # with completed/failed runs is a true dedup/resume case.
        is_new = duplicate is None or not existing_runs or force
        try:
            estimated_cost = max(0.0, float(self.provider.estimate_cost(config)))
        except Exception:
            # A provider that cannot estimate cost is not allowed to bypass a
            # configured allowance; the observed result will still be audited.
            estimated_cost = float("inf")
        if not self.project.budget.can_schedule(
            experiments=1 if is_new else 0,
            runs=requested_runs,
            cost=estimated_cost * requested_runs,
        ):
            raise BudgetError("Research budget exhausted: reduce runs or increase limits explicitly")
        if is_new:
            self.store.save_experiment(config)
            if all(e.id != config.id for e in self.experiments):
                self.experiments.append(config)
            self.project.budget.consume(experiments=1)
        self.project.status = ProjectStatus.RUNNING
        self.store.save_project(self.project)
        queue_item = self.enqueue_experiment(config)
        queue_item.status = ExperimentStatus.RUNNING
        queue_item.started_at = utc_now()
        self.store.enqueue(queue_item)
        results: list[ExperimentResult] = []
        run_objects: list[ExperimentRun] = []
        budget_exhausted = False
        for index in range(requested_runs):
            if self._stop_requested:
                queue_item.status = ExperimentStatus.CANCELLED
                queue_item.finished_at = utc_now()
                self.store.enqueue(queue_item)
                raise StopRequested("Execution stopped after completed runs")
            # Reserve one run immediately before the provider call.  This
            # keeps persisted counters accurate if a runtime/cost ceiling
            # stops a partially completed batch.
            try:
                self.project.budget.consume(runs=1)
            except BudgetError:
                budget_exhausted = True
                self._audit(
                    "BudgetLimitReached",
                    "Run skipped because the research budget is exhausted",
                    entity_type="experiment",
                    entity_id=config.id,
                    rule="budget_guard",
                )
                break
            seed = int(config.seed) + index
            run = ExperimentRun(experiment_id=config.id, run_index=index, seed=seed, status=ExperimentStatus.RUNNING, started_at=utc_now(), config_hash=config.config_hash(), code_version=self.code_version, environment=environment_snapshot())
            self.store.save_run(run)
            try:
                provider_result = self.provider.run(config, seed)
                result = ExperimentResult(run_id=run.id, experiment_id=config.id, metrics=provider_result.metrics, observations=provider_result.observations, success=provider_result.success, raw_output=provider_result.raw_output, failure_type=provider_result.failure_type, error=provider_result.error, data_hash=_hash_data(provider_result.raw_output))
                self.store.save_result(result)
                run.result_id = result.id
                run.completed_at = utc_now()
                run.status = ExperimentStatus.COMPLETE if provider_result.success else ExperimentStatus.FAILED
                run.error = provider_result.error
                run.failure_type = provider_result.failure_type
                run.logs.extend(provider_result.observations)
                results.append(result)
            except Exception as exc:  # provider boundary: preserve failed run
                run.status = ExperimentStatus.FAILED
                run.completed_at = utc_now()
                run.error = f"{type(exc).__name__}: {exc}"
                run.failure_type = FailureType.INFRASTRUCTURE
                result = ExperimentResult(run_id=run.id, experiment_id=config.id, success=False, error=run.error, failure_type=FailureType.INFRASTRUCTURE)
                self.store.save_result(result)
                run.result_id = result.id
                results.append(result)
                provider_result = None
            # Runtime and observed provider cost are charged after the result
            # is persisted.  A provider cannot retroactively be allowed to
            # exceed the configured ceiling; the current run remains
            # traceable, and the queue is stopped before another call.
            if provider_result is not None:
                try:
                    self.project.budget.consume(
                        runtime_seconds=max(0.0, float(provider_result.runtime_seconds)),
                        cost=max(0.0, float(provider_result.estimated_cost)),
                    )
                except BudgetError:
                    budget_exhausted = True
                    self._audit(
                        "BudgetLimitReached",
                        "Execution stopped after an observed runtime or cost overrun",
                        entity_type="run",
                        entity_id=run.id,
                        rule="budget_guard",
                        details={
                            "runtime_seconds": provider_result.runtime_seconds,
                            "estimated_cost": provider_result.estimated_cost,
                        },
                    )
            self.store.save_run(run)
            run_objects.append(run)
            self._audit("RunCompleted" if run.status == ExperimentStatus.COMPLETE else "RunFailed", f"Run {run.id} finished with status {run.status.value}", entity_type="run", entity_id=run.id, details={"experiment_id": config.id, "result_id": run.result_id, "failure_type": str(run.failure_type) if run.failure_type else None})
            self.store.save_project(self.project)
            if budget_exhausted:
                break
        if budget_exhausted:
            queue_item.status = ExperimentStatus.CANCELLED
            self.project.status = ProjectStatus.PAUSED
        else:
            queue_item.status = ExperimentStatus.COMPLETE if run_objects and all(r.status == ExperimentStatus.COMPLETE for r in run_objects) else ExperimentStatus.FAILED
        queue_item.finished_at = utc_now()
        self.store.enqueue(queue_item)
        analysis = self.analyze_experiment(config)
        self._update_hypothesis_for_experiment(config, analysis)
        self._journal_for_experiment(config, analysis)
        return ExperimentExecution(config=config, runs=run_objects, results=results, analysis=analysis, duplicate=False)

    def run_round(
        self,
        configs: Sequence[ExperimentConfig] | None = None,
        *,
        runs_per_experiment: int = 4,
        selection_strategy: SelectionStrategy = SelectionStrategy.BEST_EXPECTED_INFORMATION,
    ) -> list[ExperimentExecution]:
        if not self.hypotheses:
            self.generate_hypotheses()
        if not self.experiments:
            self.design_experiments()
        selected = list(configs or self.experiments)
        executions: list[ExperimentExecution] = []
        for config in selected:
            if self._stop_requested:
                break
            try:
                executions.append(self.run_experiment(config, runs=runs_per_experiment))
            except (BudgetError, StopRequested):
                break
        return executions

    def select_next_experiment(self, strategy: SelectionStrategy = SelectionStrategy.BEST_EXPECTED_INFORMATION) -> ExperimentConfig | None:
        if not self.experiments:
            self.design_experiments()
        completed = {e.id for e in self.experiments if self.store.list_runs(e.id)}
        candidate = super().select_candidate(self.hypotheses, self.experiments, strategy=strategy, completed_experiment_ids=completed, seed=self.project.budget.spent_runs)
        if candidate is None and strategy == SelectionStrategy.FOLLOW_UP_FAILED_RESULT:
            failed = [e for e in self.experiments if any(r.status == ExperimentStatus.FAILED for r in self.store.list_runs(e.id))]
            if failed:
                base = failed[0]
                candidate = ExperimentConfig.from_dict(base.to_dict())
                candidate.id = new_id("exp_followup_")
                candidate.seed += len(self.store.list_runs(base.id)) + 1
                candidate.sample_size *= 2
                candidate.metadata["follow_up_failed"] = True
                self.store.save_experiment(candidate)
                self.experiments.append(candidate)
        if candidate:
            self._audit("ExperimentSelected", f"Selected {candidate.name} using {SelectionStrategy(strategy).value}", entity_type="experiment", entity_id=candidate.id, rule=SelectionStrategy(strategy).value)
        return candidate

    # ---------- analysis and update ----------
    def analyze_experiment(self, config: ExperimentConfig | str) -> AnalysisSummary:
        if isinstance(config, str):
            found = self.store.get_experiment(config)
            if found is None:
                raise KeyError(config)
            config = found
        values: list[float] = []
        failures = 0
        for run in self.store.list_runs(config.id):
            result = self.store.get_result(run.result_id) if run.result_id else None
            if result is None:
                failures += 1
                continue
            value = result.metrics.get(config.metric)
            if value is None:
                failures += 1
            else:
                values.append(value)
        # Keep failure counts in metadata rather than pretending failed runs
        # were numerical observations.
        summary = AnalysisSummary.from_values(
            values,
            metadata={"metric": config.metric, "failed_runs": failures},
        ) if values else AnalysisSummary(n=0, metadata={"metric": config.metric, "failed_runs": failures})
        return summary

    def analyze_all(self) -> dict[str, AnalysisSummary]:
        return {config.id: self.analyze_experiment(config) for config in self.experiments if self.store.list_runs(config.id)}

    def _update_hypothesis_for_experiment(self, config: ExperimentConfig, analysis: AnalysisSummary) -> Hypothesis | None:
        if not config.hypothesis_id:
            return None
        hypothesis = next((h for h in self.hypotheses if h.id == config.hypothesis_id), None) or self.store.get_hypothesis(config.hypothesis_id)
        if hypothesis is None:
            return None
        baseline_mean = self._baseline_mean(config)
        status, confidence, rule = self.update_hypothesis_from_analysis(hypothesis, analysis, baseline_mean=baseline_mean)
        hypothesis.status = status
        hypothesis.confidence = confidence
        hypothesis.updated_at = utc_now()
        evidence = {"experiment_id": config.id, "metric": config.metric, "analysis": analysis.to_dict(), "rule": rule}
        hypothesis.evidence.append(evidence)
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)
        self.store.save_hypothesis(hypothesis)
        self._audit("HypothesisUpdated", f"{hypothesis.id}: {status.value} (confidence {confidence:.3f})", entity_type="hypothesis", entity_id=hypothesis.id, rule=rule, details=evidence)
        self.store.add_memory(MemoryRecord(category="result", content=f"{config.name}: {status.value}; {rule}", references=[config.id, hypothesis.id]))
        return hypothesis

    def _baseline_mean(self, config: ExperimentConfig) -> float | None:
        if "memory" not in (config.dataset or "").lower():
            return None
        baseline = next((e for e in self.experiments if str(e.independent_variable) == "No Memory" and e.id != config.id), None)
        if not baseline:
            return None
        summary = self.analyze_experiment(baseline)
        return summary.mean if summary.n else None

    # ---------- study orchestration ----------
    def run_study(self, *, first_round_runs: int = 4, second_round_runs: int = 3, second_round_count: int = 2) -> dict[str, Any]:
        if not self.hypotheses:
            self.generate_hypotheses()
        if not self.experiments:
            self.design_experiments()
        first = self.run_round(runs_per_experiment=first_round_runs)
        second_configs: list[ExperimentConfig] = []
        # Selection is deliberately visible and bounded: at most the requested
        # count, and no budget overrun.
        for _ in range(max(0, second_round_count)):
            candidate = self.select_next_experiment(SelectionStrategy.UNCERTAINTY_FIRST)
            if candidate is None:
                # Follow-up failed result gets a new config/seed, otherwise
                # revisit the lowest-confidence treatment with extra samples.
                candidate = self._make_follow_up()
            if candidate is None:
                break
            second_configs.append(candidate)
            try:
                self.run_experiment(candidate, runs=second_round_runs)
            except (BudgetError, StopRequested):
                break
        self.project.status = (
            ProjectStatus.PAUSED
            if self._stop_requested or self.project.status == ProjectStatus.PAUSED
            else ProjectStatus.COMPLETED
        )
        self.project.touch()
        self.store.save_project(self.project)
        report = self.generate_report_data()
        report["rounds"] = {"first": len(first), "second": len(second_configs)}
        return report

    def _make_follow_up(self) -> ExperimentConfig | None:
        if not self.experiments:
            return None
        # Choose the least certain hypothesis and increase sample size.  The
        # changed metadata/seed makes deduplication intentional and explicit.
        by_hyp = {h.id: h for h in self.hypotheses}
        base = min(self.experiments, key=lambda e: abs(by_hyp.get(e.hypothesis_id, Hypothesis("")).confidence - 0.5))
        follow = ExperimentConfig.from_dict(base.to_dict())
        follow.id = new_id("exp_followup_")
        # Derive a globally unused seed, not merely ``base.seed + run_count``;
        # repeated rounds may follow up the same base experiment.
        used_seeds = [e.seed for e in self.experiments]
        follow.seed = (max(used_seeds) + 1) if used_seeds else (base.seed + 101)
        follow.sample_size = max(base.sample_size * 2, base.sample_size + 5)
        follow.runs_per_config = 1
        follow.metadata["follow_up_of"] = base.id
        follow.metadata["follow_up_reason"] = "uncertainty-first additional sample"
        follow.metadata["follow_up_index"] = sum(1 for e in self.experiments if e.metadata.get("follow_up_of")) + 1
        self.store.save_experiment(follow)
        self.experiments.append(follow)
        self._audit("FollowUpDesigned", f"Follow-up for {base.id}", entity_type="experiment", entity_id=follow.id, rule="Uncertainty First", details={"parent": base.id})
        self._audit("ExperimentSelected", f"Selected follow-up {follow.name}", entity_type="experiment", entity_id=follow.id, rule="Uncertainty First", details={"parent": base.id})
        return follow

    def generate_report_data(self) -> dict[str, Any]:
        analyses = {eid: summary.to_dict() for eid, summary in self.analyze_all().items()}
        best = self.best_hypothesis()
        next_exp = self.select_next_experiment(SelectionStrategy.BEST_EXPECTED_INFORMATION)
        # Enrich result records with their immutable experiment/hypothesis
        # lineage.  The result row itself remains provider-neutral, while this
        # report projection makes every plotted observation traceable without
        # hidden joins in a renderer.
        experiment_by_id = {item.id: item for item in self.experiments}
        reporting_results: list[dict[str, Any]] = []
        for result in self.store.list_results():
            row = result.to_dict()
            run = self.store.get_run(result.run_id)
            experiment_id = result.experiment_id or (run.experiment_id if run else None)
            config = experiment_by_id.get(experiment_id or "")
            row["experiment_id"] = experiment_id
            if config is not None:
                row["hypothesis_id"] = config.hypothesis_id
                row["group"] = str(config.independent_variable)
                row["strategy"] = str(config.independent_variable)
            reporting_results.append(row)
        return {
            "research_question": self.project.research_question,
            "project": self.project.to_dict(),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "experiments": [e.to_dict() for e in self.experiments],
            "runs": [r.to_dict() for r in self.store.list_runs()],
            "results": reporting_results,
            "analyses": analyses,
            "best_hypothesis": best.to_dict() if best else None,
            "next_experiment": next_exp.to_dict() if next_exp else None,
            "budget": self.project.budget.to_dict(),
            "audit": [a.to_dict() for a in self.store.list_audit()],
            "journal": [j.to_dict() for j in self.store.list_journal()],
            "memory": [m.to_dict() for m in self.store.list_memory()],
            "limitations": ["Synthetic and mock providers are not evidence about the external world.", "Rule-based confidence updates are heuristics and require human interpretation."],
        }

    def best_hypothesis(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        status_rank = {HypothesisStatus.SUPPORTED: 3, HypothesisStatus.INCONCLUSIVE: 2, HypothesisStatus.UNTESTED: 1, HypothesisStatus.REJECTED: 0}
        return max(self.hypotheses, key=lambda h: (status_rank.get(h.status, 0), h.confidence))

    def snapshot(self) -> dict[str, Any]:
        return self.store.export_json()

    # ---------- desktop / CLI convenience adapters ----------
    def set_approval_mode(self, mode: str) -> str:
        normalised = str(mode).strip()
        if normalised not in {"Autonomous Sandbox", "Human Approval"}:
            raise ValueError("approval mode must be 'Autonomous Sandbox' or 'Human Approval'")
        self.approval_mode = normalised
        self._audit("ApprovalModeChanged", f"Approval mode set to {normalised}", entity_type="project", entity_id=self.project.id)
        return self.approval_mode

    def approve(self) -> None:
        """Record a human acknowledgement and resume a stopped local loop.

        Built-in providers are already safe to run in Human Approval mode;
        external adapters still require the explicit ``approval=True`` call on
        ``run_experiment``.  This method only controls the dashboard's stop
        token and does not spend any budget.
        """

        self.clear_stop()
        self._audit("HumanAcknowledged", "User acknowledged the next allowed action", entity_type="project", entity_id=self.project.id)

    def stop(self) -> None:
        self.request_stop()

    def run_next_experiment(self, *, runs: int = 1, approval: bool = False) -> ExperimentExecution | None:
        """Prepare and execute exactly one bounded candidate for the UI."""

        if self._stop_requested:
            raise StopRequested("Stop button is active")
        self.generate_hypotheses()
        self.design_experiments()
        candidate = self.select_next_experiment(SelectionStrategy.BEST_EXPECTED_INFORMATION)
        if candidate is None:
            candidate = self._make_follow_up()
        if candidate is None:
            return None
        return self.run_experiment(candidate, runs=max(1, int(runs)), approval=approval)

    def run_demo(self) -> dict[str, Any]:
        self.clear_stop()
        return self.run_study(first_round_runs=4, second_round_runs=3, second_round_count=2)

    run_demo_study = run_demo
    run_all = run_demo

    def generate_report(self, output_dir: str | Path | None = None) -> str:
        """Create a public Markdown report, optionally writing figures too."""

        from ..reporting import generate_report
        from ..visualization import save_charts

        context = self.generate_report_data()
        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            context["charts"] = save_charts(context["results"], destination, group_field="group", prefix="research")
        report = generate_report(context)
        if output_dir is not None:
            destination = Path(output_dir)
            (destination / "research_report.md").write_text(report.markdown, encoding="utf-8")
            (destination / "research_report.html").write_text(report.html, encoding="utf-8")
        return report.markdown

    def save(self, path: str | Path) -> Path:
        """Export the current research state as portable JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target

    save_state = save

    def load(self, path: str | Path) -> Project:
        """Import a JSON state export into the current store and resume it.

        Existing records with the same IDs are updated; unrelated records are
        preserved.  This makes import recoverable and avoids destructive
        replacement of a user workspace.
        """

        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        projects = [Project.from_dict(item) for item in payload.get("projects", [])]
        if not projects and isinstance(payload.get("project"), Mapping):
            projects = [Project.from_dict(payload["project"])]
        for item in projects:
            self.store.save_project(item)
        for item in payload.get("hypotheses", []):
            self.store.save_hypothesis(Hypothesis.from_dict(item))
        for item in payload.get("experiments", []):
            self.store.save_experiment(ExperimentConfig.from_dict(item))
        for item in payload.get("runs", []):
            self.store.save_run(ExperimentRun.from_dict(item))
        for item in payload.get("results", []):
            self.store.save_result(ExperimentResult.from_dict(item))
        if not projects:
            raise ValueError("state export contains no project")
        self.project = projects[-1]
        self.hypotheses = self.store.list_hypotheses(self.project.id)
        self.experiments = self.store.list_experiments(self.project.id)
        self._audit("StateImported", f"Imported saved state from {source.name}", entity_type="project", entity_id=self.project.id)
        return self.project

    load_state = load

    # ---------- audit helpers ----------
    def _audit(self, event_type: str, message: str, *, entity_type: str = "", entity_id: str = "", rule: str = "", details: Mapping[str, Any] | None = None) -> AuditEvent:
        return self.store.add_audit(AuditEvent(event_type=event_type, message=message, entity_type=entity_type, entity_id=entity_id, rule=rule, details=dict(details or {})))

    def _journal_for_experiment(self, config: ExperimentConfig, analysis: AnalysisSummary) -> JournalEntry:
        entry = JournalEntry(
            why=f"Test hypothesis {config.hypothesis_id or 'unlinked'} under the bounded project scope.",
            configuration=config.to_dict(),
            what_happened=f"Completed {analysis.n} usable runs and observed mean {analysis.mean:.3f} for {config.metric}." if analysis.n else "No usable result was produced.",
            result_summary=json.dumps(analysis.to_dict(), ensure_ascii=False, sort_keys=True),
            next_step="Run a bounded follow-up selected by uncertainty-first heuristic.",
            experiment_id=config.id,
        )
        return self.store.add_journal(entry)

    def _is_local_provider(self) -> bool:
        # PythonFunctionProvider is intentionally *not* whitelisted: a local
        # callable may itself perform network/file/service I/O, so Human
        # Approval must gate it.  Only the two guaranteed offline providers
        # can run automatically in approval mode.
        return isinstance(self.provider, (SyntheticProvider, MockLLMExperimentProvider))


def _hash_data(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ApprovalRequired",
    "ExperimentExecution",
    "RuleBasedScientist",
    "ScientificEngine",
    "StopRequested",
]
