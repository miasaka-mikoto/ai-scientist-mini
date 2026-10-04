"""High-level, auditable scientific-loop engine.

``ScientistEngine`` is the API consumed by the dashboard, CLI and future
adapters.  It coordinates a project, a local provider, statistical analysis,
budget enforcement, deduplication and restart-safe persistence without
depending on any external model API.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .analysis import ResultAnalyzer
from .models import (
    ApprovalMode,
    AuditEvent,
    BudgetExceeded,
    Evidence,
    Experiment,
    ExperimentDesign,
    ExperimentRun,
    FailureType,
    Hypothesis,
    HypothesisStatus,
    JournalEntry,
    ProjectStatus,
    QueueStatus,
    ResearchBudget,
    ResearchProject,
    SelectionStrategy,
    utc_now,
    make_id,
    project_from_dict,
    to_dict,
)
from .providers import (
    ExperimentProvider,
    MockLLMExperimentProvider,
    ProviderError,
    ProviderResult,
    SyntheticProvider,
)


class DuplicateExperiment(ValueError):
    """Raised when a new design duplicates an existing configuration."""


class ApprovalRequired(PermissionError):
    """Raised in Human Approval mode before an external resource is used."""


class StopRequested(RuntimeError):
    """Raised internally when the user presses the stop button."""


def _canonical_design(design: ExperimentDesign, *, include_seed: bool = True) -> dict[str, Any]:
    """Return fields defining experimental identity, excluding volatile IDs."""

    data = {
        "hypothesis_id": design.hypothesis_id,
        "name": design.name,
        "independent_variable": design.independent_variable,
        "dependent_variable": design.dependent_variable,
        "control": design.control,
        "dataset": design.dataset,
        "sample_size": design.sample_size,
        "metric": design.metric,
        "procedure": list(design.procedure),
        "success_criteria": design.success_criteria,
        "config": design.config,
    }
    if include_seed:
        data["seed"] = design.seed
    return data


def design_fingerprint(design: ExperimentDesign, *, include_seed: bool = True) -> str:
    payload = json.dumps(_canonical_design(design, include_seed=include_seed), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScientistEngine:
    """Run a bounded, reproducible scientific workflow."""

    code_version = "ai-scientist-mini-core/0.1"

    def __init__(
        self,
        project: ResearchProject | None = None,
        *,
        provider: ExperimentProvider | None = None,
        approval_mode: ApprovalMode | str | None = None,
        analyzer: ResultAnalyzer | None = None,
    ):
        self.project = project
        self.provider = provider or SyntheticProvider()
        self.analyzer = analyzer or ResultAnalyzer()
        self._stop_requested = False
        if self.project is not None and approval_mode is not None:
            self.project.approval_mode = ApprovalMode(approval_mode)

    # -- project setup --------------------------------------------------

    def create_project(
        self,
        research_question: str,
        *,
        name: str = "AI Scientist Study",
        scope: str = "",
        constraints: Iterable[str] = (),
        available_resources: Iterable[str] = (),
        metrics: Iterable[str] = ("score",),
        budget: ResearchBudget | None = None,
        max_experiments: int | None = None,
        approval_mode: ApprovalMode | str = ApprovalMode.AUTONOMOUS_SANDBOX,
    ) -> ResearchProject:
        if not str(research_question).strip():
            raise ValueError("research_question is required")
        budget = budget or ResearchBudget()
        if max_experiments is not None:
            budget.maximum_experiments = min(budget.maximum_experiments, int(max_experiments))
        self.project = ResearchProject(
            research_question=str(research_question).strip(),
            name=name,
            scope=scope,
            constraints=list(constraints),
            available_resources=list(available_resources),
            metrics=list(metrics) or ["score"],
            budget=budget,
            max_experiments=max_experiments if max_experiments is not None else budget.maximum_experiments,
            approval_mode=ApprovalMode(approval_mode),
            status=ProjectStatus.ACTIVE,
        )
        self._journal("project_created", "Research project created", "User supplied a research question")
        self._audit("project_created", self.project.id, "initialize project from explicit user input", {}, {"question": self.project.research_question})
        return self.project

    def require_project(self) -> ResearchProject:
        if self.project is None:
            raise RuntimeError("Create or load a project first")
        return self.project

    def set_provider(self, provider: ExperimentProvider) -> None:
        self.provider = provider

    # -- transparent scientist rules -----------------------------------

    def generate_background(self, question: str | None = None) -> str:
        question = question or self.require_project().research_question
        return (
            f"This sandbox study examines {question.strip()}. "
            "Background is generated from the question and synthetic assumptions; "
            "it is not a literature review or a claim about the real world."
        )

    def propose_hypotheses(self, question: str | None = None) -> list[Hypothesis]:
        project = self.require_project()
        question = question or project.research_question
        lower = question.lower()
        if "memory" in lower and ("agent" in lower or "long" in lower):
            candidates = [
                ("No Memory", "A stateless agent provides a transparent baseline.", 0.52),
                ("Sliding Window", "Recent context may preserve short-term continuity.", 0.66),
                ("Summary", "A compact summary may retain useful history with low context cost.", 0.73),
                ("Episodic", "Retrievable episodes may preserve salient long-term events.", 0.81),
                ("Vector", "Vector retrieval may recover semantically related history.", 0.77),
            ]
        else:
            candidates = [
                ("Baseline", "A simple baseline establishes a reference point.", 0.55),
                ("Conservative Variant", "A constrained variant may improve reliability.", 0.65),
                ("Enhanced Variant", "An enhanced variant may improve the selected metric.", 0.75),
            ]
        existing_statements = {h.statement for h in project.hypotheses}
        created: list[Hypothesis] = []
        for label, rationale, latent_mean in candidates:
            statement = f"{label} performs well for the study question."
            if statement in existing_statements:
                continue
            h = Hypothesis(
                statement=statement,
                rationale=rationale,
                predictions=[f"The {label} configuration will produce a measurable score."],
                test_method="Run the deterministic synthetic benchmark and compare the selected metric.",
                confidence=0.5,
                metadata={"strategy": label, "latent_mean_hint": latent_mean},
            )
            project.hypotheses.append(h)
            created.append(h)
            self._journal("hypothesis_proposed", f"Proposed hypothesis: {statement}", "Rule-based candidate generation from the research question", hypothesis_id=h.id)
            self._audit("hypothesis_proposed", h.id, "rule_based_candidate_template", {"question": question}, {"strategy": label})
        project.updated_at = utc_now()
        return created

    def design_experiment(
        self,
        hypothesis: Hypothesis | str,
        *,
        seed: int = 42,
        sample_size: int = 5,
        metric: str | None = None,
        baseline_strategy: str = "No Memory",
        success_threshold: float = 0.60,
    ) -> ExperimentDesign:
        project = self.require_project()
        if isinstance(hypothesis, str):
            found = project.hypothesis(hypothesis)
            if found is None:
                raise KeyError(f"Unknown hypothesis: {hypothesis}")
            hypothesis = found
        strategy = str(hypothesis.metadata.get("strategy", hypothesis.statement.split(" performs", 1)[0]))
        metric = metric or (project.metrics[0] if project.metrics else "score")
        return ExperimentDesign(
            hypothesis_id=hypothesis.id,
            name=f"{strategy} synthetic benchmark",
            independent_variable="memory_strategy",
            dependent_variable=metric,
            control=baseline_strategy,
            dataset="synthetic-memory-benchmark-v1",
            sample_size=sample_size,
            seed=seed,
            metric=metric,
            procedure=[
                "Initialize deterministic synthetic benchmark",
                f"Evaluate strategy: {strategy}",
                "Collect metric for each run",
                "Summarize results and compare with the control",
            ],
            success_criteria=f"mean({metric}) >= {success_threshold:.3f}",
            config={
                "strategy": strategy,
                "baseline_strategy": baseline_strategy,
                "success_threshold": float(success_threshold),
                "scientist_rule": "memory_strategy_template_v1",
            },
        )

    def create_experiment_plan(
        self,
        *,
        seed: int = 42,
        sample_size: int = 5,
        hypotheses: Iterable[Hypothesis] | None = None,
    ) -> list[ExperimentDesign]:
        project = self.require_project()
        hypotheses = list(hypotheses or project.hypotheses)
        return [self.design_experiment(h, seed=seed + i, sample_size=sample_size) for i, h in enumerate(hypotheses)]

    # -- queue and execution -------------------------------------------

    def queue_experiment(
        self,
        design: ExperimentDesign,
        *,
        allow_duplicate: bool = False,
        allow_new_seed: bool = False,
    ) -> Experiment:
        project = self.require_project()
        if not project.budget.can_start_experiment():
            raise BudgetExceeded("Maximum experiments reached")
        fp = design_fingerprint(design, include_seed=False)
        exact_fp = design_fingerprint(design, include_seed=True)
        same_design = [
            e for e in project.experiments
            if design_fingerprint(e.design, include_seed=False) == fp
        ]
        exact = next(
            (e for e in same_design if design_fingerprint(e.design, include_seed=True) == exact_fp),
            None,
        )
        if exact is not None and not allow_duplicate:
            raise DuplicateExperiment(f"Same configuration and seed already exists: {exact.id}")
        # A fresh seed is a legitimate follow-up only when the caller opts in.
        # This keeps ordinary UI retries idempotent while allowing the scientist
        # to collect additional independent evidence in a later selection round.
        if same_design and not allow_duplicate and not allow_new_seed:
            existing = same_design[0]
            raise DuplicateExperiment(f"Equivalent experiment already exists: {existing.id}")
        if same_design and not allow_duplicate and allow_new_seed:
            if any(e.status in {QueueStatus.QUEUED, QueueStatus.RUNNING} for e in same_design):
                raise DuplicateExperiment("A follow-up configuration is already queued or running")
        # A new experiment consumes an experiment slot immediately. Runs are
        # accounted for as they actually start, including failed runs.
        project.budget.reserve_experiment(0)
        experiment = Experiment(design=design)
        project.experiments.append(experiment)
        self._journal(
            "experiment_queued",
            f"Queued experiment: {design.name}",
            "Selected a non-duplicate design within the experiment budget",
            experiment_id=experiment.id,
            hypothesis_id=design.hypothesis_id,
        )
        self._audit(
            "experiment_queued",
            experiment.id,
            "deduplicate_by_configuration_and_seed",
            {"fingerprint": fp, "exact_fingerprint": exact_fp, "allow_new_seed": allow_new_seed},
            {"queued": True},
        )
        project.updated_at = utc_now()
        return experiment

    def queue_plan(self, designs: Iterable[ExperimentDesign], *, allow_duplicate: bool = False) -> list[Experiment]:
        queued: list[Experiment] = []
        for design in designs:
            try:
                queued.append(self.queue_experiment(design, allow_duplicate=allow_duplicate))
            except DuplicateExperiment:
                # Plans are idempotent by default, which is convenient when a
                # UI retries a button click after a restart.
                if allow_duplicate:
                    raise
        return queued

    def request_stop(self) -> None:
        self._stop_requested = True

    def clear_stop(self) -> None:
        self._stop_requested = False

    def _check_approval(self, approved: bool) -> None:
        project = self.require_project()
        if project.approval_mode == ApprovalMode.HUMAN_APPROVAL and not approved:
            raise ApprovalRequired("Human approval is required before running this provider")

    def _run_seed(self, design: ExperimentDesign, run_index: int) -> int:
        # Distinct, reproducible seeds per run while preserving the design seed.
        return int(design.seed) + int(run_index)

    def _existing_run(self, experiment: Experiment, seed: int) -> ExperimentRun | None:
        for run in experiment.runs:
            if run.seed == seed and run.status in {QueueStatus.COMPLETE, QueueStatus.FAILED}:
                return run
        return None

    def _make_run(self, experiment: Experiment, run_index: int, *, approved: bool = False) -> ExperimentRun:
        project = self.require_project()
        self._check_approval(approved)
        if self._stop_requested:
            raise StopRequested("Stop requested")
        if not project.budget.can_run(1):
            raise BudgetExceeded("Maximum runs reached")
        seed = self._run_seed(experiment.design, run_index)
        existing = self._existing_run(experiment, seed)
        if existing is not None:
            self._audit("run_deduplicated", existing.id, "same experiment configuration and seed already has a result", {"seed": seed}, {"run_id": existing.id})
            return existing
        run = ExperimentRun(
            experiment_id=experiment.id,
            run_index=run_index,
            seed=seed,
            status=QueueStatus.RUNNING,
            started_at=utc_now(),
            environment={"python": platform.python_version(), "platform": platform.platform()},
            code_version=self.code_version,
        )
        experiment.runs.append(run)
        try:
            result = self.provider.execute(experiment.design, seed=seed, run_index=run_index)
            if not isinstance(result, ProviderResult):
                # Be forgiving to simple third-party adapters returning a dict.
                if isinstance(result, Mapping):
                    result = ProviderResult(metrics={k: float(v) for k, v in result.items() if isinstance(v, (int, float))}, raw_result=dict(result))
                else:
                    raise ProviderError("Provider returned an unsupported result type", FailureType.INFRASTRUCTURE_FAILURE)
            run.metrics = dict(result.metrics)
            run.raw_result = dict(result.raw_result)
            run.logs.extend(result.logs)
            run.runtime_seconds = max(0.0, float(result.runtime_seconds))
            run.api_cost = max(0.0, float(result.api_cost))
            run.data_hash = result.data_hash
            run.environment.update(result.environment)
            run.status = QueueStatus.COMPLETE
            run.finished_at = utc_now()
            # Consume budget after execution, but preflight the result so an
            # over-budget provider cannot leave a misleading completed run.
            if not project.budget.can_spend(run.api_cost, run.runtime_seconds):
                raise BudgetExceeded("Provider result exceeds runtime or API-cost budget")
            project.budget.consume_run(run.runtime_seconds, run.api_cost)
            self._audit("run_completed", run.id, "provider_result_recorded", {"seed": seed}, {"metrics": run.metrics})
        except BudgetExceeded as exc:
            run.status = QueueStatus.FAILED
            run.failure_type = FailureType.INFRASTRUCTURE_FAILURE
            run.error = str(exc)
            run.finished_at = utc_now()
            # Budget limits are not a provider/infrastructure scientific result;
            # they still consume no additional run slot beyond the preflight.
            self._audit("run_blocked", run.id, "budget_guard", {}, {"error": str(exc)})
            raise
        except ProviderError as exc:
            run.status = QueueStatus.FAILED
            run.failure_type = exc.failure_type
            run.error = str(exc)
            run.finished_at = utc_now()
            # Failed provider attempts consume a bounded run slot.
            project.budget.consume_run(0.0, 0.0)
            self._audit("run_failed", run.id, exc.failure_type.value, {}, {"error": str(exc)})
        except Exception as exc:  # defensive boundary for adapter bugs
            run.status = QueueStatus.FAILED
            run.failure_type = FailureType.INFRASTRUCTURE_FAILURE
            run.error = str(exc)
            run.finished_at = utc_now()
            project.budget.consume_run(0.0, 0.0)
            self._audit("run_failed", run.id, FailureType.INFRASTRUCTURE_FAILURE.value, {}, {"error": str(exc)})
        return run

    def run_experiment(
        self,
        experiment: Experiment | str,
        *,
        runs: int | None = None,
        approved: bool = False,
    ) -> Experiment:
        project = self.require_project()
        if isinstance(experiment, str):
            experiment = project.experiment(experiment)
            if experiment is None:
                raise KeyError(f"Unknown experiment: {experiment}")
        if experiment.status == QueueStatus.CANCELLED:
            return experiment
        target = int(runs if runs is not None else experiment.design.sample_size)
        if target < 1:
            raise ValueError("runs must be positive")
        # Reject an over-sized request up front.  Silently running only a
        # prefix leaves the queue item in a misleading RUNNING state and
        # makes the run limit difficult to audit.
        if target > project.budget.remaining_runs or not project.budget.can_run(target):
            raise BudgetExceeded("Requested runs exceed remaining budget")
        experiment.status = QueueStatus.RUNNING
        experiment.started_at = experiment.started_at or utc_now()
        self._journal("experiment_started", f"Started {experiment.design.name}", "Queue selected the experiment", experiment_id=experiment.id, hypothesis_id=experiment.design.hypothesis_id)
        for i in range(target):
            if self._stop_requested:
                experiment.status = QueueStatus.CANCELLED
                self._journal("experiment_cancelled", "Experiment stopped by user", "Stop button requested cancellation", experiment_id=experiment.id)
                self._audit("experiment_cancelled", experiment.id, "user_stop_button", {}, {})
                return experiment
            if not project.budget.can_run(1):
                break
            try:
                self._make_run(experiment, i, approved=approved)
            except ApprovalRequired:
                # Approval denial is a gate, not a scientific failure.  Keep
                # the queue item resumable and prove that the provider was not
                # invoked.
                experiment.status = QueueStatus.QUEUED
                experiment.finished_at = None
                self._journal(
                    "approval_required",
                    "Human approval is required before this provider can run",
                    "Approval gate stopped execution before external resource use",
                    experiment_id=experiment.id,
                )
                self._audit("run_blocked", experiment.id, "human_approval_gate", {}, {"status": QueueStatus.QUEUED.value})
                project.updated_at = utc_now()
                raise
            except StopRequested:
                # A provider/UI callback may request a stop while a run is in
                # progress.  Keep the experiment restart-safe and report a
                # cancellation instead of leaking an exception with status
                # RUNNING.
                experiment.status = QueueStatus.CANCELLED
                self._journal("experiment_cancelled", "Experiment stopped by user", "Stop button requested cancellation", experiment_id=experiment.id)
                self._audit("experiment_cancelled", experiment.id, "user_stop_button", {}, {})
                experiment.finished_at = utc_now()
                project.updated_at = utc_now()
                return experiment
        complete = [r for r in experiment.runs if r.status == QueueStatus.COMPLETE]
        failed = [r for r in experiment.runs if r.status == QueueStatus.FAILED]
        metric = experiment.design.metric or (project.metrics[0] if project.metrics else "score")
        if complete:
            baseline = self._baseline_values(experiment, metric)
            threshold = (experiment.design.config or {}).get("success_threshold")
            try:
                threshold_value = float(threshold) if threshold is not None else None
            except (TypeError, ValueError):
                threshold_value = None
            experiment.analysis = self.analyzer.from_runs(
                complete,
                metric=metric,
                success_threshold=threshold_value,
                baseline_values=baseline,
            )
            experiment.status = QueueStatus.COMPLETE
            experiment.failure_type = None
            self._journal("experiment_completed", f"Collected {len(complete)} completed run(s)", "At least one run produced usable metrics", experiment_id=experiment.id, hypothesis_id=experiment.design.hypothesis_id, metadata={"analysis": to_dict(experiment.analysis)})
            if experiment.id not in project.memory.previous_experiments:
                project.memory.previous_experiments.append(experiment.id)
            project.memory.results.append({"experiment_id": experiment.id, "metric": metric, "mean": experiment.analysis.mean, "sample_count": experiment.analysis.sample_count})
        elif failed:
            experiment.status = QueueStatus.FAILED
            experiment.failure_type = failed[0].failure_type or FailureType.INFRASTRUCTURE_FAILURE
            experiment.error = failed[0].error
            self._journal("experiment_failed", f"Experiment failed: {experiment.error or 'unknown error'}", "No run produced usable metrics", experiment_id=experiment.id, hypothesis_id=experiment.design.hypothesis_id)
            if experiment.id not in project.memory.previous_experiments:
                project.memory.previous_experiments.append(experiment.id)
            project.memory.failed_ideas.append(self._strategy(experiment))
        experiment.finished_at = utc_now()
        if experiment.analysis is not None:
            self.update_hypothesis(experiment)
        project.updated_at = utc_now()
        return experiment

    def run_pending(self, *, max_experiments: int | None = None, runs_per_experiment: int | None = None, approved: bool = False) -> list[Experiment]:
        project = self.require_project()
        self.clear_stop()
        output: list[Experiment] = []
        limit = max_experiments if max_experiments is not None else len(project.experiments)
        for experiment in project.experiments:
            if len(output) >= limit:
                break
            if experiment.status not in {QueueStatus.QUEUED, QueueStatus.RUNNING}:
                continue
            try:
                output.append(self.run_experiment(experiment, runs=runs_per_experiment, approved=approved))
            except (BudgetExceeded, ApprovalRequired):
                raise
        return output

    # -- analysis and hypothesis update --------------------------------

    def _strategy(self, experiment: Experiment) -> str:
        cfg = experiment.design.config or {}
        return str(cfg.get("strategy", experiment.design.name))

    def _baseline_values(self, experiment: Experiment, metric: str) -> list[float]:
        baseline_name = str((experiment.design.config or {}).get("baseline_strategy", experiment.design.control or ""))
        if not baseline_name:
            return []
        values: list[float] = []
        for other in self.require_project().experiments:
            if other.id == experiment.id or other.status != QueueStatus.COMPLETE:
                continue
            if self._strategy(other).strip().lower() == baseline_name.strip().lower():
                values.extend(float(r.metrics[metric]) for r in other.runs if r.status == QueueStatus.COMPLETE and metric in r.metrics)
        return values

    def update_hypothesis(self, experiment: Experiment | str) -> Hypothesis:
        project = self.require_project()
        if isinstance(experiment, str):
            experiment = project.experiment(experiment)
            if experiment is None:
                raise KeyError("Unknown experiment")
        hypothesis = project.hypothesis(experiment.design.hypothesis_id)
        if hypothesis is None:
            raise KeyError(f"Unknown hypothesis: {experiment.design.hypothesis_id}")
        analysis = experiment.analysis
        if analysis is None or analysis.sample_count == 0:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            self._audit("hypothesis_updated", hypothesis.id, "insufficient_data", {"experiment_id": experiment.id}, {"status": hypothesis.status.value})
            return hypothesis
        baseline = analysis.baseline_mean
        threshold = float((experiment.design.config or {}).get("success_threshold", 0.60))
        ci = analysis.confidence_interval or (analysis.mean or 0.0, analysis.mean or 0.0)
        if baseline is not None:
            delta = float(analysis.mean or 0.0) - baseline
            support = ci[0] > baseline
            reject = ci[1] < baseline
            rule = "ci_above_or_below_control_mean"
        else:
            delta = float(analysis.mean or 0.0) - threshold
            support = ci[0] >= threshold
            reject = ci[1] < threshold
            rule = "ci_against_success_threshold"
        old_conf = hypothesis.confidence
        if support:
            hypothesis.status = HypothesisStatus.SUPPORTED
            hypothesis.confidence = min(0.99, max(old_conf, 0.5 + min(0.49, abs(delta))))
            claim = f"Observed {analysis.metric} supports the hypothesis under the transparent rule ({rule})."
            classification = "positive_result"
        elif reject:
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.confidence = max(0.01, min(old_conf, 0.5 - min(0.49, abs(delta))))
            claim = f"Observed {analysis.metric} is below the comparison criterion; the hypothesis is rejected by this experiment."
            classification = "negative_result"
        else:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            hypothesis.confidence = min(0.85, max(0.15, old_conf * 0.8 + 0.5 * 0.2))
            claim = "The confidence interval overlaps the comparison criterion; evidence is inconclusive."
            classification = "inconclusive_result"
        hypothesis.updated_at = utc_now()
        evidence = Evidence(
            hypothesis_id=hypothesis.id,
            experiment_id=experiment.id,
            run_ids=[r.id for r in experiment.runs if r.status == QueueStatus.COMPLETE],
            claim=claim,
            strength=min(1.0, abs(delta) + (0.1 if analysis.sample_count > 1 else 0.0)),
            rule=rule,
            classification=classification,
        )
        project.evidence.append(evidence)
        hypothesis.evidence.append(evidence.id)
        self._journal("hypothesis_updated", claim, f"Applied transparent rule: {rule}", experiment_id=experiment.id, hypothesis_id=hypothesis.id, metadata={"old_confidence": old_conf, "new_confidence": hypothesis.confidence, "rule": rule})
        self._audit(
            "hypothesis_updated",
            hypothesis.id,
            rule,
            {"experiment_id": experiment.id, "mean": analysis.mean, "ci": analysis.confidence_interval, "baseline": baseline, "threshold": threshold},
            {"status": hypothesis.status.value, "confidence": hypothesis.confidence, "evidence_id": evidence.id, "classification": classification},
        )
        return hypothesis

    # -- selection ------------------------------------------------------

    def select_next_experiment(
        self,
        strategy: SelectionStrategy | str = SelectionStrategy.UNCERTAINTY_FIRST,
        *,
        candidates: Iterable[Experiment] | None = None,
        seed: int = 0,
    ) -> Experiment | None:
        project = self.require_project()
        strategy = SelectionStrategy(strategy)
        pool = [e for e in (list(candidates) if candidates is not None else project.experiments) if e.status == QueueStatus.QUEUED]
        if not pool:
            return None
        if strategy == SelectionStrategy.RANDOM:
            return random.Random(seed).choice(pool)
        if strategy == SelectionStrategy.FOLLOW_UP_FAILED_RESULT:
            # A failed follow-up is selected from explicit evidence, never
            # inferred from a low score alone.  This keeps infrastructure or
            # invalid-design failures distinct from a valid negative result.
            failed_hypotheses = {
                evidence.hypothesis_id
                for evidence in project.evidence
                if evidence.classification in {"negative_result", "failed_result"}
                or "rejected" in evidence.claim.lower()
            }
            failed_hypotheses.update(
                experiment.design.hypothesis_id
                for experiment in project.experiments
                if experiment.status == QueueStatus.FAILED
            )
            for e in pool:
                if e.design.hypothesis_id in failed_hypotheses:
                    return e
            return pool[0]
        def uncertainty(e: Experiment) -> float:
            h = project.hypothesis(e.design.hypothesis_id)
            return 1.0 - abs((h.confidence if h else 0.5) - 0.5) * 2.0
        # Both information and uncertainty strategies use the same auditable
        # first heuristic in v1; information gets a small sample-size bonus.
        if strategy == SelectionStrategy.BEST_EXPECTED_INFORMATION:
            return max(pool, key=lambda e: (uncertainty(e) + 0.01 / max(1, e.design.sample_size), -e.design.seed))
        return max(pool, key=lambda e: (uncertainty(e), -e.design.seed))

    # Compatibility alias used by adapters.
    select_next_design = select_next_experiment

    def run_selected(
        self,
        strategy: SelectionStrategy | str = SelectionStrategy.UNCERTAINTY_FIRST,
        *,
        runs: int | None = None,
        approved: bool = False,
    ) -> Experiment | None:
        selected = self.select_next_experiment(strategy)
        return self.run_experiment(selected, runs=runs, approved=approved) if selected else None

    # -- audit/journal --------------------------------------------------

    def _journal(
        self,
        event_type: str,
        summary: str,
        why: str,
        *,
        experiment_id: str | None = None,
        hypothesis_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> JournalEntry:
        project = self.project
        entry = JournalEntry(event_type=event_type, summary=summary, why=why, experiment_id=experiment_id, hypothesis_id=hypothesis_id, metadata=dict(metadata or {}))
        if project is not None:
            project.journal.append(entry)
        return entry

    def _audit(self, event_type: str, subject_id: str, rule: str, inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> AuditEvent:
        event = AuditEvent(event_type=event_type, subject_id=subject_id, rule=rule, inputs=dict(inputs), outputs=dict(outputs))
        if self.project is not None:
            self.project.audit_trail.append(event)
        return event

    # -- persistence ----------------------------------------------------

    def save_state(self, path: str | os.PathLike[str]) -> Path:
        project = self.require_project()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(to_dict(project), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, target)
        return target

    @classmethod
    def load_state(
        cls,
        path: str | os.PathLike[str],
        *,
        provider: ExperimentProvider | None = None,
    ) -> "ScientistEngine":
        target = Path(path)
        project = project_from_dict(json.loads(target.read_text(encoding="utf-8")))
        return cls(project, provider=provider)

    # -- convenience demo cycle ---------------------------------------

    def run_scientific_cycle(
        self,
        *,
        rounds: int = 2,
        runs_per_experiment: int = 5,
        approved: bool = False,
        seed: int = 42,
    ) -> list[Experiment]:
        project = self.require_project()
        if not project.hypotheses:
            self.propose_hypotheses()
        if not project.experiments:
            self.queue_plan(self.create_experiment_plan(seed=int(seed), sample_size=runs_per_experiment))
        completed: list[Experiment] = []
        round_count = max(1, int(rounds))
        recorded_round = int(project.metadata.get("selection_round", 0) or 0)
        for round_index in range(1, round_count + 1):
            # Round one provides broad coverage.  Each later round deliberately
            # creates a small, auditable follow-up set: the current best arm and
            # the arm with the greatest uncertainty.  A fresh seed is required,
            # and the explicit allow_new_seed flag prevents accidental repeats.
            if round_index > 1 and recorded_round < round_index:
                available = list(project.hypotheses)
                if available:
                    means: dict[str, float] = {}
                    for item in project.experiments:
                        if item.analysis is not None and item.analysis.mean is not None:
                            means[item.design.hypothesis_id] = float(item.analysis.mean)
                    best = max(available, key=lambda h: (means.get(h.id, -1.0), h.id))
                    uncertain = max(
                        available,
                        key=lambda h: (1.0 - abs(float(h.confidence) - 0.5), h.id),
                    )
                    chosen: list[Hypothesis] = [best]
                    if uncertain.id != best.id:
                        chosen.append(uncertain)
                    for offset, hypothesis in enumerate(chosen):
                        design = self.design_experiment(
                            hypothesis,
                            seed=int(seed) + 1000 * round_index + offset + len(project.experiments),
                            sample_size=runs_per_experiment,
                        )
                        try:
                            self.queue_experiment(design, allow_new_seed=True)
                        except DuplicateExperiment:
                            # A restart may have already queued this follow-up;
                            # idempotent continuation should simply reuse it.
                            pass
                recorded_round = round_index
                project.metadata["selection_round"] = recorded_round
                self._journal(
                    "experiment_selection_round",
                    f"Prepared follow-up selection round {round_index}",
                    "Selected the current best and most uncertain hypotheses with fresh seeds",
                    metadata={"round": round_index, "strategy": SelectionStrategy.BEST_EXPECTED_INFORMATION.value},
                )
            while True:
                selected = self.select_next_experiment(SelectionStrategy.BEST_EXPECTED_INFORMATION)
                if selected is None:
                    break
                try:
                    completed.append(self.run_experiment(selected, runs=runs_per_experiment, approved=approved))
                except BudgetExceeded:
                    break
                if len(project.experiments) >= project.budget.maximum_experiments:
                    break
            if recorded_round < round_index:
                recorded_round = round_index
                project.metadata["selection_round"] = recorded_round
        project.status = ProjectStatus.COMPLETE if completed else project.status
        return completed

    # -- dashboard/service facade -------------------------------------

    def start_demo(self, *, seed: int = 42, runs_per_experiment: int = 4) -> dict[str, Any]:
        """Create the built-in Memory Strategy Study and queue round one."""

        self.create_project(
            "Which memory strategy is most suitable for a long-term Agent?",
            name="Memory Strategy Study",
            scope="Synthetic long-horizon memory benchmark",
            constraints=("No network or paid model API", "Deterministic seed", "Evidence is benchmark-scoped"),
            available_resources=("RuleBasedScientist", "SyntheticProvider", "Local Python runtime"),
            metrics=("score",),
            budget=ResearchBudget(
                maximum_experiments=10,
                maximum_runs=100,
                maximum_runtime_seconds=30,
                estimated_api_cost=0.0,
            ),
        )
        self.propose_hypotheses()
        self.queue_plan(self.create_experiment_plan(seed=seed, sample_size=runs_per_experiment))
        return self.snapshot()

    def create_study(self, question: str, **kwargs: Any) -> dict[str, Any]:
        """Dashboard-friendly alias for creating and planning a study."""

        self.create_project(question, **kwargs)
        self.propose_hypotheses()
        self.queue_plan(self.create_experiment_plan(seed=int(kwargs.get("seed", 42)), sample_size=int(kwargs.get("sample_size", 5))))
        return self.snapshot()

    def resume(self) -> dict[str, Any]:
        """Return the current checkpoint projection after a restart."""

        return self.snapshot()

    def run_next(self, *, runs: int | None = None, approved: bool = False) -> dict[str, Any]:
        selected = self.select_next_experiment(SelectionStrategy.BEST_EXPECTED_INFORMATION)
        if selected is not None:
            self.run_experiment(selected, runs=runs, approved=approved)
        return self.snapshot()

    def run_all(self, *, rounds: int = 2, runs_per_experiment: int = 4, approved: bool = False) -> dict[str, Any]:
        self.run_scientific_cycle(rounds=rounds, runs_per_experiment=runs_per_experiment, approved=approved)
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self.request_stop()
        return self.snapshot()

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        project = self.project
        if project is None:
            return {"nodes": [], "edges": []}
        nodes: list[dict[str, Any]] = [{"id": "Q", "label": "Question", "type": "question"}]
        edges: list[dict[str, Any]] = []
        for hypothesis in project.hypotheses:
            nodes.append({"id": hypothesis.id, "label": hypothesis.metadata.get("strategy", hypothesis.id), "type": "hypothesis"})
            edges.append({"source": "Q", "target": hypothesis.id})
        for experiment in project.experiments:
            nodes.append({"id": experiment.id, "label": experiment.design.name, "type": "experiment"})
            edges.append({"source": experiment.design.hypothesis_id, "target": experiment.id})
            if experiment.analysis is not None:
                result_id = f"R-{experiment.id}"
                nodes.append({"id": result_id, "label": "Result", "type": "result"})
                edges.append({"source": experiment.id, "target": result_id})
        for evidence in project.evidence:
            nodes.append({"id": evidence.id, "label": evidence.id, "type": "evidence"})
            edges.append({"source": f"R-{evidence.experiment_id}", "target": evidence.id})
            edges.append({"source": evidence.id, "target": evidence.hypothesis_id, "relation": "revises"})
        return {"nodes": nodes, "edges": edges}

    def snapshot(self) -> dict[str, Any]:
        """Return the JSON-like dashboard projection used by Tkinter/API."""

        project = self.project
        if project is None:
            return {"question": "", "status": "Idle", "hypotheses": [], "experiments": [], "evidence": [], "budget": {}, "graph": {"nodes": [], "edges": []}}
        enum_value = lambda value: getattr(value, "value", value)
        hypotheses = [
            {
                "id": h.id,
                "statement": h.statement,
                "confidence": h.confidence,
                "status": enum_value(h.status),
                "evidence": list(h.evidence),
                "metadata": dict(h.metadata),
            }
            for h in project.hypotheses
        ]
        experiments: list[dict[str, Any]] = []
        for item in project.experiments:
            analysis = item.analysis
            experiments.append(
                {
                    "id": item.id,
                    "hypothesis_id": item.design.hypothesis_id,
                    "strategy": item.design.config.get("strategy", item.design.name),
                    "status": enum_value(item.status),
                    "seed": item.design.seed,
                    "result": to_dict(analysis) if analysis is not None else {},
                    "failure_type": enum_value(item.failure_type) if item.failure_type else None,
                    "error": item.error,
                }
            )
        best = max(hypotheses, key=lambda row: float(row.get("confidence", 0.0)), default=None)
        queued = next((row for row in experiments if row.get("status") == QueueStatus.QUEUED.value), None)
        return {
            "question": project.research_question,
            "research_question": project.research_question,
            "status": enum_value(project.status),
            "hypotheses": hypotheses,
            "experiments": experiments,
            "evidence": [to_dict(evidence) for evidence in project.evidence],
            "best_hypothesis": best,
            "next_experiment": queued,
            "budget": {
                "used_experiments": project.budget.experiments_used,
                "max_experiments": project.budget.maximum_experiments,
                "used_runs": project.budget.runs_used,
                "max_runs": project.budget.maximum_runs,
                "runtime_seconds": project.budget.runtime_seconds_used,
                "max_runtime_seconds": project.budget.maximum_runtime_seconds,
                "estimated_api_cost": project.budget.actual_api_cost,
                "budget_cost": project.budget.estimated_api_cost,
            },
            "graph": self.graph(),
            "journal": [to_dict(entry) for entry in project.journal],
            "message": "Local rule-based/synthetic study; no external API used by default.",
        }

    dashboard_snapshot = snapshot

    def generate_report(self, output_path: str | os.PathLike[str] | None = None) -> str:
        from .reporting import write_project_report

        project = self.require_project()
        target = Path(output_path or "research_report.md")
        return str(write_project_report(project, target))
