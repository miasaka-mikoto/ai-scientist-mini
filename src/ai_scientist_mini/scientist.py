"""Rule-based scientist facade.

This module keeps candidate generation separate from execution so an adapter
can inspect or edit a plan before spending even a synthetic run.
"""

from __future__ import annotations

from typing import Iterable

from .engine import ScientistEngine
from .models import ExperimentDesign, Hypothesis, ResearchProject


class RuleBasedScientist:
    """Transparent planner that delegates to :class:`ScientistEngine` rules."""

    def __init__(self, project: ResearchProject | None = None):
        self.engine = ScientistEngine(project) if project is not None else ScientistEngine()

    def background(self, question: str) -> str:
        return self.engine.generate_background(question)

    def propose_hypotheses(self, question: str | None = None) -> list[Hypothesis]:
        if self.engine.project is None:
            self.engine.create_project(question or "", name="Rule-Based Study")
        elif question and question != self.engine.project.research_question:
            self.engine.project.research_question = question
        return self.engine.propose_hypotheses(question)

    # Common spelling used by integrations.
    generate_hypotheses = propose_hypotheses

    def design_experiments(
        self,
        hypotheses: Iterable[Hypothesis] | None = None,
        *,
        seed: int = 42,
        sample_size: int = 5,
    ) -> list[ExperimentDesign]:
        if self.engine.project is None:
            raise RuntimeError("Call propose_hypotheses first or provide a project")
        return self.engine.create_experiment_plan(seed=seed, sample_size=sample_size, hypotheses=hypotheses)

    def create_plan(self, question: str | None = None, *, seed: int = 42, sample_size: int = 5) -> list[ExperimentDesign]:
        if not self.engine.project:
            self.propose_hypotheses(question)
        elif not self.engine.project.hypotheses:
            self.propose_hypotheses(question)
        return self.design_experiments(seed=seed, sample_size=sample_size)

