from __future__ import annotations

from pathlib import Path

from aiscientist.core.models import (
    AuditEvent,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRun,
    Hypothesis,
    JournalEntry,
    MemoryRecord,
    Project,
    QueueItem,
)
from aiscientist.core.store import ResearchStore


def test_store_reopen_preserves_entities_queue_memory_and_audit(tmp_path: Path) -> None:
    db = tmp_path / "study.sqlite3"
    project = Project("Does episodic memory improve score?")
    hypothesis = Hypothesis("Episodic memory wins")
    config = ExperimentConfig(
        independent_variable="episodic",
        project_id=project.id,
        hypothesis_id=hypothesis.id,
        seed=7,
    )
    run = ExperimentRun(config.id, seed=7, config_hash=config.config_hash())
    result = ExperimentResult(run.id, metrics={"score": 0.8})
    run.result_id = result.id
    with ResearchStore(db) as store:
        store.save_project(project)
        store.save_hypothesis(hypothesis)
        store.save_experiment(config)
        store.save_run(run)
        store.save_result(result)
        store.enqueue(QueueItem(config.id))
        store.add_audit(
            AuditEvent(
                event_type="confidence_update",
                message="evidence recorded",
                entity_type="hypothesis",
                entity_id=hypothesis.id,
                rule="test_rule",
            )
        )
        store.add_journal(JournalEntry("test persistence", experiment_id=config.id))
        store.add_memory(MemoryRecord("observation", "episodic had the best score", [result.id]))

    with ResearchStore(db) as reopened:
        assert reopened.get_project(project.id).id == project.id
        assert reopened.get_hypothesis(hypothesis.id).statement == hypothesis.statement
        assert reopened.get_experiment(config.id).seed == 7
        assert reopened.get_run(run.id).result_id == result.id
        assert reopened.get_result(result.id).metrics["score"] == 0.8
        assert reopened.get_queue_item(config.id).experiment_id == config.id
        assert reopened.list_audit(entity_id=hypothesis.id)[0].rule == "test_rule"
        assert reopened.list_journal(experiment_id=config.id)
        assert reopened.search_memory("episodic")


def test_store_deduplicates_same_config_and_seed(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "dedup.sqlite3")
    try:
        first = ExperimentConfig("summary", project_id="p", seed=1)
        second = ExperimentConfig("summary", project_id="p", seed=1)
        third = ExperimentConfig("summary", project_id="p", seed=2)
        store.save_experiment(first)
        assert store.find_duplicate(second) is not None
        assert store.find_duplicate(third) is None
        assert store.find_duplicate(third, include_seed=False) is not None
    finally:
        store.close()

