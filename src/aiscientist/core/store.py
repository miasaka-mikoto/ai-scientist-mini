"""Small SQLite persistence layer for AI Scientist Mini.

The store uses one typed JSON table rather than a brittle web of relational
columns.  SQLite still gives us transactions, durable writes and safe resume,
while the JSON payload lets the domain models evolve without migrations for
every optional field.  The audit/event tables are append-only.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, Iterator, Mapping, TypeVar

from .models import (
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


T = TypeVar("T")


class ResearchStore:
    """Persist projects and their complete scientific audit trail.

    Parameters
    ----------
    path:
        SQLite file path.  ``None`` creates an in-memory store, useful for
        tests and the GUI's unsaved workspace.  Parent directories are created
        automatically for file-backed stores.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = ":memory:" if path is None else str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    kind TEXT NOT NULL,
                    id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(kind, id)
                );
                CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_events_kind ON events(event_kind);
                CREATE TABLE IF NOT EXISTS queue (
                    experiment_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            # Checkpoint WAL-backed study files before closing so a copied
            # demo/database is a single portable SQLite file rather than an
            # accidental trio of ``-wal``/``-shm`` sidecars.
            if self.path != ":memory:":
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
            self._conn.close()

    def __enter__(self) -> "ResearchStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ---------- generic JSON entity operations ----------
    def save_entity(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO entities(kind,id,payload,updated_at) VALUES(?,?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload, updated_at=CURRENT_TIMESTAMP",
                (kind, entity_id, encoded),
            )

    def get_entity(self, kind: str, entity_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM entities WHERE kind=? AND id=?", (kind, entity_id)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_entities(self, kind: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM entities WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def delete_entity(self, kind: str, entity_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM entities WHERE kind=? AND id=?", (kind, entity_id))

    # ---------- typed entity operations ----------
    def save_project(self, project: Project) -> Project:
        self.save_entity("project", project.id, project.to_dict())
        return project

    def get_project(self, project_id: str) -> Project | None:
        data = self.get_entity("project", project_id)
        return Project.from_dict(data) if data else None

    def list_projects(self) -> list[Project]:
        return [Project.from_dict(d) for d in self.list_entities("project")]

    def save_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        self.save_entity("hypothesis", hypothesis.id, hypothesis.to_dict())
        return hypothesis

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        data = self.get_entity("hypothesis", hypothesis_id)
        return Hypothesis.from_dict(data) if data else None

    def list_hypotheses(self, project_id: str | None = None) -> list[Hypothesis]:
        values = [Hypothesis.from_dict(d) for d in self.list_entities("hypothesis")]
        if project_id is None:
            return values
        return [h for h in values if h.project_id == project_id]

    def save_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        self.save_entity("experiment", config.id, config.to_dict())
        return config

    def get_experiment(self, experiment_id: str) -> ExperimentConfig | None:
        data = self.get_entity("experiment", experiment_id)
        return ExperimentConfig.from_dict(data) if data else None

    def list_experiments(self, project_id: str | None = None) -> list[ExperimentConfig]:
        values = [ExperimentConfig.from_dict(d) for d in self.list_entities("experiment")]
        return [e for e in values if project_id is None or e.project_id == project_id]

    def save_run(self, run: ExperimentRun) -> ExperimentRun:
        self.save_entity("run", run.id, run.to_dict())
        return run

    def get_run(self, run_id: str) -> ExperimentRun | None:
        data = self.get_entity("run", run_id)
        return ExperimentRun.from_dict(data) if data else None

    def list_runs(self, experiment_id: str | None = None) -> list[ExperimentRun]:
        values = [ExperimentRun.from_dict(d) for d in self.list_entities("run")]
        return [r for r in values if experiment_id is None or r.experiment_id == experiment_id]

    def save_result(self, result: ExperimentResult) -> ExperimentResult:
        self.save_entity("result", result.id, result.to_dict())
        return result

    def get_result(self, result_id: str) -> ExperimentResult | None:
        data = self.get_entity("result", result_id)
        return ExperimentResult.from_dict(data) if data else None

    def list_results(self, run_id: str | None = None) -> list[ExperimentResult]:
        values = [ExperimentResult.from_dict(d) for d in self.list_entities("result")]
        return [r for r in values if run_id is None or r.run_id == run_id]

    # ---------- queue ----------
    def enqueue(self, item: QueueItem) -> QueueItem:
        payload = json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO queue(experiment_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(experiment_id) DO UPDATE SET payload=excluded.payload,updated_at=CURRENT_TIMESTAMP",
                (item.experiment_id, payload),
            )
        return item

    def get_queue_item(self, experiment_id: str) -> QueueItem | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM queue WHERE experiment_id=?", (experiment_id,)).fetchone()
        return QueueItem(**json.loads(row["payload"])) if row else None

    def list_queue(self) -> list[QueueItem]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM queue ORDER BY updated_at, rowid").fetchall()
        return [QueueItem(**json.loads(row["payload"])) for row in rows]

    # ---------- append-only events and research memory ----------
    def add_audit(self, event: AuditEvent) -> AuditEvent:
        self._add_event("audit", event.to_dict(), timestamp=event.timestamp)
        return event

    def list_audit(self, *, entity_id: str | None = None) -> list[AuditEvent]:
        events = [AuditEvent(**d) for d in self._list_events("audit")]
        return [e for e in events if entity_id is None or e.entity_id == entity_id]

    def add_journal(self, entry: JournalEntry) -> JournalEntry:
        self._add_event("journal", entry.to_dict(), timestamp=entry.timestamp)
        return entry

    def list_journal(self, *, experiment_id: str | None = None) -> list[JournalEntry]:
        values = [JournalEntry(**d) for d in self._list_events("journal")]
        return [e for e in values if experiment_id is None or e.experiment_id == experiment_id]

    def add_memory(self, memory: MemoryRecord) -> MemoryRecord:
        self._add_event("memory", memory.to_dict(), timestamp=memory.created_at)
        return memory

    def list_memory(self, *, category: str | None = None) -> list[MemoryRecord]:
        values = [MemoryRecord(**d) for d in self._list_events("memory")]
        return [m for m in values if category is None or m.category == category]

    def search_memory(self, query: str) -> list[MemoryRecord]:
        query_l = query.lower()
        return [m for m in self.list_memory() if query_l in m.content.lower()]

    def _add_event(self, kind: str, payload: Mapping[str, Any], *, timestamp: str | None = None) -> None:
        encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events(event_kind,payload,timestamp) VALUES(?,?,COALESCE(?,CURRENT_TIMESTAMP))",
                (kind, encoded, timestamp),
            )

    def _list_events(self, kind: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM events WHERE event_kind=? ORDER BY id", (kind,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # ---------- deduplication helpers ----------
    def has_experiment_hash(self, config_hash: str, *, include_seed: bool = True) -> bool:
        for config in self.list_experiments():
            try:
                if config.config_hash(include_seed=include_seed) == config_hash:
                    return True
            except Exception:
                continue
        return False

    def find_duplicate(self, config: ExperimentConfig, *, include_seed: bool = True) -> ExperimentConfig | None:
        wanted = config.config_hash(include_seed=include_seed)
        for current in self.list_experiments(config.project_id):
            if current.config_hash(include_seed=include_seed) == wanted:
                return current
        return None

    def count_runs(self, project_id: str | None = None) -> int:
        runs = self.list_runs()
        if project_id is None:
            return len(runs)
        experiment_ids = {e.id for e in self.list_experiments(project_id)}
        return sum(r.experiment_id in experiment_ids for r in runs)

    def export_json(self) -> dict[str, Any]:
        """Return a complete portable snapshot for reports/debugging."""

        return {
            "projects": [p.to_dict() for p in self.list_projects()],
            "hypotheses": self.list_entities("hypothesis"),
            "experiments": self.list_entities("experiment"),
            "runs": self.list_entities("run"),
            "results": self.list_entities("result"),
            "audit": [e.to_dict() for e in self.list_audit()],
            "journal": [j.to_dict() for j in self.list_journal()],
            "memory": [m.to_dict() for m in self.list_memory()],
            "queue": [q.to_dict() for q in self.list_queue()],
        }


__all__ = ["ResearchStore"]
