"""Restart-safe project persistence.

The first release uses a human-readable JSON database.  It is intentionally
atomic (write a sibling temporary file, then replace) and stores the complete
research graph, journal and audit trail.  A future SQLite adapter can implement
the same tiny protocol without changing the engine/UI.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .models import ResearchProject, project_from_dict, to_dict


class JsonProjectStore:
    """Read/write one complete :class:`ResearchProject` JSON document."""

    format_name = "ai-scientist-mini-json-v1"

    def save(self, project: ResearchProject, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": self.format_name, "project": to_dict(project)}
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, target)
        return target

    def load(self, path: str | os.PathLike[str]) -> ResearchProject:
        target = Path(path)
        payload: Any = json.loads(target.read_text(encoding="utf-8"))
        # Accept both the wrapped database format and the bare engine format.
        if isinstance(payload, dict) and "project" in payload:
            payload = payload["project"]
        return project_from_dict(payload)


class ProjectRepository(JsonProjectStore):
    """Backwards-compatible repository name for adapters and the UI."""


class SQLiteProjectStore:
    """Small transactional SQLite checkpoint store.

    The domain object remains the source of truth and is stored as canonical
    JSON in one row, while audit events are also indexed in a separate table
    for cheap inspection.  This keeps the first release migration-friendly
    without coupling every model field to a hand-written SQL schema.
    """

    format_name = "ai-scientist-mini-sqlite-v1"

    @staticmethod
    def _connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(target))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS project_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                project_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                rule TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('format', ?)",
            (SQLiteProjectStore.format_name,),
        )
        connection.commit()
        return connection

    def save(self, project: ResearchProject, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        payload = json.dumps(to_dict(project), ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._connect(target) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO project_state(singleton, project_id, payload, updated_at)
                   VALUES(1, ?, ?, datetime('now'))
                   ON CONFLICT(singleton) DO UPDATE SET project_id=excluded.project_id,
                   payload=excluded.payload, updated_at=excluded.updated_at""",
                (project.id, payload),
            )
            # Rebuilding this small index is deterministic and makes a copied
            # checkpoint self-contained.  The append-only audit history remains
            # in the JSON payload even if a future schema adds incremental rows.
            connection.execute("DELETE FROM audit_events")
            for event in project.audit_trail:
                event_data = to_dict(event)
                connection.execute(
                    """INSERT INTO audit_events(event_id,event_type,subject_id,rule,payload,timestamp)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        str(event_data.get("id", "")),
                        str(event_data.get("event_type", "")),
                        str(event_data.get("subject_id", "")),
                        str(event_data.get("rule", "")),
                        json.dumps(event_data, ensure_ascii=False, sort_keys=True),
                        str(event_data.get("timestamp", "")),
                    ),
                )
            connection.commit()
        return target

    def load(self, path: str | os.PathLike[str]) -> ResearchProject:
        target = Path(path)
        with self._connect(target) as connection:
            row = connection.execute("SELECT payload FROM project_state WHERE singleton=1").fetchone()
        if row is None:
            raise FileNotFoundError(f"No project checkpoint in {target}")
        return project_from_dict(json.loads(str(row["payload"])))

    def audit_rows(self, path: str | os.PathLike[str]) -> list[dict[str, Any]]:
        target = Path(path)
        with self._connect(target) as connection:
            rows = connection.execute("SELECT payload FROM audit_events ORDER BY sequence").fetchall()
        return [json.loads(str(row["payload"])) for row in rows]


def save_project(project: ResearchProject, path: str | os.PathLike[str]) -> Path:
    return JsonProjectStore().save(project, path)


def load_project(path: str | os.PathLike[str]) -> ResearchProject:
    if Path(path).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return SQLiteProjectStore().load(path)
    return JsonProjectStore().load(path)


def save_sqlite(project: ResearchProject, path: str | os.PathLike[str]) -> Path:
    return SQLiteProjectStore().save(project, path)


def load_sqlite(path: str | os.PathLike[str]) -> ResearchProject:
    return SQLiteProjectStore().load(path)
