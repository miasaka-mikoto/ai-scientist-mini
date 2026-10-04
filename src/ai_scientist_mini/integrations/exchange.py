"""JSON interchange, reproducibility hashes, and safe public export.

The exchange format is intentionally boring: UTF-8 JSON with a small envelope
and a schema version.  It can be consumed by a CLI, an HTTP client, or another
language without importing AI Scientist Mini.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from .contracts import API_VERSION


SCHEMA_PREFIX = "ai-scientist-mini"
FORMAT_VERSION = API_VERSION

# These names are deliberately explicit.  We never try to infer or serialize
# hidden model reasoning.  If a caller accidentally puts one in a record, its
# value is replaced in public exports.
PRIVATE_KEYS = frozenset({
    "chain_of_thought",
    "chain-of-thought",
    "private_reasoning",
    "private-reasoning",
    "hidden_reasoning",
    "internal_reasoning",
    "model_thoughts",
})


class ExchangeError(ValueError):
    """Invalid or incompatible interchange data."""


def _to_jsonable(value: Any, *, redact_private: bool = True) -> Any:
    """Convert common Python/domain values to JSON-safe values."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_string = str(key)
            if redact_private and key_string.lower() in PRIVATE_KEYS:
                result[key_string] = "[redacted: private reasoning]"
            else:
                result[key_string] = _to_jsonable(item, redact_private=redact_private)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item, redact_private=redact_private) for item in value]
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value), redact_private=redact_private)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump(), redact_private=redact_private)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict(), redact_private=redact_private)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Last-resort representation is preferable to failing an audit export.
    return str(value)


def canonical_json(value: Any, *, redact_private: bool = True) -> str:
    """Serialize a value deterministically for hashing and transport."""

    return json.dumps(
        _to_jsonable(value, redact_private=redact_private),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any, *, algorithm: str = "sha256", redact_private: bool = True) -> str:
    """Return a reproducible content hash of a JSON-compatible value."""

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"unsupported hash algorithm: {algorithm}") from exc
    digest.update(canonical_json(value, redact_private=redact_private).encode("utf-8"))
    return digest.hexdigest()


def make_envelope(kind: str, payload: Any, *, metadata: Mapping[str, Any] | None = None,
                  redact_private: bool = True) -> dict[str, Any]:
    """Create a versioned envelope used by all CLI/API exchanges."""

    if not kind or not isinstance(kind, str):
        raise ValueError("kind must be a non-empty string")
    normalized = _to_jsonable(payload, redact_private=redact_private)
    return {
        "schema": f"{SCHEMA_PREFIX}.{kind}",
        "schema_version": FORMAT_VERSION,
        "kind": kind,
        "payload": normalized,
        "metadata": _to_jsonable(dict(metadata or {}), redact_private=redact_private),
    }


def parse_envelope(value: Mapping[str, Any], *, expected_kind: str | None = None) -> dict[str, Any]:
    """Validate and return an envelope without mutating it."""

    if not isinstance(value, Mapping):
        raise ExchangeError("JSON root must be an object")
    schema = str(value.get("schema", ""))
    kind = str(value.get("kind", ""))
    version = str(value.get("schema_version", ""))
    if not schema.startswith(SCHEMA_PREFIX + "."):
        raise ExchangeError(f"unsupported schema: {schema or '<missing>'}")
    if not kind:
        # Older clients may only send schema; infer the suffix.
        kind = schema.split(".", 1)[1]
    if expected_kind and kind != expected_kind:
        raise ExchangeError(f"expected kind {expected_kind!r}, got {kind!r}")
    # Minor/patch-compatible versions can be handled by this module.  Reject
    # only an obviously incompatible major version rather than blocking a
    # harmless metadata difference.
    try:
        major = int(version.split(".", 1)[0])
        current_major = int(FORMAT_VERSION.split(".", 1)[0])
    except (ValueError, AttributeError):
        raise ExchangeError(f"invalid schema_version: {version!r}")
    if major != current_major:
        raise ExchangeError(f"incompatible schema version: {version}")
    if "payload" not in value:
        raise ExchangeError("envelope is missing payload")
    return {
        "schema": schema,
        "schema_version": version,
        "kind": kind,
        "payload": value["payload"],
        "metadata": value.get("metadata", {}),
    }


def export_json(value: Any, path: str | os.PathLike[str], *, kind: str = "record",
                metadata: Mapping[str, Any] | None = None, envelope: bool = True,
                redact_private: bool = True, indent: int = 2) -> Path:
    """Write a public JSON export atomically and return its path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = make_envelope(kind, value, metadata=metadata, redact_private=redact_private) if envelope \
        else _to_jsonable(value, redact_private=redact_private)
    text = json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    # NamedTemporaryFile in the destination directory makes os.replace atomic
    # on Windows as well as POSIX, and avoids half-written study checkpoints.
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def import_json(path: str | os.PathLike[str], *, expected_kind: str | None = None,
                require_envelope: bool = True) -> Any:
    """Read an export and return its payload.

    ``import_envelope`` is available when callers need schema metadata.  The
    payload-first API keeps normal CLI use concise.
    """

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ExchangeError(f"could not read JSON {source}: {exc}") from exc
    if not require_envelope:
        return value
    return parse_envelope(value, expected_kind=expected_kind)["payload"]


def import_envelope(path: str | os.PathLike[str], *, expected_kind: str | None = None) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExchangeError(f"could not read JSON {source}: {exc}") from exc
    return parse_envelope(value, expected_kind=expected_kind)


def export_study_state(state: Any, path: str | os.PathLike[str], *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Convenience boundary for a full project/checkpoint export."""

    return export_json(state, path, kind="study-state", metadata=metadata)


def import_study_state(path: str | os.PathLike[str]) -> Any:
    return import_json(path, expected_kind="study-state")


def export_experiment_request(request: Any, path: str | os.PathLike[str], *, metadata: Mapping[str, Any] | None = None) -> Path:
    return export_json(request, path, kind="experiment-request", metadata=metadata)


def export_experiment_result(result: Any, path: str | os.PathLike[str], *, metadata: Mapping[str, Any] | None = None) -> Path:
    return export_json(result, path, kind="experiment-result", metadata=metadata)

