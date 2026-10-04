"""Public-facing analysis, journaling, graph and report helpers.

The reporting layer deliberately has no dependency on the execution engine.  It
accepts dictionaries, dataclasses, or ordinary objects and reads fields using a
small duck-typed adapter.  This keeps reports reproducible when the engine's
internal models evolve, and also makes the module useful from the CLI.

No model reasoning is persisted here.  Journal entries are decision summaries
and references to observable evidence only; keys commonly used for private
chain-of-thought are removed before anything is serialised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
from statistics import mean, median, stdev
import textwrap
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


__all__ = [
    "AnalysisResult",
    "ResultAnalyzer",
    "analyze_results",
    "ScientistJournal",
    "build_scientist_journal",
    "build_hypothesis_graph",
    "hypothesis_graph_to_mermaid",
    "render_hypothesis_graph",
    "save_hypothesis_graph",
    "ResearchReport",
    "render_report",
    "generate_report",
    "save_report",
    "build_demo_report",
    "generate_demo_report",
]


_MISSING = object()


def _get(value: Any, *names: str, default: Any = None) -> Any:
    """Read the first present field from a mapping or an object.

    ``to_dict`` is intentionally not required.  A few execution adapters use
    Pydantic-like objects, while the demo uses plain dictionaries.
    """

    for name in names:
        if value is None:
            continue
        if isinstance(value, Mapping) and name in value:
            return value[name]
        try:
            result = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if result is not None:
            return result
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _success_value(record: Any) -> bool | None:
    # Provider results often carry two different notions of success: the run
    # may have executed successfully while the measured task outcome failed
    # its criterion.  Prefer an explicit metric-level success rate when it is
    # present so the report does not turn every completed infrastructure run
    # into a scientific "pass".
    metrics = _get(record, "metrics", "measurements", "scores", default=None)
    if isinstance(metrics, Mapping):
        metric_value = metrics.get("success_rate", metrics.get("pass_rate", _MISSING))
        if metric_value is not _MISSING:
            numeric = _number(metric_value)
            if numeric is not None:
                return numeric >= 0.5
    value = _get(record, "success", "passed", "is_success", "ok", default=_MISSING)
    if value is _MISSING:
        status = str(_get(record, "status", "state", default="")).lower()
        if status in {"success", "succeeded", "passed", "complete", "completed"}:
            return True
        if status in {"failed", "failure", "error", "cancelled", "canceled"}:
            return False
        return None
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1", "pass", "passed", "success", "succeeded"}:
            return True
        if lower in {"false", "no", "0", "fail", "failed", "failure", "error"}:
            return False
    return bool(value) if value is not None else None


def _record_rows(records: Any, metric: str = "score") -> list[dict[str, Any]]:
    """Normalise result records into metric rows.

    Supported shapes include ``{"metric": 1.2}``,
    ``{"metrics": {"score": 1.2}}``, dataclass instances, and bare numbers.
    One record with several metrics is expanded only for the requested metric.
    """

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(_as_list(records)):
        if isinstance(record, (int, float)) and not isinstance(record, bool):
            value = _number(record)
            if value is not None:
                rows.append({"value": value, "index": index})
            continue
        metrics = _get(record, "metrics", "measurements", "scores", default=None)
        value = _get(record, metric, "value", "metric_value", "score", default=_MISSING)
        if value is _MISSING and isinstance(metrics, Mapping):
            value = metrics.get(metric, _MISSING)
            if value is _MISSING and len(metrics) == 1:
                value = next(iter(metrics.values()))
        numeric = _number(value)
        if numeric is None:
            continue
        row = {
            "value": numeric,
            "index": index,
            "record": record,
            "run_id": _get(record, "run_id", "id", default=None),
            "experiment_id": _get(record, "experiment_id", "experiment", default=None),
            "hypothesis_id": _get(record, "hypothesis_id", "hypothesis", "group", default=None),
            "group": _get(record, "group", "hypothesis_id", "hypothesis", default=None),
            "success": _success_value(record),
            "status": _get(record, "status", "state", default=None),
        }
        rows.append(row)
    return rows


def _analysis_dict(value: Any) -> dict[str, Any]:
    """Normalise either a mapping or a core ``AnalysisSummary`` object."""

    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            converted = value.to_dict()
            return dict(converted) if isinstance(converted, Mapping) else {}
        except TypeError:
            # Some summaries accept an ``include_values`` flag only as a
            # keyword; retrying without it keeps this adapter permissive.
            try:
                converted = value.to_dict(include_values=True)
                return dict(converted) if isinstance(converted, Mapping) else {}
            except Exception:
                return {}
        except Exception:
            return {}
    return {}


def _std(values: Sequence[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _z_critical(level: float) -> float:
    # A compact table avoids a SciPy dependency.  95% is the default and the
    # extra values cover common report settings.
    if level >= 0.999:
        return 3.291
    if level >= 0.99:
        return 2.576
    if level >= 0.98:
        return 2.326
    if level >= 0.90:
        return 1.645
    return 1.96


def _confidence_interval(values: Sequence[float], level: float = 0.95) -> dict[str, float | int]:
    values = list(values)
    n = len(values)
    if not n:
        return {"level": level, "low": None, "high": None, "margin": None, "n": 0}
    m = float(mean(values))
    margin = _z_critical(level) * (_std(values) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "level": level,
        "low": m - margin,
        "high": m + margin,
        "margin": margin,
        "n": n,
    }


def _cohens_d(values: Sequence[float], baseline: Sequence[float] | float | None) -> float | None:
    if baseline is None or not values:
        return None
    treatment = list(values)
    control = [float(baseline)] if isinstance(baseline, (int, float)) else [float(x) for x in baseline]
    if not control:
        return None
    pooled_num = max(len(treatment) - 1, 0) * (_std(treatment) ** 2)
    pooled_num += max(len(control) - 1, 0) * (_std(control) ** 2)
    denom_n = len(treatment) + len(control) - 2
    pooled = math.sqrt(pooled_num / denom_n) if denom_n > 0 else 0.0
    difference = float(mean(treatment) - mean(control))
    if pooled == 0:
        return 0.0 if difference == 0 else (math.inf if difference > 0 else -math.inf)
    return difference / pooled


@dataclass
class AnalysisResult:
    """Serializable summary returned by :class:`ResultAnalyzer`.

    It behaves like a small mapping for callers that historically expected a
    dictionary (``result["mean"]``), while retaining attribute access.
    """

    metric: str
    n: int
    mean: float | None
    median: float | None
    std: float | None
    confidence_interval: dict[str, Any]
    effect_size: float | None = None
    success_rate: float | None = None
    values: list[float] = field(default_factory=list)
    groups: dict[str, Any] = field(default_factory=dict)
    excluded: int = 0
    level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        data = {
            "metric": self.metric,
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "confidence_interval": _jsonable(self.confidence_interval),
            "effect_size": self.effect_size,
            "success_rate": self.success_rate,
            "values": list(self.values),
            "groups": _jsonable(self.groups),
            "excluded": self.excluded,
            "level": self.level,
        }
        ci = self.confidence_interval
        # Flat aliases are convenient for CSV/table consumers.
        data["ci_low"] = ci.get("low")
        data["ci_high"] = ci.get("high")
        return _jsonable(data)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


class ResultAnalyzer:
    """Compute transparent, dependency-free descriptive statistics."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = float(confidence_level)

    @staticmethod
    def confidence_interval(values: Sequence[float], level: float = 0.95) -> dict[str, Any]:
        return _confidence_interval(values, level)

    @staticmethod
    def effect_size(values: Sequence[float], baseline: Sequence[float] | float | None = None) -> float | None:
        return _cohens_d(values, baseline)

    @staticmethod
    def success_rate(records: Any) -> float | None:
        flags = [_success_value(record) for record in _as_list(records)]
        flags = [flag for flag in flags if flag is not None]
        return sum(flags) / len(flags) if flags else None

    def analyze(
        self,
        records: Any,
        metric: str = "score",
        baseline: Sequence[float] | float | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        rows = _record_rows(records, metric)
        values = [row["value"] for row in rows]
        ci = _confidence_interval(values, self.confidence_level)
        groups: dict[str, Any] = {}
        if group_by:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                raw = _get(row["record"], group_by, default=row.get(group_by))
                key = str(raw if raw is not None else "ungrouped")
                grouped.setdefault(key, []).append(row)
            for key, group_rows in grouped.items():
                groups[key] = self._summary_for_rows(group_rows, metric, baseline).to_dict()
        summary = AnalysisResult(
            metric=metric,
            n=len(values),
            mean=float(mean(values)) if values else None,
            median=float(median(values)) if values else None,
            std=_std(values) if values else None,
            confidence_interval=ci,
            effect_size=_cohens_d(values, baseline),
            success_rate=self.success_rate([row["record"] for row in rows]),
            values=values,
            groups=groups,
            excluded=max(0, len(_as_list(records)) - len(rows)),
            level=self.confidence_level,
        )
        return summary.to_dict()

    def _summary_for_rows(
        self,
        rows: Sequence[dict[str, Any]],
        metric: str,
        baseline: Sequence[float] | float | None = None,
    ) -> AnalysisResult:
        values = [row["value"] for row in rows]
        return AnalysisResult(
            metric=metric,
            n=len(values),
            mean=float(mean(values)) if values else None,
            median=float(median(values)) if values else None,
            std=_std(values) if values else None,
            confidence_interval=_confidence_interval(values, self.confidence_level),
            effect_size=_cohens_d(values, baseline),
            success_rate=self.success_rate([row["record"] for row in rows]),
            values=values,
            level=self.confidence_level,
        )

    def summarize(self, records: Any, metric: str = "score", **kwargs: Any) -> dict[str, Any]:
        return self.analyze(records, metric=metric, **kwargs)

    def compare_groups(
        self,
        records: Any,
        metric: str = "score",
        group_field: str = "hypothesis_id",
        baseline_group: str | None = None,
    ) -> dict[str, Any]:
        rows = _record_rows(records, metric)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            raw = _get(row["record"], group_field, default=row.get(group_field))
            key = str(raw if raw is not None else "ungrouped")
            grouped.setdefault(key, []).append(row)
        baseline_values = None
        if baseline_group is not None and baseline_group in grouped:
            baseline_values = [r["value"] for r in grouped[baseline_group]]
        summaries = {
            key: self._summary_for_rows(group_rows, metric, baseline_values).to_dict()
            for key, group_rows in grouped.items()
        }
        pairwise: dict[str, Any] = {}
        keys = list(grouped)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                pairwise[f"{left}__vs__{right}"] = _cohens_d(
                    [r["value"] for r in grouped[left]], [r["value"] for r in grouped[right]]
                )
        return {
            "metric": metric,
            "group_field": group_field,
            "baseline_group": baseline_group,
            "groups": summaries,
            "pairwise_effect_size": pairwise,
            "n": len(rows),
        }

    # Names used by older adapters.
    compare = compare_groups
    run = analyze


def analyze_results(
    records: Any,
    metric: str = "score",
    confidence_level: float = 0.95,
    baseline: Sequence[float] | float | None = None,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Functional convenience wrapper around :class:`ResultAnalyzer`."""

    return ResultAnalyzer(confidence_level).analyze(records, metric, baseline, group_by)


# ---------------------------------------------------------------------------
# Public decision journal


_PRIVATE_KEYS = {
    "chain_of_thought",
    "cot",
    "private_reasoning",
    "hidden_reasoning",
    "internal_reasoning",
    "scratchpad",
    "thoughts",
    "deliberation",
}


def _public_copy(value: Any, key: str = "") -> Any:
    key_lower = key.lower().replace("-", "_").replace(" ", "_")
    if key_lower in _PRIVATE_KEYS or any(token in key_lower for token in ("chain_of_thought", "scratchpad")):
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _public_copy(v, str(k))
            for k, v in value.items()
            if _public_copy(v, str(k)) is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [_public_copy(v, key) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _public_copy(value.to_dict(), key)
        except Exception:
            return str(value)
    if isinstance(value, str) and len(value) > 4000:
        return value[:3997] + "..."
    return _jsonable(value)


class ScientistJournal:
    """Append-only public audit trail for research decisions.

    Entries contain observable facts, configuration identifiers and the rule
    used for a decision.  They intentionally do not contain hidden model
    reasoning.
    """

    def __init__(self, entries: Iterable[Mapping[str, Any]] | None = None, study_id: str | None = None):
        self.study_id = study_id
        self.entries: list[dict[str, Any]] = []
        for entry in entries or []:
            if isinstance(entry, Mapping):
                payload = dict(entry)
            elif hasattr(entry, "to_dict") and callable(entry.to_dict):
                payload = entry.to_dict()
            else:
                payload = {
                    "timestamp": _get(entry, "timestamp", default=None),
                    "event_type": "experiment",
                    "summary": _get(entry, "why", "summary", default=""),
                    "details": {
                        "configuration": _get(entry, "configuration", default={}),
                        "what_happened": _get(entry, "what_happened", default=""),
                        "result_summary": _get(entry, "result_summary", default=""),
                        "next_step": _get(entry, "next_step", default=""),
                    },
                    "references": {"experiment_id": _get(entry, "experiment_id", default=None)},
                }
            if "summary" not in payload and "why" in payload:
                payload = {
                    "timestamp": payload.get("timestamp") or _now_iso(),
                    "event_type": "experiment",
                    "summary": payload.get("why") or "Recorded an experiment decision.",
                    "details": {
                        "configuration": payload.get("configuration", {}),
                        "what_happened": payload.get("what_happened", ""),
                        "result_summary": payload.get("result_summary", ""),
                        "next_step": payload.get("next_step", ""),
                    },
                    "references": {"experiment_id": payload.get("experiment_id")},
                }
            self.entries.append(_public_copy(payload))

    def append(
        self,
        event_type: str,
        summary: str,
        *,
        details: Mapping[str, Any] | None = None,
        refs: Mapping[str, Any] | Sequence[Any] | None = None,
        timestamp: str | None = None,
        rule: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "timestamp": timestamp or _now_iso(),
            "event_type": str(event_type),
            "summary": str(summary),
            "details": _public_copy(details or {}),
            "references": _public_copy(refs or {}),
        }
        if rule:
            entry["decision_rule"] = str(rule)
        self.entries.append(entry)
        return entry

    add = append
    log = append

    def record(
        self,
        *,
        why: str = "",
        configuration: Any = None,
        what_happened: str = "",
        result_summary: Any = None,
        next_step: str = "",
        experiment_id: str | None = None,
        rule: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility adapter for the core ``JournalEntry`` schema."""

        return self.append(
            "experiment",
            why or "Recorded an experiment decision.",
            details={
                "configuration": configuration or {},
                "what_happened": what_happened,
                "result_summary": result_summary,
                "next_step": next_step,
            },
            refs={"experiment_id": experiment_id},
            rule=rule,
        )

    add_entry = record

    def record_experiment(
        self,
        experiment: Any = None,
        result: Any = None,
        *,
        reason: str | None = None,
        next_step: str | None = None,
        rule: str | None = None,
    ) -> dict[str, Any]:
        experiment_id = _get(experiment, "experiment_id", "id", default="unknown-experiment")
        hypothesis_id = _get(experiment, "hypothesis_id", "hypothesis", default=None)
        status = _get(result, "status", "state", default=_get(experiment, "status", default="complete"))
        summary = reason or f"Recorded experiment {experiment_id} with observable status {status}."
        details = {
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "status": status,
            "configuration": _get(experiment, "config", "configuration", default=experiment),
            "result": result,
        }
        if next_step:
            details["next_step"] = next_step
        return self.append(
            "experiment",
            summary,
            details=details,
            refs={"experiment_id": experiment_id, "hypothesis_id": hypothesis_id},
            rule=rule,
        )

    def record_decision(
        self,
        summary: str,
        *,
        rule: str,
        references: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.append("decision", summary, details=details, refs=references, rule=rule)

    def record_hypothesis_update(
        self,
        hypothesis: Any,
        *,
        evidence: Any = None,
        rule: str = "transparent_threshold_update",
    ) -> dict[str, Any]:
        hid = _get(hypothesis, "id", "hypothesis_id", default="unknown-hypothesis")
        status = _get(hypothesis, "status", default="Inconclusive")
        confidence = _get(hypothesis, "confidence", default=None)
        return self.append(
            "hypothesis_update",
            f"Hypothesis {hid} updated to {status} based on recorded evidence.",
            details={"hypothesis_id": hid, "status": status, "confidence": confidence, "evidence": evidence},
            refs={"hypothesis_id": hid},
            rule=rule,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"study_id": self.study_id, "entries": _public_copy(self.entries)}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScientistJournal":
        return cls(data.get("entries", []), study_id=data.get("study_id"))

    def render_markdown(self) -> str:
        lines = ["## Scientist Journal", ""]
        if self.study_id:
            lines += [f"Study: `{self.study_id}`", ""]
        if not self.entries:
            lines.append("_No public decisions recorded._")
            return "\n".join(lines) + "\n"
        for entry in self.entries:
            stamp = entry.get("timestamp", "")
            event = entry.get("event_type", "event")
            lines.append(f"### {stamp} — {event}")
            lines.append(str(entry.get("summary", "")))
            if entry.get("decision_rule"):
                lines.append(f"- Decision rule: `{entry['decision_rule']}`")
            refs = entry.get("references") or {}
            if refs:
                lines.append(f"- References: `{json.dumps(refs, ensure_ascii=False, sort_keys=True)}`")
            details = entry.get("details") or {}
            if details:
                lines.append("```json")
                lines.append(json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True))
                lines.append("```")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        target = Path(path)
        fmt = (format or target.suffix.lstrip(".") or "json").lower()
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt in {"md", "markdown"}:
            target.write_text(self.render_markdown(), encoding="utf-8")
        else:
            target.write_text(self.to_json(), encoding="utf-8")
        return target


def build_scientist_journal(
    experiments: Iterable[Any] | None = None,
    results: Iterable[Any] | None = None,
    hypotheses: Iterable[Any] | None = None,
    *,
    study_id: str | None = None,
) -> ScientistJournal:
    """Build a journal from observable engine objects.

    This helper is intentionally deterministic apart from timestamps.  Engines
    may also call the journal methods incrementally during execution.
    """

    journal = ScientistJournal(study_id=study_id)
    result_by_exp: dict[str, list[Any]] = {}
    for result in results or []:
        rid = str(_get(result, "experiment_id", "experiment", default=""))
        result_by_exp.setdefault(rid, []).append(result)
    for experiment in experiments or []:
        eid = str(_get(experiment, "experiment_id", "id", default="unknown-experiment"))
        exp_results = result_by_exp.get(eid, [])
        if exp_results:
            for result in exp_results:
                journal.record_experiment(experiment, result)
        else:
            journal.record_experiment(experiment)
    for hypothesis in hypotheses or []:
        if _get(hypothesis, "status", default=None) not in (None, "Untested", "untested"):
            journal.record_hypothesis_update(hypothesis)
    return journal


# ---------------------------------------------------------------------------
# Hypothesis graph


def _node_id(prefix: str, value: Any, fallback: int | str = "0") -> str:
    text = str(value if value not in (None, "") else fallback)
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return f"{prefix}:{text}"


def build_hypothesis_graph(
    project: Any = None,
    hypotheses: Iterable[Any] | None = None,
    experiments: Iterable[Any] | None = None,
    results: Iterable[Any] | None = None,
    evidence: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return a serialisable Question → Hypothesis → Experiment → Result graph."""

    if project is not None:
        hypotheses = hypotheses if hypotheses is not None else _get(project, "hypotheses", default=[])
        experiments = experiments if experiments is not None else _get(project, "experiments", default=[])
        results = results if results is not None else _get(project, "results", "experiment_results", default=[])
    hs, exps, rs, evs = map(_as_list, (hypotheses, experiments, results, evidence))
    question = _get(project, "research_question", "question", "name", default="Research Question")
    nodes: list[dict[str, Any]] = [{"id": "question:root", "type": "question", "label": str(question)}]
    edges: list[dict[str, Any]] = []
    seen: set[str] = {"question:root"}

    def add(node: Mapping[str, Any]) -> None:
        if node["id"] not in seen:
            nodes.append(dict(node))
            seen.add(node["id"])

    for index, hypothesis in enumerate(hs):
        hid = _get(hypothesis, "id", "hypothesis_id", default=f"H{index + 1}")
        nid = _node_id("hypothesis", hid, index + 1)
        add({
            "id": nid,
            "type": "hypothesis",
            "label": str(_get(hypothesis, "statement", "name", default=hid)),
            "status": _get(hypothesis, "status", default="Untested"),
            "confidence": _get(hypothesis, "confidence", default=None),
        })
        edges.append({"source": "question:root", "target": nid, "type": "proposes"})

    hypothesis_ids = {str(_get(h, "id", "hypothesis_id", default=f"H{i + 1}")): _node_id("hypothesis", _get(h, "id", "hypothesis_id", default=f"H{i + 1}"), i + 1) for i, h in enumerate(hs)}
    hypothesis_objects = {
        str(_get(h, "id", "hypothesis_id", default=f"H{i + 1}")): h
        for i, h in enumerate(hs)
    }
    for index, experiment in enumerate(exps):
        eid = _get(experiment, "experiment_id", "id", default=f"E{index + 1}")
        enid = _node_id("experiment", eid, index + 1)
        hid = _get(experiment, "hypothesis_id", "hypothesis", default=None)
        add({
            "id": enid,
            "type": "experiment",
            "label": str(_get(experiment, "name", "title", default=eid)),
            "status": _get(experiment, "status", "state", default="Queued"),
            "config": _public_copy(_get(experiment, "config", "configuration", default=experiment)),
        })
        parent = hypothesis_ids.get(str(hid)) if hid is not None else None
        if parent:
            edges.append({"source": parent, "target": enid, "type": "tests"})
        else:
            edges.append({"source": "question:root", "target": enid, "type": "plans"})

    exp_ids = {str(_get(e, "experiment_id", "id", default=f"E{i + 1}")): _node_id("experiment", _get(e, "experiment_id", "id", default=f"E{i + 1}"), i + 1) for i, e in enumerate(exps)}
    exp_to_hyp = {
        str(_get(e, "experiment_id", "id", default=f"E{i + 1}")): _get(e, "hypothesis_id", "hypothesis", default=None)
        for i, e in enumerate(exps)
    }
    for index, result in enumerate(rs):
        rid = _get(result, "run_id", "result_id", "id", default=f"R{index + 1}")
        rnid = _node_id("result", rid, index + 1)
        eid = _get(result, "experiment_id", "experiment", default=None)
        value = _get(result, "value", "metric_value", "score", default=None)
        if value is None:
            metrics = _get(result, "metrics", "measurements", "scores", default=None)
            if isinstance(metrics, Mapping) and metrics:
                # The graph displays the first configured metric when the
                # result object does not carry an explicit scalar.
                value = next(iter(metrics.values()))
        status = _get(result, "status", "state", default="Complete")
        add({
            "id": rnid,
            "type": "result",
            "label": f"{rid}: {value if value is not None else status}",
            "status": status,
            "value": _number(value),
            "success": _success_value(result),
        })
        source = exp_ids.get(str(eid)) if eid is not None else None
        if source:
            edges.append({"source": source, "target": rnid, "type": "produces"})
        # Every result also gets an explicit evidence node so audit trails can
        # point to a concrete run rather than an inferred conclusion.
        evid_id = _node_id("evidence", rid, index + 1)
        evid_type = "supports" if _success_value(result) is True else "contradicts" if _success_value(result) is False else "inconclusive"
        add({"id": evid_id, "type": "evidence", "label": f"Evidence from {rid}", "evidence_type": evid_type, "result_id": rnid})
        edges.append({"source": rnid, "target": evid_id, "type": "evidences"})
        # Make the update step explicit.  It is a separate node (rather than
        # mutating the original hypothesis node) so the audit graph preserves
        # the temporal distinction between a prior and its evidence-informed
        # revision.
        hid = _get(result, "hypothesis_id", "hypothesis", default=None)
        if hid is None and eid is not None:
            hid = exp_to_hyp.get(str(eid))
        if hid is not None and str(hid) in hypothesis_objects:
            source_h = hypothesis_objects[str(hid)]
            revision_id = f"hypothesis_revision:{str(hid)}:{str(rid)}"
            add({
                "id": revision_id,
                "type": "revised_hypothesis",
                "label": str(_get(source_h, "statement", "name", default=hid)),
                "status": _get(source_h, "status", default="Inconclusive"),
                "confidence": _get(source_h, "confidence", default=None),
                "based_on": evid_id,
            })
            edges.append({"source": evid_id, "target": revision_id, "type": "updates"})
    for index, item in enumerate(evs):
        evid_id = _get(item, "id", "evidence_id", default=f"EV{index + 1}")
        enid = _node_id("evidence", evid_id, index + 1)
        add({"id": enid, "type": "evidence", "label": str(_get(item, "statement", "label", default=evid_id)), "evidence_type": _get(item, "type", default="observation")})
    return {"directed": True, "nodes": nodes, "edges": edges, "question": str(question)}


def hypothesis_graph_to_mermaid(graph: Mapping[str, Any]) -> str:
    """Render a graph dictionary as a compact Mermaid flowchart."""

    def quote(label: Any) -> str:
        return str(label).replace('"', "'").replace("\n", " ")

    lines = ["flowchart TD"]
    for node in graph.get("nodes", []):
        nid = str(node.get("id", "node")).replace(":", "_").replace("-", "_")
        label = quote(node.get("label", node.get("id", "")))
        lines.append(f'    {nid}["{label}"]')
    for edge in graph.get("edges", []):
        source = str(edge.get("source", "")).replace(":", "_").replace("-", "_")
        target = str(edge.get("target", "")).replace(":", "_").replace("-", "_")
        label = quote(edge.get("type", "relates"))
        lines.append(f"    {source} -->|{label}| {target}")
    return "\n".join(lines) + "\n"


def render_hypothesis_graph(graph: Mapping[str, Any], format: str = "mermaid") -> str:
    fmt = format.lower()
    if fmt in {"json", "dict"}:
        return json.dumps(_jsonable(graph), ensure_ascii=False, indent=2, sort_keys=True)
    if fmt in {"dot", "graphviz"}:
        lines = ["digraph hypothesis_graph {"]
        for node in graph.get("nodes", []):
            lines.append(f'  "{node.get("id", "")}" [label="{str(node.get("label", "")).replace(chr(34), chr(39))}"];')
        for edge in graph.get("edges", []):
            lines.append(f'  "{edge.get("source", "")}" -> "{edge.get("target", "")}" [label="{edge.get("type", "")}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"
    return hypothesis_graph_to_mermaid(graph)


def save_hypothesis_graph(
    graph: Mapping[str, Any],
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    """Persist a hypothesis graph as Mermaid, Graphviz DOT, or JSON."""

    target = Path(path)
    fmt = (format or target.suffix.lstrip(".") or "mmd").lower()
    if fmt in {"mmd", "mermaid"}:
        content = render_hypothesis_graph(graph, "mermaid")
    elif fmt in {"dot", "graphviz"}:
        content = render_hypothesis_graph(graph, "dot")
    elif fmt in {"json", "dict"}:
        content = render_hypothesis_graph(graph, "json")
    else:
        raise ValueError(f"Unsupported graph format: {format}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Report rendering


def _context(source: Any = None, **kwargs: Any) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if source is not None:
        if isinstance(source, Mapping):
            context.update(source)
        else:
            for key in ("project", "hypotheses", "experiments", "results", "journal", "analysis", "charts", "graph"):
                value = _get(source, key, default=_MISSING)
                if value is not _MISSING:
                    context[key] = value
            # A project-like object may itself be the project.
            if not context:
                context["project"] = source
    context.update({key: value for key, value in kwargs.items() if value is not None})
    project = context.get("project")
    for key, aliases in {
        "hypotheses": ("hypotheses",),
        "experiments": ("experiments", "experiment_queue"),
        "results": ("results", "experiment_results", "runs"),
    }.items():
        if key not in context and project is not None:
            context[key] = _get(project, *aliases, default=[])
    # ``ScientificEngine.generate_report_data`` stores runs and results in
    # separate lists.  Join them here so the report/graph can trace every
    # metric back to its experiment and hypothesis without changing the core
    # persistence schema.
    runs = _as_list(context.get("runs"))
    if runs and context.get("results"):
        experiments_by_id = {
            str(_get(e, "id", "experiment_id", default="")): e
            for e in _as_list(context.get("experiments"))
        }
        run_by_result = {
            str(_get(run, "result_id", default="")): run
            for run in runs
            if _get(run, "result_id", default=None)
        }
        enriched: list[Any] = []
        for result in _as_list(context.get("results")):
            if isinstance(result, Mapping):
                row = dict(result)
            elif hasattr(result, "to_dict"):
                row = dict(result.to_dict())
            else:
                row = {"value": result}
            run = run_by_result.get(str(row.get("id", row.get("result_id", ""))))
            if run is not None:
                row.setdefault("run_id", _get(run, "id", "run_id", default=None))
                row.setdefault("experiment_id", _get(run, "experiment_id", default=None))
            experiment = experiments_by_id.get(str(row.get("experiment_id", "")))
            if experiment is not None:
                row.setdefault("hypothesis_id", _get(experiment, "hypothesis_id", default=None))
                treatment = _get(experiment, "independent_variable", "name", default=None)
                row.setdefault("group", treatment)
            enriched.append(row)
        context["results"] = enriched
    # Queue status is stored separately from an immutable experiment design.
    # Add a read-only status projection for reports and graph consumers so a
    # completed experiment is not rendered as an unexplained em dash.
    if context.get("experiments"):
        runs_by_experiment: dict[str, list[Any]] = {}
        for run in runs:
            eid = str(_get(run, "experiment_id", "experiment", default=""))
            runs_by_experiment.setdefault(eid, []).append(run)
        result_experiment_ids = {
            str(_get(row, "experiment_id", "experiment", default=""))
            for row in _as_list(context.get("results"))
            if _get(row, "experiment_id", "experiment", default=None)
        }
        projected_experiments: list[Any] = []
        for experiment in _as_list(context.get("experiments")):
            if isinstance(experiment, Mapping):
                row = dict(experiment)
            elif hasattr(experiment, "to_dict"):
                row = dict(experiment.to_dict())
            else:
                row = {"id": str(experiment)}
            eid = str(_get(row, "id", "experiment_id", default=""))
            matching = runs_by_experiment.get(eid, [])
            raw_status = _get(row, "status", "state", default=None)
            status = getattr(raw_status, "value", raw_status)
            if matching:
                states = {getattr(_get(run, "status", "state", default=""), "value", _get(run, "status", "state", default="")) for run in matching}
                if "Running" in states:
                    status = "Running"
                elif "Failed" in states:
                    status = "Failed"
                elif "Cancelled" in states or "Canceled" in states:
                    status = "Cancelled"
                elif all(state == "Complete" for state in states):
                    status = "Complete"
            elif eid in result_experiment_ids:
                status = "Complete"
            elif not status:
                status = "Queued"
            row["status"] = status
            projected_experiments.append(row)
        context["experiments"] = projected_experiments
    # Existing engine reports call this field ``analyses`` (one summary per
    # experiment).  Prefer recomputing one aggregate from the enriched result
    # rows when they are available.  Simply keying per-experiment summaries by
    # treatment silently overwrites repeated/follow-up experiments (for
    # example, a Sliding Window follow-up), which would make the report under-
    # count evidence while the database still contains every run.
    if "analysis" not in context:
        metric = "score"
        project_metrics = _get(project, "metrics", default=None)
        if project_metrics:
            metric = str(project_metrics[0])
        if context.get("results"):
            context["analysis"] = ResultAnalyzer().compare_groups(
                context["results"],
                metric=metric,
                group_field="group",
                baseline_group="No Memory" if any(
                    str(_get(row, "group", default="")) == "No Memory"
                    for row in _as_list(context.get("results"))
                ) else None,
            )
        elif isinstance(context.get("analyses"), Mapping):
            summaries = []
            for experiment in _as_list(context.get("experiments")):
                eid = str(_get(experiment, "id", "experiment_id", default=""))
                summary = context["analyses"].get(eid)
                if summary is None:
                    continue
                row = dict(_analysis_dict(summary))
                row["group"] = _get(experiment, "independent_variable", "name", default=eid)
                summaries.append(row)
            if summaries:
                context["analysis"] = {
                    "metric": metric,
                    "groups": {
                        str(row["group"]): {k: v for k, v in row.items() if k != "group"}
                        for row in summaries
                    },
                }
    # A store journal is a list of typed JournalEntry objects rather than the
    # public-journal envelope used by this module.
    if isinstance(context.get("journal"), list):
        converted = []
        for entry in context["journal"]:
            converted.append({
                "timestamp": _get(entry, "timestamp", default=_now_iso()),
                "event_type": "experiment",
                "summary": _get(entry, "why", default="Recorded experiment decision."),
                "details": {
                    "configuration": _get(entry, "configuration", default={}),
                    "what_happened": _get(entry, "what_happened", default=""),
                    "result_summary": _get(entry, "result_summary", default=""),
                    "next_step": _get(entry, "next_step", default=""),
                },
                "references": {"experiment_id": _get(entry, "experiment_id", default=None)},
            })
        context["journal"] = {"entries": converted}
    return context


def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    if not rows:
        return "_None recorded._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(out) + "\n"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "∞" if value > 0 else "−∞"
        return f"{value:.{digits}f}"
    return str(value)


@dataclass
class ResearchReport:
    """A rendered report plus its structured source data."""

    markdown: str
    html: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"markdown": self.markdown, "html": self.html, "data": _jsonable(self.data)}


def _render_markdown(context: Mapping[str, Any], *, title: str | None = None, include_graph: bool = True) -> tuple[str, dict[str, Any]]:
    project = context.get("project")
    question = _get(project, "research_question", "question", "name", default=context.get("research_question", "Research Question"))
    hypotheses = _as_list(context.get("hypotheses"))
    experiments = _as_list(context.get("experiments"))
    results = _as_list(context.get("results"))
    analysis = context.get("analysis")
    if analysis is None:
        analysis = ResultAnalyzer().compare_groups(results) if results else ResultAnalyzer().analyze(results)
    analysis = _analysis_dict(analysis)
    graph = context.get("graph") or build_hypothesis_graph(project, hypotheses, experiments, results)
    journal = context.get("journal")
    if not isinstance(journal, ScientistJournal):
        journal = ScientistJournal.from_dict(journal) if isinstance(journal, Mapping) and "entries" in journal else build_scientist_journal(experiments, results, hypotheses, study_id=_get(project, "id", "project_id", default=None))

    report_title = title or _get(project, "name", "title", default="AI Scientist Mini Research Report")
    lines = [f"# {report_title}", "", f"**Research Question:** {question}", ""]
    lines += ["## Research Question", "", str(question), ""]
    lines += ["## Hypotheses", ""]
    hrows = []
    for h in hypotheses:
        hrows.append([
            _get(h, "id", "hypothesis_id", default="—"),
            _get(h, "statement", "name", default="—"),
            _get(h, "status", default="Untested"),
            _fmt(_get(h, "confidence", default=None)),
        ])
    lines.append(_table(hrows, ["ID", "Statement", "Status", "Confidence"]))

    lines += ["## Methods", "", "Experiments were executed within the configured sandbox and budget. Statistics use descriptive mean, median, sample standard deviation, and a normal-approximation confidence interval; hypothesis updates are rule-based and auditable.", ""]
    lines += ["## Experiments", ""]
    erows = []
    for e in experiments:
        erows.append([
            _get(e, "experiment_id", "id", default="—"),
            _get(e, "hypothesis_id", "hypothesis", default="—"),
            _get(e, "status", "state", default="—"),
            _get(e, "seed", default="—"),
        ])
    lines.append(_table(erows, ["Experiment", "Hypothesis", "Status", "Seed"]))

    lines += ["## Results", ""]
    if "groups" in analysis and isinstance(analysis.get("groups"), Mapping):
        rrows = []
        for group, summary in analysis["groups"].items():
            rrows.append([group, summary.get("n", 0), _fmt(summary.get("mean")), _fmt(summary.get("median")), _fmt(summary.get("std")), _fmt(summary.get("success_rate"))])
        lines.append(_table(rrows, ["Group", "N", "Mean", "Median", "Std", "Success Rate"]))
    else:
        lines.append(_table([[analysis.get("n", 0), _fmt(analysis.get("mean")), _fmt(analysis.get("median")), _fmt(analysis.get("std")), _fmt(analysis.get("success_rate"))]], ["N", "Mean", "Median", "Std", "Success Rate"]))

    lines += ["## Figures", ""]
    charts = context.get("charts") or []
    if isinstance(charts, Mapping):
        charts = list(charts.values())
    if charts:
        for chart in charts:
            lines.append(f"- [{Path(str(chart)).name}]({chart})")
    else:
        lines.append("Charts can be generated with `save_charts(analysis, output_dir)`.")
    lines.append("")

    lines += ["## Interpretation", ""]
    groups = analysis.get("groups", {}) if isinstance(analysis, Mapping) else {}
    if groups:
        ranked = sorted(((k, v) for k, v in groups.items() if v.get("mean") is not None), key=lambda item: item[1]["mean"], reverse=True)
        if ranked:
            best, best_summary = ranked[0]
            lines.append(f"The highest observed mean for `{best}` was {_fmt(best_summary.get('mean'))} across {best_summary.get('n', 0)} run(s). This is an observed comparison, not proof of universal superiority.")
        else:
            lines.append("No group had enough numeric observations for an interpretation.")
    elif analysis.get("n", 0):
        lines.append(f"The observed mean was {_fmt(analysis.get('mean'))} across {analysis.get('n')} run(s); this summary is descriptive.")
    else:
        lines.append("No numeric observations were available for interpretation.")
    lines.append("")

    lines += ["## Rejected Hypotheses", ""]
    rejected = [h for h in hypotheses if str(_get(h, "status", default="")).lower() == "rejected"]
    if rejected:
        lines.extend([f"- `{_get(h, 'id', 'hypothesis_id', default='—')}`: {_get(h, 'statement', default='—')}" for h in rejected])
    else:
        lines.append("None recorded.")
    lines.append("")

    lines += ["## Limitations", "", "- Synthetic or mock providers do not establish real-world validity.", "- Confidence intervals use a simple normal approximation and should not be over-interpreted for small samples.", "- Failed infrastructure runs are kept distinct from negative hypothesis evidence.", ""]
    lines += ["## Next Work", "", "The next experiment should be selected by the configured policy (information gain, uncertainty, random, or follow-up of a failed result) while respecting the remaining budget.", ""]
    if include_graph:
        lines += ["## Hypothesis Graph", "", "```mermaid", hypothesis_graph_to_mermaid(graph).rstrip(), "```", ""]
    lines.append(journal.render_markdown().rstrip())
    markdown = "\n".join(lines).rstrip() + "\n"
    data = {
        "research_question": question,
        "hypotheses": [_public_copy(h) for h in hypotheses],
        "experiments": [_public_copy(e) for e in experiments],
        "results": [_public_copy(r) for r in results],
        "analysis": _public_copy(analysis),
        "graph": _public_copy(graph),
        "journal": journal.to_dict(),
        "charts": _public_copy(charts),
    }
    return markdown, data


def _markdown_to_html(markdown: str, *, title: str = "AI Scientist Mini Research Report") -> str:
    """Small Markdown subset renderer with no external dependency."""

    out: list[str] = []
    in_code = False
    code_lang = ""
    table_buffer: list[str] = []
    in_ul = False

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        rows = [line.strip().strip("|").split("|") for line in table_buffer]
        if len(rows) >= 2:
            out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c.strip())}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for row in rows[2:] if all(set(c.strip()) <= {"-", " "} for c in rows[1]) else rows[1:]:
                out.append("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
        table_buffer = []

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            flush_table()
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                code_lang = line[3:].strip()
                out.append(f'<pre><code class="language-{html.escape(code_lang)}">')
                in_code = True
            continue
        if in_code:
            out.append(html.escape(line) + "\n")
            continue
        if line.startswith("|"):
            table_buffer.append(line)
            continue
        flush_table()
        if not line:
            continue
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            level = len(match.group(1))
            out.append(f"<h{level}>{html.escape(match.group(2))}</h{level}>")
        elif line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            # Basic inline code and bold/links.
            escaped = html.escape(line)
            escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
            escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
            escaped = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
            out.append(f"<p>{escaped}</p>")
    flush_table()
    if in_ul:
        out.append("</ul>")
    body = "\n".join(out)
    return "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>" + html.escape(title) + "</title>" + "<style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1f2937}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d1d5db;padding:.4rem;text-align:left}th{background:#f3f4f6}pre{background:#111827;color:#e5e7eb;padding:1rem;overflow:auto}code{font-family:ui-monospace,monospace}a{color:#2563eb}@media print{body{max-width:none}}</style></head><body>" + body + "</body></html>"


def render_report(
    source: Any = None,
    *,
    format: str = "markdown",
    title: str | None = None,
    include_graph: bool = True,
    **kwargs: Any,
) -> str | ResearchReport:
    """Render a complete report in Markdown, HTML, or print-ready PDF-ish HTML.

    ``source`` may be a project/study object or a context mapping.  Passing
    ``return_object=True`` in ``kwargs`` returns :class:`ResearchReport`.
    """

    return_object = bool(kwargs.pop("return_object", False))
    context = _context(source, **kwargs)
    markdown, data = _render_markdown(context, title=title, include_graph=include_graph)
    html_text = _markdown_to_html(markdown, title=title or "AI Scientist Mini Research Report")
    fmt = format.lower()
    if fmt in {"md", "markdown", "txt"}:
        rendered: str | ResearchReport = markdown
    elif fmt in {"html", "htm", "pdf", "pdf-ish", "print"}:
        rendered = html_text
    elif fmt in {"json", "data"}:
        rendered = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        raise ValueError(f"Unsupported report format: {format}")
    if return_object:
        return ResearchReport(markdown=markdown, html=html_text, data=data)
    return rendered


def generate_report(source: Any = None, **kwargs: Any) -> ResearchReport:
    """Return both Markdown and HTML report forms."""

    kwargs["return_object"] = True
    return render_report(source, **kwargs)  # type: ignore[return-value]


def save_report(
    source: Any,
    path: str | Path,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> Path:
    target = Path(path)
    fmt = (format or target.suffix.lstrip(".") or "md").lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pdf":
        # ReportLab is optional.  The source distribution remains dependency
        # light, but when it is available we provide a genuine PDF rather than
        # merely renaming the print-ready HTML.
        report = generate_report(source, **kwargs)
        if _save_pdf(report.markdown, target, title=kwargs.get("title")):
            return target
        # Dependency-free fallback: keep the requested extension a valid PDF
        # even in the small PyInstaller build that excludes ReportLab.
        if _save_minimal_pdf(report.markdown, target, title=kwargs.get("title")):
            return target
        # Last-resort fallback remains inspectable HTML rather than failing a
        # study solely because a renderer is unavailable.
        target.write_text(report.html, encoding="utf-8")
        return target
    rendered = render_report(source, format=fmt, **kwargs)
    if isinstance(rendered, ResearchReport):
        rendered = rendered.markdown
    # Non-PDF formats are written as their requested textual representation.
    target.write_text(str(rendered), encoding="utf-8")
    return target


def _save_pdf(markdown: str, target: Path, *, title: str | None = None) -> bool:
    """Write a modest text-first PDF when ReportLab is installed.

    This intentionally keeps the conversion deterministic and avoids invoking
    a browser or a subprocess.  Figures remain available as SVG artefacts and
    are linked by the Markdown/HTML versions.
    """

    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except Exception:
        return False

    try:
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT))
        story: list[Any] = []
        in_code = False
        code_lines: list[str] = []
        for raw in markdown.splitlines():
            line = raw.rstrip()
            if line.startswith("```"):
                if in_code:
                    story.append(Preformatted("\n".join(code_lines), styles["Code"]))
                    story.append(Spacer(1, 3 * mm))
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue
            if not line:
                story.append(Spacer(1, 2 * mm))
                continue
            if line.startswith("# "):
                story.append(Paragraph(html.escape(line[2:]), styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(html.escape(line[3:]), styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(html.escape(line[4:]), styles["Heading3"]))
            elif line.startswith("- "):
                story.append(Paragraph("• " + html.escape(line[2:]), styles["ReportSmall"]))
            elif line.startswith("|"):
                # Keep table rows readable in the text-first PDF; rich table
                # layout is available in the HTML report.
                story.append(Paragraph(html.escape(line), styles["ReportSmall"]))
            else:
                story.append(Paragraph(html.escape(line), styles["ReportSmall"]))
        doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title or "AI Scientist Mini Research Report")
        doc.build(story)
        return target.exists() and target.stat().st_size > 0
    except Exception:
        return False


def _pdf_escape(text: str) -> bytes:
    """Encode one text fragment for a built-in Helvetica PDF stream."""

    safe = str(text).encode("latin-1", "replace").decode("latin-1")
    safe = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("latin-1")


def _save_minimal_pdf(markdown: str, target: Path, *, title: str | None = None) -> bool:
    """Write a valid text-first PDF using only the Python standard library.

    This is intentionally modest: SVG/PNG figures remain separate report
    artefacts, while the PDF itself stays portable in a dependency-free
    executable.  It is preferable to a misleading HTML file named ``.pdf``.
    """

    try:
        logical_lines: list[str] = []
        for raw in str(markdown).splitlines():
            line = raw.strip()
            if line.startswith("```"):
                continue
            # Markdown tables are useful as text in a print-first fallback.
            if line.startswith("|"):
                line = line.replace("|", "  ")
            if line:
                logical_lines.extend(textwrap.wrap(line, width=102, break_long_words=False) or [""])
            else:
                logical_lines.append("")
        page_lines = 48
        pages = [logical_lines[i : i + page_lines] for i in range(0, max(1, len(logical_lines)), page_lines)]

        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")  # 1
        # Each page contributes a content stream followed by a page object;
        # the first four objects above are catalog/pages/font/info.
        page_object_ids = [6 + 2 * index for index in range(len(pages))]
        kids = " ".join(f"{oid} 0 R" for oid in page_object_ids)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))  # 2
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")  # 3
        objects.append(b"<< /Producer (AI Scientist Mini) /Title (AI Scientist Mini Research Report) >>")  # 4
        for index, lines in enumerate(pages):
            content_parts = [b"BT", b"/F1 9 Tf", b"50 770 Td"]
            for line in lines:
                content_parts.append(b"(" + _pdf_escape(line) + b") Tj")
                content_parts.append(b"0 -14 Td")
            content_parts.append(b"ET")
            stream = b"\n".join(content_parts)
            objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
            page_id = page_object_ids[index]
            content_id = page_id - 1
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
            )

        payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_id, body in enumerate(objects, start=1):
            offsets.append(len(payload))
            payload.extend(f"{object_id} 0 obj\n".encode("ascii"))
            payload.extend(body)
            payload.extend(b"\nendobj\n")
        xref = len(payload)
        payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        payload.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        payload.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(payload))
        return target.exists() and target.stat().st_size > 0
    except Exception:
        return False


def build_demo_report() -> dict[str, Any]:
    """Create a deterministic five-strategy, 20-run demo study context."""

    strategies = ["No Memory", "Sliding Window", "Summary", "Episodic", "Vector"]
    means = {"No Memory": 0.42, "Sliding Window": 0.56, "Summary": 0.66, "Episodic": 0.74, "Vector": 0.70}
    hypotheses = [{"id": f"H{i + 1}", "statement": f"{name} improves long-term agent score.", "rationale": "Synthetic benchmark prior.", "confidence": 0.2, "status": "Untested"} for i, name in enumerate(strategies)]
    experiments: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    run_index = 0
    offsets = [-0.06, -0.02, 0.02, 0.06]
    for h, strategy in enumerate(strategies):
        eid = f"E{h + 1}"
        experiments.append({"experiment_id": eid, "hypothesis_id": f"H{h + 1}", "name": f"{strategy} benchmark", "status": "Complete", "seed": 20261004 + h})
        for j, offset in enumerate(offsets):
            run_index += 1
            value = round(means[strategy] + offset, 4)
            results.append({"run_id": f"R{run_index}", "experiment_id": eid, "hypothesis_id": f"H{h + 1}", "group": strategy, "score": value, "success": value >= 0.5, "status": "Complete"})
    analysis = ResultAnalyzer().compare_groups(results, metric="score", group_field="group", baseline_group="No Memory")
    # Apply a transparent demo threshold solely to make the report useful.
    for h in hypotheses:
        summary = analysis["groups"].get(strategies[int(h["id"][1:]) - 1], {})
        avg = summary.get("mean")
        h["confidence"] = round(max(0.0, min(1.0, (avg or 0.0))), 3)
        h["status"] = "Supported" if (avg or 0) >= 0.65 else "Inconclusive"
        h["evidence"] = {"mean": avg, "n": summary.get("n")}
    project = {"id": "memory-strategy-demo", "name": "Memory Strategy Study", "research_question": "Which memory strategy is best for a long-term agent?"}
    journal = build_scientist_journal(experiments, results, hypotheses, study_id=project["id"])
    graph = build_hypothesis_graph(project, hypotheses, experiments, results)
    return {"project": project, "hypotheses": hypotheses, "experiments": experiments, "results": results, "analysis": analysis, "journal": journal, "graph": graph}


def generate_demo_report(output_dir: str | Path | None = None) -> ResearchReport:
    """Build the deterministic demo and optionally write report artifacts."""

    context = build_demo_report()
    if output_dir is not None:
        from .visualization import save_charts

        chart_paths = save_charts(context["analysis"], output_dir, prefix="memory_strategy")
        context["charts"] = chart_paths
        save_report(context, Path(output_dir) / "research_report.md", format="markdown")
        save_report(context, Path(output_dir) / "research_report.html", format="html")
        Path(output_dir, "hypothesis_graph.mmd").write_text(hypothesis_graph_to_mermaid(context["graph"]), encoding="utf-8")
        context["journal"].save(Path(output_dir) / "scientist_journal.json")
    return generate_report(context)
