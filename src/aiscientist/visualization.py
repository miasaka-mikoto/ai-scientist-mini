"""Small, deterministic SVG chart helpers for AI Scientist Mini.

Charts are generated without a GUI or a plotting server.  SVG is both portable
and suitable for embedding in Markdown/HTML reports.  If callers need PNG/PDF,
they can rasterise the returned SVG using their preferred tool; the research
engine itself never invokes an external service.
"""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reporting import (
    ResultAnalyzer,
    _as_list,
    _get,
    _jsonable,
    _number,
    _record_rows,
    build_hypothesis_graph,
    hypothesis_graph_to_mermaid,
    render_hypothesis_graph,
)

__all__ = [
    "metric_comparison_svg",
    "distribution_svg",
    "plot_metric_comparison",
    "plot_distribution",
    "render_metric_comparison",
    "render_distribution",
    "save_charts",
    "create_charts",
    "chart_manifest",
    "build_hypothesis_graph",
    "hypothesis_graph_to_mermaid",
    "render_hypothesis_graph",
]


PALETTE = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5"]


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "∞" if number > 0 else "−∞"
    return f"{number:.3f}"


def _summaries(analysis: Any, metric: str = "score", group_field: str = "hypothesis_id") -> dict[str, dict[str, Any]]:
    if not isinstance(analysis, Mapping) and hasattr(analysis, "to_dict"):
        try:
            analysis = analysis.to_dict(include_values=True)
        except TypeError:
            try:
                analysis = analysis.to_dict()
            except Exception:
                pass
    if isinstance(analysis, Mapping) and isinstance(analysis.get("groups"), Mapping):
        return {str(k): dict(v) for k, v in analysis["groups"].items()}
    if isinstance(analysis, Mapping) and "mean" in analysis:
        return {"all": dict(analysis)}
    # Accept raw result records for convenience.
    compared = ResultAnalyzer().compare_groups(analysis, metric=metric, group_field=group_field)
    return {str(k): dict(v) for k, v in compared.get("groups", {}).items()}


def _ci_bounds(summary: Mapping[str, Any]) -> tuple[Any, Any]:
    """Accept both reporting's dict CI and core AnalysisSummary's tuple/list."""

    ci = summary.get("confidence_interval") or {}
    if isinstance(ci, Mapping):
        return ci.get("low", summary.get("ci_low")), ci.get("high", summary.get("ci_high"))
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        return ci[0], ci[1]
    return summary.get("ci_low"), summary.get("ci_high")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:system-ui,-apple-system,Segoe UI,sans-serif;fill:#1f2937}.muted{fill:#6b7280;font-size:12px}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.bar{rx:4}.error{stroke:#111827;stroke-width:2}</style>',
        f'<text x="24" y="28" font-size="18" font-weight="600">{escape(title)}</text>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def metric_comparison_svg(
    analysis: Any,
    *,
    metric: str = "score",
    title: str | None = None,
    width: int = 960,
    height: int = 500,
) -> str:
    """Render group means with optional 95% confidence bars as SVG text."""

    title = title or f"{metric.title()} by hypothesis"
    summaries = _summaries(analysis, metric)
    labels = list(summaries)
    means = [float(summaries[label].get("mean")) for label in labels if summaries[label].get("mean") is not None]
    if not means:
        means = [0.0]
    low_values = [_ci_bounds(summaries[label])[0] for label in labels]
    high_values = [_ci_bounds(summaries[label])[1] for label in labels]
    finite_ci = [float(x) for x in low_values + high_values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    min_value = min([0.0] + means + finite_ci)
    max_value = max([1.0] + means + finite_ci)
    if math.isclose(max_value, min_value):
        max_value = min_value + 1.0
    margin_left, margin_right, margin_top, margin_bottom = 170, 32, 64, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_h

    def y(value: float) -> float:
        return margin_top + (max_value - value) / (max_value - min_value) * plot_h

    lines = _svg_header(width, height, title)
    for tick in range(5):
        value = min_value + (max_value - min_value) * tick / 4
        yy = y(value)
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}"/>')
        lines.append(f'<text class="muted" x="{margin_left - 10}" y="{yy + 4:.1f}" text-anchor="end">{_fmt(value)}</text>')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{baseline_y}" x2="{width - margin_right}" y2="{baseline_y}"/>')
    if labels:
        slot = plot_w / len(labels)
        bar_w = min(90, slot * 0.58)
        for index, label in enumerate(labels):
            summary = summaries[label]
            value = summary.get("mean")
            x = margin_left + slot * index + (slot - bar_w) / 2
            if value is None:
                lines.append(f'<text class="muted" x="{x + bar_w / 2:.1f}" y="{baseline_y - 8:.1f}" text-anchor="middle">n/a</text>')
                continue
            yy = y(float(value))
            top = min(yy, baseline_y)
            bar_h = max(abs(baseline_y - yy), 1)
            color = PALETTE[index % len(PALETTE)]
            lines.append(f'<rect class="bar" x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity=".86"><title>{escape(label)}: {_fmt(value)}</title></rect>')
            low, high = _ci_bounds(summary)
            if isinstance(low, (int, float)) and isinstance(high, (int, float)) and math.isfinite(float(low)) and math.isfinite(float(high)):
                cy1, cy2 = y(float(low)), y(float(high))
                cx = x + bar_w / 2
                lines.append(f'<line class="error" x1="{cx:.1f}" y1="{cy1:.1f}" x2="{cx:.1f}" y2="{cy2:.1f}"/>')
                lines.append(f'<line class="error" x1="{cx - 7:.1f}" y1="{cy1:.1f}" x2="{cx + 7:.1f}" y2="{cy1:.1f}"/>')
                lines.append(f'<line class="error" x1="{cx - 7:.1f}" y1="{cy2:.1f}" x2="{cx + 7:.1f}" y2="{cy2:.1f}"/>')
            short = label if len(label) <= 22 else label[:19] + "..."
            lines.append(f'<text class="muted" transform="translate({x + bar_w / 2:.1f},{baseline_y + 18:.1f}) rotate(25)" text-anchor="start">{escape(short)}</text>')
    lines.append(f'<text class="muted" x="{width - margin_right}" y="{height - 12}" text-anchor="end">bars = mean; whiskers = {int(95)}% CI</text>')
    lines += _svg_footer()
    return "\n".join(lines)


def distribution_svg(
    records_or_analysis: Any,
    *,
    metric: str = "score",
    group_field: str = "hypothesis_id",
    title: str | None = None,
    width: int = 960,
    height: int = 500,
) -> str:
    """Render individual run values as a deterministic dot/strip plot."""

    title = title or f"Run distribution ({metric})"
    if not isinstance(records_or_analysis, Mapping) and hasattr(records_or_analysis, "to_dict"):
        try:
            records_or_analysis = records_or_analysis.to_dict(include_values=True)
        except TypeError:
            try:
                records_or_analysis = records_or_analysis.to_dict()
            except Exception:
                pass
    rows = _record_rows(records_or_analysis, metric) if not (isinstance(records_or_analysis, Mapping) and "groups" in records_or_analysis) else []
    if isinstance(records_or_analysis, Mapping) and "groups" not in records_or_analysis and records_or_analysis.get("values"):
        rows = [{"value": float(value), "group": "all", "record": {group_field: "all"}} for value in records_or_analysis["values"]]
    if not rows and isinstance(records_or_analysis, Mapping):
        # Reconstruct a compact representative distribution from summaries when
        # only analysis JSON is available.
        for group, summary in records_or_analysis.get("groups", {}).items():
            values = summary.get("values") or []
            if not values and summary.get("mean") is not None:
                values = [summary["mean"]] * int(summary.get("n", 1))
            for value in values:
                rows.append({"value": value, "group": group, "record": {group_field: group}})
    groups: dict[str, list[float]] = {}
    for row in rows:
        record = row.get("record")
        group = _get(record, group_field, "group", "hypothesis_id", default=row.get("group", "all"))
        groups.setdefault(str(group), []).append(float(row["value"]))
    labels = list(groups)
    values = [v for vals in groups.values() for v in vals]
    min_value = min([0.0] + values) if values else 0.0
    max_value = max([1.0] + values) if values else 1.0
    if math.isclose(min_value, max_value):
        max_value = min_value + 1.0
    margin_left, margin_right, margin_top, margin_bottom = 170, 32, 64, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_h

    def x(value: float) -> float:
        return margin_left + (value - min_value) / (max_value - min_value) * plot_w

    lines = _svg_header(width, height, title)
    for tick in range(5):
        value = min_value + (max_value - min_value) * tick / 4
        xx = x(value)
        lines.append(f'<line class="grid" x1="{xx:.1f}" y1="{margin_top}" x2="{xx:.1f}" y2="{baseline_y}"/>')
        lines.append(f'<text class="muted" x="{xx:.1f}" y="{baseline_y + 24}" text-anchor="middle">{_fmt(value)}</text>')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{baseline_y}" x2="{width - margin_right}" y2="{baseline_y}"/>')
    if labels:
        row_h = plot_h / max(len(labels), 1)
        for index, label in enumerate(labels):
            yy = margin_top + row_h * (index + 0.5)
            lines.append(f'<text class="muted" x="{margin_left - 10}" y="{yy + 4:.1f}" text-anchor="end">{escape(label[:24])}</text>')
            vals = groups[label]
            # Slight vertical jitter is deterministic and based on index.
            for j, value in enumerate(vals):
                cy = yy + ((j % 5) - 2) * 5
                cx = x(value)
                color = PALETTE[index % len(PALETTE)]
                lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" opacity=".85"><title>{escape(label)}: {_fmt(value)}</title></circle>')
    else:
        lines.append(f'<text class="muted" x="{width / 2}" y="{height / 2}" text-anchor="middle">No numeric runs</text>')
    lines += _svg_footer()
    return "\n".join(lines)


def plot_metric_comparison(analysis: Any, output_path: str | Path | None = None, **kwargs: Any) -> str | Path:
    svg = metric_comparison_svg(analysis, **kwargs)
    if output_path is None:
        return svg
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")
    return target


def plot_distribution(records_or_analysis: Any, output_path: str | Path | None = None, **kwargs: Any) -> str | Path:
    svg = distribution_svg(records_or_analysis, **kwargs)
    if output_path is None:
        return svg
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")
    return target


# Explicit ``render_*`` names are convenient when a caller wants SVG text and
# should not accidentally create a file.  Keep both spellings as stable API.
render_metric_comparison = metric_comparison_svg
render_distribution = distribution_svg


def chart_manifest(paths: Sequence[str | Path], *, analysis: Any = None) -> dict[str, Any]:
    return {
        "charts": [str(Path(path)) for path in paths],
        "analysis": _jsonable(analysis) if analysis is not None else None,
        "format": "svg",
        "generator": "AI Scientist Mini visualization",
    }


def save_charts(
    analysis_or_records: Any,
    output_dir: str | Path,
    *,
    metric: str = "score",
    group_field: str = "hypothesis_id",
    prefix: str = "research",
) -> list[str]:
    """Write comparison/distribution SVGs and a JSON manifest.

    Returns absolute-ish string paths (as supplied by ``output_dir``), making
    them directly usable as report figure links.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(analysis_or_records, Mapping) and hasattr(analysis_or_records, "to_dict"):
        try:
            analysis_or_records = analysis_or_records.to_dict(include_values=True)
        except TypeError:
            try:
                analysis_or_records = analysis_or_records.to_dict()
            except Exception:
                pass
    if not (isinstance(analysis_or_records, Mapping) and ("groups" in analysis_or_records or "mean" in analysis_or_records)):
        analysis = ResultAnalyzer().compare_groups(analysis_or_records, metric=metric, group_field=group_field)
    else:
        analysis = analysis_or_records
    comparison_path = output / f"{prefix}_comparison.svg"
    distribution_path = output / f"{prefix}_distribution.svg"
    comparison_path.write_text(metric_comparison_svg(analysis, metric=metric), encoding="utf-8")
    distribution_path.write_text(distribution_svg(analysis_or_records, metric=metric, group_field=group_field), encoding="utf-8")
    paths = [str(comparison_path), str(distribution_path)]
    (output / f"{prefix}_charts.json").write_text(json.dumps(chart_manifest(paths, analysis=analysis), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return paths


create_charts = save_charts
