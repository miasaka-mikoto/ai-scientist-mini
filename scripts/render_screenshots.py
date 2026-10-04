#!/usr/bin/env python3
"""Render reproducible PNG screenshots from the offline demo artifacts.

The runtime report intentionally emits dependency-light SVG.  This optional
developer utility rasterises the same JSON data for README/release screenshots
without changing the core engine or contacting a service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render(output: Path, screenshots: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    screenshots.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    report = json.loads((output / "memory_strategy_report.json").read_text(encoding="utf-8"))
    groups = analysis.get("groups", {})
    labels = list(groups)
    means = [groups[label].get("mean", 0.0) for label in labels]
    lows = [groups[label].get("ci_lower", groups[label].get("ci_low", value)) for label, value in zip(labels, means)]
    highs = [groups[label].get("ci_upper", groups[label].get("ci_high", value)) for label, value in zip(labels, means)]
    lower = [max(0.0, value - mean) for value, mean in zip(means, lows)]
    upper = [max(0.0, value - mean) for value, mean in zip(highs, means)]
    palette = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.edgecolor": "#9ca3af"})
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.bar(labels, means, yerr=[lower, upper], capsize=5, color=palette[: len(labels)], alpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Observed score")
    ax.set_title("Score by hypothesis", loc="left", fontsize=16, pad=14)
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.text(0.99, 0.02, "bars = mean; whiskers = 95% CI", ha="right", color="#6b7280", fontsize=9)
    fig.tight_layout()
    comparison = screenshots / "memory_strategy_comparison.png"
    fig.savefig(comparison, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    result_rows = report.get("results", [])
    values = {label: [] for label in labels}
    for row in result_rows:
        group = row.get("group", row.get("strategy"))
        metrics = row.get("metrics", {})
        if group in values and isinstance(metrics, dict) and isinstance(metrics.get("score"), (int, float)):
            values[group].append(float(metrics["score"]))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for index, label in enumerate(labels):
        ys = [index + 1] * len(values[label])
        ax.scatter(values[label], ys, s=38, color=palette[index % len(palette)], alpha=0.9)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score")
    ax.set_title("Run distribution (score)", loc="left", fontsize=16, pad=14)
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    distribution = screenshots / "memory_strategy_distribution.png"
    fig.savefig(distribution, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # A compact dashboard screenshot assembled from the exact chart outputs.
    comparison_img = Image.open(comparison).convert("RGB")
    distribution_img = Image.open(distribution).convert("RGB")
    width = 1000
    chart_width = 940
    chart_height = int(chart_width * comparison_img.height / comparison_img.width)
    dist_height = int(chart_width * distribution_img.height / distribution_img.width)
    canvas = Image.new("RGB", (width, 120 + chart_height + 80 + dist_height), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 22), "AI Scientist Mini | Automatic Experiment Scientist", fill="#1f2937", font=_font(25, True))
    draw.text((30, 62), f"Memory Strategy Study | {len(report.get('hypotheses', []))} hypotheses | {len(report.get('experiments', []))} configurations | {len(report.get('runs', []))} runs | cost $0", fill="#4b5563", font=_font(16))
    comp = comparison_img.resize((chart_width, chart_height))
    canvas.paste(comp, (30, 100))
    draw.text((30, 100 + chart_height + 18), "Run distribution", fill="#374151", font=_font(16))
    dist = distribution_img.resize((chart_width, dist_height))
    canvas.paste(dist, (30, 100 + chart_height + 48))
    canvas.save(screenshots / "research_dashboard_preview.png", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument("--screenshots", type=Path, default=Path("screenshots"))
    args = parser.parse_args()
    render(args.output, args.screenshots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
