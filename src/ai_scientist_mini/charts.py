"""Small, dependency-free chart renderer for AI Scientist Mini reports.

SVG is intentionally the primary output: it renders in a desktop UI and can
be inspected without matplotlib.  A PNG companion is generated when
matplotlib is installed, but the demo never requires it.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .demo_study import STRATEGIES, DemoStudy, summarize_by_strategy


def _svg_document(body: str, *, width: int = 900, height: int = 500, title: str = "Chart") -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <title>{html.escape(title)}</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>text {{ font-family: Inter, Arial, sans-serif; fill: #172033; }} .axis {{ stroke:#61708a; stroke-width:1; }} .grid {{ stroke:#dfe5ef; stroke-width:1; }} .bar {{ fill:#3b82f6; }} .bar2 {{ fill:#14b8a6; }} .label {{ font-size:13px; }} .small {{ font-size:11px; }} .title {{ font-size:20px; font-weight:600; }}</style>
{body}
</svg>\n'''


def _bar_chart(summary: Mapping[str, Mapping[str, Any]], *, value_key: str, title: str, y_label: str, filename: str) -> str:
    width, height = 900, 500
    left, right, top, bottom = 90, 35, 70, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = 1.0 if value_key in {"mean", "success_rate"} else max(1.0, max(float(v.get(value_key, 0.0)) for v in summary.values()))
    body: list[str] = [f'<text x="{width/2}" y="35" text-anchor="middle" class="title">{html.escape(title)}</text>']
    for tick in range(6):
        ratio = tick / 5
        y = top + plot_h * (1 - ratio)
        value = ratio * max_value
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small">{value:.2f}</text>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>')
    slot = plot_w / len(STRATEGIES)
    bar_w = slot * 0.56
    for i, strategy in enumerate(STRATEGIES):
        value = max(0.0, float(summary.get(strategy, {}).get(value_key, 0.0)))
        x = left + slot * i + (slot - bar_w) / 2
        y = top + plot_h * (1 - value / max_value)
        h = top + plot_h - y
        color = "#14b8a6" if strategy in {"Summary", "Episodic", "Vector"} else "#3b82f6"
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        body.append(f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
        body.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="label">{html.escape(strategy)}</text>')
    body.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" class="small">{html.escape(y_label)}</text>')
    Path(filename).write_text(_svg_document("\n".join(body), width=width, height=height, title=title), encoding="utf-8")
    return filename


def _round_chart(study: DemoStudy, *, filename: str) -> str:
    width, height = 900, 520
    left, right, top, bottom = 90, 35, 70, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    rounds = (1, 2)
    body: list[str] = [f'<text x="{width/2}" y="35" text-anchor="middle" class="title">Mean score by strategy and selection round</text>']
    for tick in range(6):
        ratio = tick / 5
        y = top + plot_h * (1 - ratio)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small">{ratio:.2f}</text>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>')
    slot = plot_w / len(STRATEGIES)
    bar_w = slot * 0.25
    colors = {1: "#3b82f6", 2: "#f97316"}
    for i, strategy in enumerate(STRATEGIES):
        for j, round_number in enumerate(rounds):
            runs = [r for r in study.runs if r.strategy == strategy and r.round == round_number and r.score is not None]
            value = sum(float(r.score) for r in runs) / len(runs) if runs else 0.0
            x = left + slot * i + slot * 0.19 + j * (bar_w + 4)
            y = top + plot_h * (1 - value)
            h = top + plot_h - y
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{colors[round_number]}"/>')
            if runs:
                body.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" class="small">{value:.2f}</text>')
        body.append(f'<text x="{left+slot*i+slot/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="label">{html.escape(strategy)}</text>')
    legend_x = width - 170
    for j, round_number in enumerate(rounds):
        x = legend_x + j * 75
        body.append(f'<rect x="{x}" y="52" width="12" height="12" fill="{colors[round_number]}"/><text x="{x+17}" y="63" class="small">Round {round_number}</text>')
    Path(filename).write_text(_svg_document("\n".join(body), width=width, height=height, title="Mean score by round"), encoding="utf-8")
    return filename


def write_charts(study: DemoStudy, output_dir: str | Path) -> dict[str, str]:
    """Write report charts and a chart manifest; return paths by logical name."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = summarize_by_strategy(study.runs)
    paths = {
        "mean_score": _bar_chart(summary, value_key="mean", title="Mean recall score by memory strategy", y_label="Mean score", filename=str(root / "mean_score_by_strategy.svg")),
        "success_rate": _bar_chart(summary, value_key="success_rate", title="Success rate by memory strategy", y_label="Success rate", filename=str(root / "success_rate_by_strategy.svg")),
        "round_comparison": _round_chart(study, filename=str(root / "mean_score_by_round.svg")),
    }
    # Keep the exact source extract beside every figure so a reader can trace
    # a visual back to concrete run IDs without reverse-engineering pixels.
    source_extract = {
        "study_id": study.id,
        "source_run_ids": [run.id for run in study.runs],
        "summaries_by_strategy": summary,
        "rounds": {
            str(round_number): {
                strategy: [run.id for run in study.runs if run.round == round_number and run.strategy == strategy]
                for strategy in STRATEGIES
            }
            for round_number in sorted({run.round for run in study.runs})
        },
    }
    extract_path = root / "chart_data.json"
    extract_path.write_text(json.dumps(source_extract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["source_extract"] = str(extract_path)
    # Optional raster companion for users who need a conventional image file.
    try:
        # Matplotlib may try to write its font cache under a read-only home in
        # CI or a packaged Windows app.  Give it a short-lived writable cache
        # instead; SVG generation above remains the guaranteed path.
        mpl_cache = Path(tempfile.mkdtemp(prefix="ai_scientist_mpl_"))
        previous_mplconfig = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)
        import matplotlib.pyplot as plt  # type: ignore

        names = list(STRATEGIES)
        means = [float(summary[n]["mean"]) for n in names]
        rates = [float(summary[n]["success_rate"]) for n in names]
        for key, values, title, ylabel, filename in (
            ("mean_score_png", means, "Mean recall score by memory strategy", "Mean score", "mean_score_by_strategy.png"),
            ("success_rate_png", rates, "Success rate by memory strategy", "Success rate", "success_rate_by_strategy.png"),
        ):
            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
            ax.bar(names, values, color=["#3b82f6", "#3b82f6", "#14b8a6", "#14b8a6", "#14b8a6"])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = root / filename
            fig.savefig(path)
            plt.close(fig)
            paths[key] = str(path)
    except Exception:
        # SVGs are the guaranteed artifact; a missing optional dependency is
        # recorded rather than making the scientific run fail.
        pass
    finally:
        try:
            if "mpl_cache" in locals():
                shutil.rmtree(mpl_cache, ignore_errors=True)
            if "previous_mplconfig" in locals():
                if previous_mplconfig is None:
                    os.environ.pop("MPLCONFIGDIR", None)
                else:
                    os.environ["MPLCONFIGDIR"] = previous_mplconfig
        except Exception:
            pass
    (root / "chart_manifest.json").write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    return paths


def write_dashboard_snapshot(study: DemoStudy, output_path: str | Path) -> str:
    """Render a compact, shareable dashboard screenshot for the demo.

    This is a report artifact rather than a second UI implementation: it
    captures the current question, evidence counts, budget and strategy means
    in one image so a reviewer can inspect the delivered state without
    launching Tkinter.  If matplotlib is unavailable, a valid SVG fallback is
    written next to the requested path.
    """

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_by_strategy(study.runs)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig = plt.figure(figsize=(14, 8), dpi=120)
        fig.patch.set_facecolor("#f8fafc")
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.90])
        ax.set_axis_off()
        ax.text(0.01, 0.96, "AI Scientist Mini", fontsize=24, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        ax.text(0.01, 0.915, "Automatic Experimental Scientist  ·  Memory Strategy Study", fontsize=12, color="#475569", transform=ax.transAxes)
        cards = [
            ("QUESTION", "Which memory strategy is most suitable for a long-term agent?"),
            ("HYPOTHESES", f"{len(study.hypotheses)} active candidates"),
            ("RUNS", f"{len(study.runs)} completed · 2 rounds"),
            ("COST", "$0.00 · local synthetic"),
        ]
        for i, (label, value) in enumerate(cards):
            x = 0.01 + i * 0.245
            rect = plt.Rectangle((x, 0.79), 0.225, 0.085, transform=ax.transAxes, facecolor="white", edgecolor="#dbe4ef", linewidth=1, zorder=0)
            ax.add_patch(rect)
            ax.text(x + 0.012, 0.848, label, fontsize=8, color="#64748b", transform=ax.transAxes)
            ax.text(x + 0.012, 0.812, value[:42], fontsize=10 if i == 0 else 12, color="#0f172a", transform=ax.transAxes)
        # Main chart.
        chart_ax = fig.add_axes([0.07, 0.16, 0.52, 0.52])
        names = list(STRATEGIES)
        values = [float(summary[name]["mean"]) for name in names]
        colors = ["#60a5fa", "#60a5fa", "#2dd4bf", "#2dd4bf", "#2dd4bf"]
        chart_ax.bar(names, values, color=colors)
        chart_ax.set_ylim(0, 1)
        chart_ax.set_ylabel("Mean score")
        chart_ax.set_title("Evidence by memory strategy", loc="left", fontweight="bold")
        chart_ax.grid(axis="y", alpha=0.22)
        chart_ax.set_axisbelow(True)
        chart_ax.tick_params(axis="x", labelrotation=18)
        for idx, value in enumerate(values):
            chart_ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
        # Decision panel.
        panel = fig.add_axes([0.66, 0.16, 0.30, 0.52])
        panel.set_axis_off()
        best = max(names, key=lambda n: (float(summary[n]["mean"]), n))
        # Keep generous vertical spacing so the compact image remains legible
        # with different matplotlib font backends and DPI settings.
        panel.text(0.0, 0.95, "Decision summary", fontsize=14, fontweight="bold", color="#0f172a")
        panel.text(0.0, 0.84, "Current best", fontsize=9, color="#64748b")
        panel.text(0.0, 0.77, best, fontsize=18, fontweight="bold", color="#0f766e")
        panel.text(0.0, 0.64, f"Mean score  {summary[best]['mean']:.3f}", fontsize=11, color="#334155")
        panel.text(0.0, 0.56, f"Success rate  {summary[best]['success_rate']:.1%}", fontsize=11, color="#334155")
        panel.text(0.0, 0.42, "Next action", fontsize=9, color="#64748b")
        panel.text(0.0, 0.32, "Repeat best +\nuncertain arm", fontsize=14, fontweight="bold", color="#1d4ed8")
        panel.text(0.0, 0.09, "Transparent rule updates;\nno private reasoning stored.", fontsize=10, color="#475569", linespacing=1.5)
        fig.savefig(target, facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(target)
    except Exception:
        fallback = target.with_suffix(".svg")
        body = (
            '<text x="40" y="60" font-size="28" font-family="Arial">AI Scientist Mini</text>'
            '<text x="40" y="100" font-size="16" font-family="Arial">Memory Strategy Study · local synthetic · cost $0.00</text>'
        )
        fallback.write_text(_svg_document(body, width=1000, height=600, title="AI Scientist Mini dashboard"), encoding="utf-8")
        return str(fallback)
