"""Run the complete offline Memory Strategy Study.

This is the canonical source-checkout entry point. It uses the same
``ScientificEngine`` as the desktop dashboard and never imports an external
LLM SDK or reads an API key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aiscientist.core.demo import run_memory_strategy_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Scientist Mini offline demo")
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output", help="artifact directory")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path (default: output/study.sqlite3)")
    parser.add_argument("--first-round-runs", type=int, default=4)
    parser.add_argument("--second-round-runs", type=int, default=3)
    parser.add_argument("--second-round-count", type=int, default=2)
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    db = args.db or (output / "memory_strategy_study.sqlite3")
    report = run_memory_strategy_study(
        db_path=db,
        output_dir=output,
        first_round_runs=max(1, args.first_round_runs),
        second_round_runs=max(1, args.second_round_runs),
        second_round_count=max(1, args.second_round_count),
    )
    summary = {
        "project": report.get("project", {}).get("research_question"),
        "hypotheses": len(report.get("hypotheses", [])),
        "experiments": len(report.get("experiments", [])),
        "runs": len(report.get("runs", [])),
        "rounds": report.get("rounds"),
        "cost": report.get("budget", {}).get("spent_api_cost", 0.0),
        "database": str(db),
        "artifacts": report.get("artifacts", {}),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
