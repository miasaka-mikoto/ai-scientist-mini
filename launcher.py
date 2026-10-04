"""Portable desktop launcher for AI Scientist Mini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Scientist Mini · 自动实验科学家")
    parser.add_argument("--demo", action="store_true", help="run the complete offline demo instead of opening the GUI")
    parser.add_argument("--headless", action="store_true", help="run the UI smoke cycle without a display")
    parser.add_argument("--output", type=Path, default=None, help="demo artifact directory")
    parser.add_argument("--db", type=Path, default=None, help="demo SQLite path")
    parser.add_argument("--first-round-runs", type=int, default=4)
    parser.add_argument("--second-round-runs", type=int, default=3)
    parser.add_argument("--second-round-count", type=int, default=2)
    parser.add_argument("--version", action="version", version="ai-scientist-mini 0.1.0")
    args, remainder = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parent
    if args.demo:
        # Import the demo function directly so a PyInstaller one-file build
        # does not depend on a loose ``run_demo.py`` next to its executable.
        # The runpy fallback keeps source-tree launches compatible with any
        # future script-only options.
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from aiscientist.core.demo import run_memory_strategy_study

            output = args.output or Path("demo_output")
            db = args.db or (output / "memory_strategy_study.sqlite3")
            report = run_memory_strategy_study(
                db_path=db,
                output_dir=output,
                first_round_runs=args.first_round_runs,
                second_round_runs=args.second_round_runs,
                second_round_count=args.second_round_count,
            )
            print(json.dumps({
                "project": report.get("research_question"),
                "hypotheses": len(report.get("hypotheses", [])),
                "experiments": len(report.get("experiments", [])),
                "runs": len(report.get("runs", [])),
                "rounds": report.get("rounds", {}),
                "cost": report.get("budget", {}).get("spent_api_cost", 0.0),
                "output": str(output),
            }, ensure_ascii=False, indent=2))
        except ImportError:
            demo_args = ["run_demo.py", "--first-round-runs", str(args.first_round_runs), "--second-round-runs", str(args.second_round_runs), "--second-round-count", str(args.second_round_count)]
            if args.output is not None:
                demo_args += ["--output", str(args.output)]
            if args.db is not None:
                demo_args += ["--db", str(args.db)]
            old = sys.argv
            try:
                sys.argv = demo_args + remainder
                runpy.run_path(str(root / "run_demo.py"), run_name="__main__")
            finally:
                sys.argv = old
        return 0
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from aiscientist.ui import main as ui_main

    return int(ui_main((["--headless"] if args.headless else []) + remainder) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
