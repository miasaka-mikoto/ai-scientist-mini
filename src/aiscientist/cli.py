"""Command-line interface for AI Scientist Mini.

All commands are offline and bounded. ``demo`` is intentionally the fastest
way to verify the complete scientific loop on a fresh checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]


def _demo(args: argparse.Namespace) -> int:
    script = ROOT / "run_demo.py"
    argv = ["run_demo.py", "--output", str(args.output)]
    if args.db is not None:
        argv += ["--db", str(args.db)]
    argv += ["--first-round-runs", str(args.first_round_runs), "--second-round-runs", str(args.second_round_runs), "--second-round-count", str(args.second_round_count)]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old
    return 0


def _inspect(args: argparse.Namespace) -> int:
    target = args.path
    if target.is_dir():
        candidate = target / "memory_strategy_report.json"
        if not candidate.exists():
            candidate = target / "memory_strategy_snapshot.json"
    else:
        candidate = target
    if not candidate.exists():
        raise SystemExit(f"No study JSON found at {candidate}")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    summary = {
        "source": str(candidate),
        "research_question": data.get("project", {}).get("research_question", data.get("research_question")),
        "hypotheses": len(data.get("hypotheses", [])),
        "experiments": len(data.get("experiments", [])),
        "runs": len(data.get("runs", [])),
        "results": len(data.get("results", [])),
        "audit_events": len(data.get("audit", [])),
        "budget": data.get("budget", data.get("project", {}).get("budget", {})),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _adapter_run(args: argparse.Namespace) -> int:
    """Run one offline provider through the stable JSON adapter contract."""

    from .adapters import LocalJsonAdapter

    text = args.request.read_text(encoding="utf-8") if args.request else sys.stdin.read()
    response = LocalJsonAdapter().execute_json(text)
    print(response)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiscientist-mini", description="Offline, auditable automatic experiment scientist")
    sub = parser.add_subparsers(dest="command")
    demo = sub.add_parser("demo", help="run the bounded Memory Strategy Study")
    demo.add_argument("--output", type=Path, default=ROOT / "demo_output")
    demo.add_argument("--db", type=Path, default=None)
    demo.add_argument("--first-round-runs", type=int, default=4)
    demo.add_argument("--second-round-runs", type=int, default=3)
    demo.add_argument("--second-round-count", type=int, default=2)
    inspect = sub.add_parser("inspect", help="summarize a demo JSON or artifact directory")
    inspect.add_argument("path", type=Path)
    adapter = sub.add_parser("adapter-run", help="run one local provider from a JSON request")
    adapter.add_argument("request", type=Path, nargs="?", help="JSON request file; stdin when omitted")
    sub.add_parser("gui", help="open the Tkinter dashboard")
    args = parser.parse_args(argv)
    command = args.command or "demo"
    if command == "demo":
        return _demo(args)
    if command == "inspect":
        return _inspect(args)
    if command == "adapter-run":
        return _adapter_run(args)
    if command == "gui":
        from .ui import main as ui_main

        return int(ui_main([]) or 0)
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
