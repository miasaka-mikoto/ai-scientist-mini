"""Command-line launcher for the AI Scientist Mini dashboard.

Examples
--------
``python -m ai_scientist_mini.ui.app``
    Open the desktop dashboard.

``python -m ai_scientist_mini.ui.app --smoke-test``
    Exercise the complete local synthetic study without opening a window.
    This is useful for CI and for verifying a packaged build on a machine
    without a display.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .dashboard import Dashboard, DashboardController, LocalDemoService, launch


def run_smoke_test() -> dict[str, object]:
    """Run a deterministic UI/service smoke scenario and return its summary."""

    service = LocalDemoService()
    controller = DashboardController(service)
    controller.start_demo()
    # Initial five configurations plus uncertainty-first follow-ups.
    for _ in range(7):
        controller.run_next()
        if controller.snapshot.status in {"Complete", "Stopped", "Budget Exceeded"}:
            break
    snapshot = controller.snapshot
    report = service.generate_report()
    checks = {
        "question_created": bool(snapshot.question),
        "hypotheses_generated": len(snapshot.hypotheses) >= 5,
        "experiments_executed": len(snapshot.experiments) >= 5,
        "evidence_recorded": len(snapshot.evidence) >= 5,
        "budget_zero_cost": float(snapshot.budget.get("estimated_api_cost", 0)) == 0.0,
        "report_generated": report.startswith("# AI Scientist Mini"),
        "graph_available": bool(snapshot.graph.get("nodes")),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "question": snapshot.question,
        "status": snapshot.status,
        "hypotheses": len(snapshot.hypotheses),
        "experiments": len(snapshot.experiments),
        "evidence": len(snapshot.evidence),
        "report_chars": len(report),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Scientist Mini desktop dashboard")
    parser.add_argument("--smoke-test", action="store_true", help="run a no-window deterministic smoke test")
    parser.add_argument("--demo", action="store_true", help="start with the built-in Memory Strategy Study")
    parser.add_argument("--state", type=Path, help="optional local JSON state path for the fallback service")
    parser.add_argument("--json", action="store_true", help="print smoke-test output as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke_test:
        result = run_smoke_test()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("AI Scientist Mini UI smoke test:", "PASS" if result["passed"] else "FAIL")
            for key, value in result["checks"].items():
                print(f"  {'✓' if value else '✗'} {key}")
        return 0 if result["passed"] else 1

    service = LocalDemoService(storage_path=args.state) if args.state else LocalDemoService()
    if args.demo:
        service.start_demo()
    launch(service)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

