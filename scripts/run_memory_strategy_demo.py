"""Generate the bundled Memory Strategy Study artifacts.

Usage::

    python scripts/run_memory_strategy_demo.py --output artifacts/memory_strategy

The script is intentionally a thin wrapper around the package API.  It never
contacts a model or network service and exits non-zero if the expected
two-round, 20+ run acceptance criteria are not met.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AI Scientist Mini's local Memory Strategy Study")
    parser.add_argument("--output", type=Path, default=Path("artifacts/memory_strategy"))
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)

    # Permit running directly from a source checkout without installation.
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from ai_scientist_mini.reporting import generate_demo_artifacts
    from ai_scientist_mini.demo_study import run_demo_study

    study = run_demo_study(seed=args.seed)
    if len(study.configurations) != 5 or len(study.hypotheses) != 5 or len(study.runs) < 20:
        raise RuntimeError("demo acceptance criteria failed: expected 5 hypotheses/configurations and 20+ runs")
    if {experiment.round for experiment in study.experiments} != {1, 2}:
        raise RuntimeError("demo acceptance criteria failed: expected two experiment-selection rounds")
    paths = generate_demo_artifacts(args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output), "hypotheses": len(study.hypotheses), "configurations": len(study.configurations), "experiments": len(study.experiments), "runs": len(study.runs), "artifacts": paths}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

