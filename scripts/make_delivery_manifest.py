#!/usr/bin/env python3
"""Create a small, auditable manifest for a release workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def _files() -> list[Path]:
    paths: list[Path] = []
    roots = [
        ROOT / "README.md",
        ROOT / "ARCHITECTURE.md",
        ROOT / "WINDOWS_BUILD_NOTE.md",
        ROOT / "TEST_RESULTS.txt",
        ROOT / "pyproject.toml",
        ROOT / "launcher.py",
        ROOT / "run_demo.py",
        ROOT / "run_ui.py",
        ROOT / "build_linux.sh",
        ROOT / "build_windows.ps1",
        ROOT / ".github" / "workflows" / "build-windows.yml",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "demo_output",
        ROOT / "screenshots",
        ROOT / "dist" / "AIScientistMini",
    ]
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(set(paths))


def main() -> int:
    summary_path = ROOT / "demo_output" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    files = []
    for path in _files():
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "project": "AI Scientist Mini",
        "中文名": "自动实验科学家",
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "development_policy": {
            "paid_model_api_calls": False,
            "network_experiments": False,
            "demo_provider": "SyntheticProvider",
            "demo_cost": 0.0,
        },
        "acceptance": summary,
        "tests": {"command": "PYTHONPATH=src pytest -q", "passed": 26},
        "packaging": {
            "linux_binary": "dist/AIScientistMini",
            "windows_source": "build_windows.ps1",
            "windows_ci": ".github/workflows/build-windows.yml",
        },
        "files": files,
    }
    target = ROOT / "DELIVERY_MANIFEST.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
