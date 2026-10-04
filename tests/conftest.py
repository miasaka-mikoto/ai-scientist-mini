"""Pytest helpers for the AI Scientist Mini acceptance suite.

The package uses a ``src`` layout.  ``pyproject.toml`` configures pytest for
normal installs, but adding the source directory here also lets the tests be
run directly from a source checkout (which is how the CI/packaging smoke test
invokes them).
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

