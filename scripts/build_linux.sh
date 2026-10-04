#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rm -rf build dist
python -m PyInstaller --noconfirm --clean --name AIScientistMini \
  --exclude-module matplotlib --exclude-module numpy --exclude-module pandas \
  --exclude-module PySide6 --paths "$ROOT/src" --onedir "$ROOT/scripts/launcher.py"
echo "Built dist/AIScientistMini/AIScientistMini"
