#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${1:-}" == "--clean" ]]; then
  rm -rf dist build
fi
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean --onefile --windowed --name AIScientistMini \
  --paths src \
  --hidden-import aiscientist.core \
  --hidden-import aiscientist.core.models \
  --hidden-import aiscientist.core.analysis \
  --hidden-import aiscientist.core.providers \
  --hidden-import aiscientist.core.store \
  --hidden-import aiscientist.core.engine \
  --hidden-import aiscientist.core.demo \
  --hidden-import aiscientist.reporting \
  --hidden-import aiscientist.visualization \
  --exclude-module scipy --exclude-module matplotlib --exclude-module reportlab \
  --exclude-module PySide6 --exclude-module PIL --exclude-module pandas \
  launcher.py
echo "Built executable: dist/AIScientistMini"
