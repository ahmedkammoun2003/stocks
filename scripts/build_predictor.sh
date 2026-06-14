#!/usr/bin/env bash
# Build stocks-predictor standalone executable + release bundle (Linux).
# For Windows .exe see scripts/BUILD_WINDOWS.md or scripts/build_predictor.bat
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d venv ]]; then
  echo "Create venv and install deps first:"
  echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

exec ./venv/bin/python scripts/build_predictor.py "$@"
