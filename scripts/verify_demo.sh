#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python scripts/smoke_inference.py
python scripts/smoke_colab_inference.py
python -c "import app; print('Streamlit dashboard import check passed.')"
