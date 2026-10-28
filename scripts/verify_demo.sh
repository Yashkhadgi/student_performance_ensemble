#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

echo "=== 1. RUNNING FULL PYTEST SUITE ==="
"$PYTHON" -m pytest -q

echo "=== 2. RUNNING HISTORICAL MODEL INFERENCE SMOKE TEST ==="
"$PYTHON" scripts/smoke_inference.py

echo "=== 3. RUNNING COLAB EXPORTS SMOKE TEST ==="
"$PYTHON" scripts/smoke_colab_inference.py

echo "=== 4. RUNNING DASHBOARD IMPORT VERIFICATION ==="
"$PYTHON" -c "import app; print('Streamlit dashboard import check passed.')"

echo ""
echo "🎉 ALL SYSTEM CHECKS PASSED SUCCESSFULLY!"
