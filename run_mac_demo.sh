#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Student Performance Ensemble Lab (SYNTHETIC DEMO PIPELINE) ==="
echo "WARNING: Running in SYNTHETIC DEMO MODE for offline unit testing only."

if [[ ! -d .venv ]]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Ensuring required dependencies are installed..."
python -m pip install -q -r requirements.txt

echo "Generating synthetic demonstration data..."
python -m src.data --demo

echo "Training models on synthetic demonstration data..."
python -m src.train --demo --fast

echo "Running experiments on synthetic demo data..."
python -m src.experiments --demo

echo "Launching Streamlit Dashboard in Demo Mode..."
python -m streamlit run app.py
