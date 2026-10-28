#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Student Performance Ensemble Lab (REAL UCI DATA PIPELINE) ==="
if [[ ! -d .venv ]]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Ensuring required dependencies are installed..."
python -m pip install -q -r requirements.txt

if [[ ! -f data/raw/student-mat.csv ]]; then
    echo "Downloading official UCI Student Performance dataset..."
    python -m src.data --download
else
    echo "Verifying existing UCI Student Performance dataset..."
    python -m src.data --verify data/raw/student-mat.csv
fi

echo "Training all 20 model pipelines on verified real data..."
python -m src.train --data data/raw/student-mat.csv

echo "Running bias-variance, regularization, and ensemble experiments..."
python -m src.experiments --data data/raw/student-mat.csv

echo "Launching Streamlit Dashboard..."
python -m streamlit run app.py
