# Codex handoff

## Completed locally

- Historical UCI integrity and saved-model inference checks: 20 of 20 pipelines load and predict with the pinned scikit-learn 1.6.1 environment.
- Dashboard pages for syllabus evidence and a Colab-ready Neural Network Lab.
- Read-only Colab artifact contract: validates provenance, raw feature order, risk label, declared held-out rows, CSV schemas, metrics and NN histories. It does not train, manufacture evidence, or unpickle arbitrary exports.

## Integrated Colab export

- `hack_o_week_exports.zip` is installed under `colab_exports/`.
- The manifest declares verified UCI Mathematics context, 395 rows, a complete/disjoint 316/79 seed-42 split, 20 ensemble pipelines and a NumPy NN.
- Local checks load all 21 artifacts and run inference. The NN uses the exported early-scenario 30 raw columns, fitted preprocessor, 16 ReLU hidden units, sigmoid output, and 0.5 threshold.
- Colab's XGBoost is 3.4.1 while local XGBoost is 2.1.4; every saved pipeline nevertheless passed a real local inference smoke test. Preserve the version note for reproducibility.

## After delivery

The final app validates the import, runs real NN inference, and keeps comparisons limited to matching scenario, target, split and metrics.

## Start command

`source .venv/bin/activate && streamlit run app.py`
