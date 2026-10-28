# Colab export contract

## Integrated export

The supplied `hack_o_week_exports.zip` is installed in `colab_exports/` and uses a legacy manifest/CSV layout: `manifest.json`, `metrics/optimizer_histories.csv`, `metrics/optimizer_comparison.csv`, model-specific metrics, and `models/numpy_neural_network.joblib`. `src/colab_artifacts.py` validates its UCI dataset declaration, disjoint complete 316/79 split, raw feature schemas, 20 ensemble files, NN artifact shape and exact preprocessing path. It is displayed separately from archived local results.

Copy user-trusted exports into `colab_exports/`. The dashboard never loads an NN joblib automatically.

Required first: `experiment_manifest.json`, containing `dataset_sha256` equal to the official UCI Mathematics hash, `risk_label` exactly `G3 < 10`, integer `seed`, `source`, `trained_at`, exact ordered feature schemas for `early` and `grade_informed`, a `held_out_test` split with zero-based `test_row_indices`, and package versions.

Optional CSVs are displayed only after schema validation:

| File | Required columns |
| --- | --- |
| `nn_history.csv` | scenario, optimizer (`SGD`/`Adam`), epoch, train_loss, val_loss |
| `nn_metrics.csv` | scenario, optimizer, task (`risk`), split (`held_out_test`), metric, value |
| `nn_predictions.csv` | scenario, optimizer, row_index, actual, predicted, risk_probability |
| `ensemble_metrics.csv` | scenario, task, model, split, metric, value |

Also hand off a description of every `nn_models/` object, preprocessing object, input order, class threshold, and a real inference example. Codex will then perform a trusted local load, exact-feature check, probability range check, and one real prediction before enabling NN prediction.
