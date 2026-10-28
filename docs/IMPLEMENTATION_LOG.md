# Implementation & Technical Engineering Log

**Project:** Student Performance Prediction and Academic Risk Analysis Using Ensemble Learning  
**Date:** 2026-10-14  
**Author:** Senior ML Engineer & Technical Auditor (Antigravity)  

---

## 1. Scope & Objective

The objective of this engineering intervention was to audit the project, migrate from synthetic demo data to verified real UCI data, eliminate silent fallback vulnerabilities, ensure complete coverage of compulsory syllabus topics (Bagging, Boosting, Bias-Variance, Overfitting/Underfitting, Regularization), implement diagnostic evaluation metrics (residuals, Brier scores, calibration diagnostics), expand unit testing, and execute all Jupyter notebooks.

---

## 2. Chronological Record of Source Code Changes

### A. Data Layer (`src/data.py`)
- **Problem:** Dataset provenance relied on fragile filename substring checks (`'synthetic' in path.name.lower()`). Any renamed synthetic file was treated as authentic.
- **Modifications:**
  - Added cryptographic `compute_sha256()` function using chunks to compute reproducible file fingerprints.
  - Defined constants for official UCI release hashes (`UCI_MAT_SHA256 = 'e47f9ee225e1...'`, `UCI_POR_SHA256 = 'a7594a11d777...'`).
  - Implemented 3-tier classification in `source_info()`: `VERIFIED REAL UCI DATA`, `SYNTHETIC DEMO DATA`, `UNVERIFIED CUSTOM DATA`.
  - Added validation in `fetch_uci()` to check SHA-256 after archive extraction.
  - Enhanced `load_data()` to validate expected columns, bounds, missing values, and risk class balance.
  - Added `--verify` command-line option for instant dataset verification.

### B. Model Factory (`src/models.py`)
- **Problem:** Scikit-Learn 1.6 and LightGBM 4.6 emitted `UserWarning: X does not have valid feature names` during inference because `OneHotEncoder` transformed feature matrices into unnamed numpy arrays.
- **Modifications:**
  - Added `transformer.set_output(transform='pandas')` to `make_preprocessor()` inside `ColumnTransformer`.
  - Maintained structured pandas feature names across all transformer pipelines, eliminating inference warnings while preserving feature importance mapping.

### C. Training & Evaluation Pipeline (`src/train.py`)
- **Problem:** Silent fallback on line 169 reverted to `student-demo-SYNTHETIC.csv` if `student-mat.csv` was missing. Lacked error residuals and probability calibration diagnostics.
- **Modifications:**
  - Removed silent fallback. Replaced with explicit mutually exclusive CLI options (`--data` vs `--demo`).
  - Added training timestamp (`trained_at`) and dataset SHA-256 fingerprint to `models/manifest.json`.
  - Added `max_error` to regression metrics and `brier_score` to classification metrics in `evaluate()`.
  - Added `residual = actual - predicted` column to `test_predictions.csv` for residual distribution analysis.
  - Replaced pandas `df.to_csv()` with `shutil.copy2` for creating `data/active_dataset.csv`, ensuring exact byte-for-byte SHA-256 preservation.

### D. Syllabus Experiments Engine (`src/experiments.py`)
- **Problem:** Silent fallback on line 108; bias-variance complexity curves evaluated depths `(1, 2, 3, 4, 6, 10, None)`, skipping compulsory depth 5; boosting regularization only tested two coarse configurations.
- **Modifications:**
  - Removed silent fallback; added explicit `--demo` flag.
  - Included depth 5 explicitly: depths evaluated are now `(1, 2, 3, 5, 8, 10, Unlimited)`.
  - Added `min_samples_split` alongside `max_depth`, `min_samples_leaf`, and `ccp_alpha` in decision tree regularization.
  - Expanded XGBoost and LightGBM regularization to systematically evaluate L1 (`reg_alpha`), L2 (`reg_lambda`), depth limits, and shrinkage.
  - Implemented Bagging vs Single Tree vs Random Forest scaling experiment across estimator counts (10, 25, 50, 75, 120 trees), outputting `results/bagging_estimators.csv`.
  - Upgraded Matplotlib figures with professional formatting and saved 8 high-resolution PNG plots in `results/figures/`.

### E. Streamlit Dashboard (`app.py`)
- **Problem:** Sidebar displayed a generic "UCI / user-supplied" badge; lacked regression residual distribution plots; lacked bagging scaling visualization; lacked numeric correlation heatmap.
- **Modifications:**
  - Added prominent 3-tier status badges (`VERIFIED REAL UCI DATA` / `SYNTHETIC DEMO DATA` / `UNVERIFIED CUSTOM DATA`) with color pills.
  - Added full dataset provenance expander on Overview page showing SHA-256, record count, and official DOI citation.
  - Added residual distribution histogram with zero-error line to Model Comparison page.
  - Added Bagging scaling visualization and expanded Boosting regularization section to Learning Lab.
  - Added Pearson correlation heatmap of academic numeric variables (`studytime`, `failures`, `absences`, `G1`, `G2`, `G3`) to Data Explorer.
  - Added clear ethical disclaimers on non-causal interpretation and secondary school context.

### F. Automated Tests (`tests/test_core.py`)
- **Problem:** Only 13 tests existed, evaluating only small demo datasets with no provenance checks or manifest validation.
- **Modifications:**
  - Expanded test suite to 35 comprehensive unit and integration tests:
    - Real dataset SHA-256 matching official UCI hash
    - Synthetic demo detection
    - Unverified custom dataset classification
    - Schema validation and invalid grade bounds (<0 or >20)
    - Target leakage prevention in both scenarios
    - Regression and classification across all 5 models and both scenarios (20 combinations)
    - Model serialization roundtrip (`joblib.dump` / `joblib.load`)
    - Manifest structure and disk artifact integrity
    - Result CSVs existence and non-emptiness
    - Test prediction residual accuracy ($y - \hat{y}$)

### G. Shell Scripts & Makefile
- **`run_mac_real.sh`:** Updated with dataset verification, step-by-step logging, and zero-fallback guarantee.
- **`run_mac_demo.sh`:** Updated with explicit `--demo` flags and warning banners.
- **`Makefile`:** Added `verify` target, updated `train-demo` and `experiments-demo` targets, and added `clean`.
- **`VERSION.txt`:** Updated from synthetic demo notice to verified real UCI dataset status (v1.1.0).

### H. Jupyter Notebooks
- Executed all 4 notebooks in-place with `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.
- All cells now contain real executed outputs, inline pandas displays, and matplotlib figures.
