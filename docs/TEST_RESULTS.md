# Automated Test Suite & Verification Results

**Project:** Student Performance Prediction and Academic Risk Analysis Using Ensemble Learning  
**Execution Environment:** macOS Darwin 25.1.0 (Apple Silicon M1), Python 3.9.6, Pytest 8.4.2  
**Test Date:** 2026-10-14  

---

## 1. Test Execution Summary

```
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/yashkhadgi/Downloads/student_performance_ensemble_project
collected 35 items

tests/test_core.py::test_real_data_provenance_and_hash PASSED            [  2%]
tests/test_core.py::test_synthetic_data_detection PASSED                 [  5%]
tests/test_core.py::test_unverified_custom_data_detection PASSED         [  8%]
tests/test_core.py::test_schema_validation_missing_columns PASSED        [ 11%]
tests/test_core.py::test_schema_validation_invalid_grade_bounds PASSED   [ 14%]
tests/test_core.py::test_nonexistent_dataset_raises_error PASSED         [ 17%]
tests/test_core.py::test_no_target_leakage PASSED                        [ 20%]
tests/test_core.py::test_grade_bounds_and_risk_balance PASSED            [ 22%]
tests/test_core.py::test_all_five_regressors[early-Decision Tree] PASSED [ 25%]
tests/test_core.py::test_all_five_regressors[early-Bagging] PASSED       [ 28%]
tests/test_core.py::test_all_five_regressors[early-Random Forest] PASSED [ 31%]
tests/test_core.py::test_all_five_regressors[early-XGBoost] PASSED       [ 34%]
tests/test_core.py::test_all_five_regressors[early-LightGBM] PASSED      [ 37%]
tests/test_core.py::test_all_five_regressors[grade_informed-Decision Tree] PASSED [ 40%]
tests/test_core.py::test_all_five_regressors[grade_informed-Bagging] PASSED [ 42%]
tests/test_core.py::test_all_five_regressors[grade_informed-Random Forest] PASSED [ 45%]
tests/test_core.py::test_all_five_regressors[grade_informed-XGBoost] PASSED [ 48%]
tests/test_core.py::test_all_five_regressors[grade_informed-LightGBM] PASSED [ 51%]
tests/test_core.py::test_all_five_classifiers[early-Decision Tree] PASSED [ 54%]
tests/test_core.py::test_all_five_classifiers[early-Bagging] PASSED      [ 57%]
tests/test_core.py::test_all_five_classifiers[early-Random Forest] PASSED [ 60%]
tests/test_core.py::test_all_five_classifiers[early-XGBoost] PASSED      [ 62%]
tests/test_core.py::test_all_five_classifiers[early-LightGBM] PASSED     [ 65%]
tests/test_core.py::test_all_five_classifiers[grade_informed-Decision Tree] PASSED [ 68%]
tests/test_core.py::test_all_five_classifiers[grade_informed-Bagging] PASSED [ 71%]
tests/test_core.py::test_all_five_classifiers[grade_informed-Random Forest] PASSED [ 74%]
tests/test_core.py::test_all_five_classifiers[grade_informed-XGBoost] PASSED [ 77%]
tests/test_core.py::test_all_five_classifiers[grade_informed-LightGBM] PASSED [ 80%]
tests/test_core.py::test_unknown_categorical_handled PASSED              [ 82%]
tests/test_core.py::test_model_serialization_and_roundtrip PASSED        [ 85%]
tests/test_core.py::test_evaluation_metrics_regression PASSED            [ 88%]
tests/test_core.py::test_evaluation_metrics_classification PASSED        [ 91%]
tests/test_core.py::test_manifest_structure_and_coverage PASSED          [ 94%]
tests/test_core.py::test_results_artifacts_exist_and_non_empty PASSED    [ 97%]
tests/test_core.py::test_test_predictions_residuals PASSED               [100%]

======================= 35 passed in 2.03s ========================
```

---

## 2. Command-by-Command Verification Log

### Command 1: Pytest Suite
- **Command:** `.venv/bin/python -m pytest -v`
- **Result:** Exit code 0, 35 passed, 0 failed.

### Command 2: Dataset Cryptographic Verification
- **Command:** `.venv/bin/python -m src.data --verify data/raw/student-mat.csv`
- **Result:** Exit code 0.
- **Output:**
  ```
  Dataset: data/raw/student-mat.csv
  Status: VERIFIED REAL UCI DATA
  SHA-256: e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80
  Description: Official UCI Student Performance Mathematics dataset, verified by exact SHA-256 cryptographic match.
  ```

### Command 3: Full 20-Model Training Pipeline
- **Command:** `.venv/bin/python -m src.train --data data/raw/student-mat.csv`
- **Result:** Exit code 0. All 20 pipelines fitted, cross-validated on training split only, evaluated on held-out test split, and saved to `models/` with `manifest.json`.
- **Warning Count:** 0 (LightGBM pandas output integration verified).

### Command 4: Syllabus Experiments Pipeline
- **Command:** `.venv/bin/python -m src.experiments --data data/raw/student-mat.csv`
- **Result:** Exit code 0.
- **Generated CSV Files:**
  - `results/bias_variance_depth.csv` (112 rows across 4 repeats and 7 depths)
  - `results/tree_regularization.csv` (48 rows across 6 regularization variants)
  - `results/boosting_regularization.csv` (16 rows evaluating XGBoost and LightGBM penalties)
  - `results/bagging_estimators.csv` (30 rows evaluating estimator scaling)
- **Generated Figures:** 8 high-resolution PNG plots in `results/figures/`.

### Command 5: In-Place Notebook Execution
- **Command:** `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`
- **Result:** Exit code 0 across all 4 notebooks:
  - `01_data_exploration.ipynb` (Executed, 41.8 KB)
  - `02_bias_variance.ipynb` (Executed, 188.2 KB)
  - `03_bagging.ipynb` (Executed, 31.7 KB)
  - `04_boosting_comparison.ipynb` (Executed, 156.4 KB)

### Command 6: Streamlit Application Server Health
- **Command:** `curl -s http://localhost:8501/healthz`
- **Result:** Returned HTTP 200 `ok`. Streamlit process running cleanly on port 8501.
