# Project Discovery & Initial Feature-by-Feature Audit

**Project:** Student Performance Prediction and Academic Risk Analysis Using Ensemble Learning  
**Environment:** macOS (Apple Silicon M1, Darwin 25.1.0), Python 3.9.6, Streamlit 1.50.0, Scikit-Learn 1.6.1, XGBoost 2.1.4, LightGBM 4.6.0  
**Audit Date:** 2026-10-14  
**Auditor:** Senior ML Engineer & Technical Auditor (Antigravity)

---

## 1. Executive Summary

This audit assesses the codebase, dataset authenticity, model training pipelines, cross-validation discipline, evaluation metrics, Streamlit user interface, and automated testing suite.

### Key Initial Findings:
1. **Dataset Authenticity (PASS with provenance gaps):**
   `data/raw/student-mat.csv` (395 rows, 33 columns) matches the official UCI Student Performance dataset SHA-256 checksum (`e47f9ee225e1ee6e...`) byte-for-byte. However, the existing code identifies data as synthetic using only simple filename substring matching (`'synthetic' in path.name.lower()`), lacks cryptographic fingerprinting, and lacks explicit categorization (`VERIFIED REAL UCI DATA` vs `SYNTHETIC DEMO DATA` vs `UNVERIFIED CUSTOM DATA`).
2. **Silent Fallback Vulnerability (FAIL):**
   `src/train.py` (line 169) and `src/experiments.py` (line 108) silently fall back to `student-demo-SYNTHETIC.csv` if `student-mat.csv` is missing. This violates academic submission rules requiring explicit user opt-in for synthetic data.
3. **Model Coverage (PASS):**
   All 5 required model families (Decision Tree, Bagging, Random Forest, XGBoost, LightGBM) are implemented for both regression (`grade`, target $G3 \in [0, 20]$) and binary classification (`risk`, target $G3 < 10$) across both Scenario A (`early`, no G1/G2/G3 in features) and Scenario B (`grade_informed`, G1 and G2 included, G3 excluded). Total: 20 pipelines.
4. **Leakage & Preprocessing (PASS):**
   Categorical encoding and median imputation occur within Scikit-Learn pipelines (`ColumnTransformer`), fitted strictly on training folds and evaluated on held-out test splits.
5. **Experiments & Syllabus Alignment (PARTIAL):**
   - Bias-variance tradeoff experiments in `src/experiments.py` evaluate tree depths `(1, 2, 3, 4, 6, 10, None)`, skipping the syllabus-mandated depth 5.
   - Regularization exploration for boosting only tested 2 coarse configurations ("Less constrained" vs "Regularized") without systematically isolating L1 (`reg_alpha`), L2 (`reg_lambda`), and depth/leaves.
6. **Streamlit UI & Transparency (PARTIAL):**
   The dashboard runs, but only displays a generic "UCI / user-supplied" badge, lacks residual distribution plots for regression models, does not display training provenance details (SHA-256 fingerprint, training timestamp), and does not test or display probability calibration diagnostics.
7. **Jupyter Notebooks (PARTIAL):**
   All 4 notebooks exists in `notebooks/`, but all execution counts and outputs are null (unexecuted).
8. **Automated Testing Suite (PARTIAL):**
   Only 13 tests exist in `tests/test_core.py`, testing basic synthetic pipelines. There are no tests for dataset provenance, SHA-256 verification, silent fallback prevention, residual computation, or manifest schema invariants.

---

## 2. Feature-by-Feature Audit Matrix

| ID | Component / Feature | Source File | Status | Severity | Observed Problem / Limitation | Required Correction |
|---|---|---|---|---|---|---|
| F-01 | Dataset Ingestion & Parsing | `src/data.py` | PASS | Low | Correctly handles semicolon-separated UCI CSVs, checks target columns and bounds. | Ensure robust type parsing and validation logging. |
| F-02 | Dataset Provenance & Fingerprinting | `src/data.py` | PARTIAL | High | Provenance logic relies on filename substring matching (`'synthetic' in path.name.lower()`). Any renamed file would be misclassified. | Implement SHA-256 hash verification against official UCI records. Classify datasets into VERIFIED REAL UCI DATA, SYNTHETIC DEMO DATA, or UNVERIFIED CUSTOM DATA. |
| F-03 | Silent Synthetic Fallback Prevention | `src/train.py`, `src/experiments.py` | FAIL | Critical | Both scripts silently fall back to `student-demo-SYNTHETIC.csv` if `student-mat.csv` is missing without requiring explicit flags. | Remove silent fallback. Require explicit `--demo` flag for synthetic execution. Raise informative error if real dataset is missing. |
| F-04 | Dual Scenario Feature Splitting | `src/data.py` | PASS | Low | `early` excludes `['G1', 'G2', 'G3']`; `grade_informed` excludes `['G3']`. No target leakage observed. | Maintain current leakage-free schema and add regression tests. |
| F-05 | 5 Model Families Implementation | `src/models.py` | PASS | Low | Decision Tree, Bagging, Random Forest, XGBoost, and LightGBM are constructed properly. | Maintain models, ensure parameter compatibility and reproducible random states. |
| F-06 | Apple Silicon & OpenMP Stability | `src/models.py` | PASS | Low | XGBoost 2.1.4 and LightGBM 4.6.0 import and execute on Apple Silicon M1 without `libomp` crashes. | Verify multi-threading flags (`n_jobs=1` for single-thread safety) and add explicit environment checks. |
| F-07 | 20 Model Pipelines (2 scenarios × 2 tasks × 5 models) | `src/train.py` | PASS | Low | Generates all 20 `.joblib` files and records them in `models/manifest.json`. | Preserve pipeline generator and add manifest integrity tests. |
| F-08 | Cross-Validation Protocol | `src/train.py` | PASS | Medium | Uses training-split-only StratifiedKFold (classification) and KFold (regression). Held-out test set is never touched during tuning. | Clarify cross-validation standard deviation in output tables and UI. |
| F-09 | Regression Metrics & Diagnostics | `src/train.py`, `app.py` | PARTIAL | Medium | MAE, RMSE, and R² are computed, but residual distributions (error histograms) are missing from the dashboard. | Add residual calculation and histogram/scatter visualization in the Model Comparison section. |
| F-10 | Classification Metrics & Diagnostics | `src/train.py`, `app.py` | PASS | Low | Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix are properly generated. | Add calibration warning/diagnostics in the UI to prevent claiming uncalibrated probabilities as ground truth. |
| F-11 | Bias-Variance Tradeoff Experiment | `src/experiments.py` | PARTIAL | Medium | Depths evaluated were `(1, 2, 3, 4, 6, 10, None)`, skipping compulsory depth 5. Statistical proxy nature is acknowledged but charts could be clearer. | Add depth 5 explicitly. Ensure repeated split metrics capture both mean and variance/std dev. |
| F-12 | Decision Tree Regularization | `src/experiments.py` | PASS | Low | Evaluates unrestricted, max_depth, min_samples_leaf, and ccp_alpha pruning. | Add min_samples_split variant to fully satisfy syllabus topics. |
| F-13 | Boosting Regularization (L1, L2, Depth, Leaves) | `src/experiments.py` | PARTIAL | Medium | Only compares 2 presets ("Less constrained" vs "Regularized"). L1 (`reg_alpha`) and L2 (`reg_lambda`) effects are not isolated. | Expand boosting experiments to isolate L1/L2 penalties, learning rates, and tree capacity. |
| F-14 | Interactive Predictor (Predict Student) | `app.py` | PASS | Low | Interactive widgets support both early and grade-informed predictions, with sensible defaults and range bounds. | Add dataset provenance banner directly inside the Predictor view. |
| F-15 | Dataset Transparency on Dashboard | `app.py` | PARTIAL | High | Shows basic `Dataset: UCI / user-supplied` text in sidebar. Does not show explicit 3-tier status badge or SHA-256 fingerprint. | Add prominent 3-tier banner (VERIFIED REAL UCI DATA / SYNTHETIC DEMO DATA / UNVERIFIED CUSTOM DATA) with record count, hash, and training timestamp. |
| F-16 | Data Explorer Page | `app.py` | PASS | Low | Plots G3 distribution, class balance, statistical summary, and preview table. | Add correlation heatmap between earlier grades, absences, study time, and final grade. |
| F-17 | Jupyter Notebooks Execution | `notebooks/*.ipynb` | PARTIAL | Medium | Notebooks 01-04 exist and contain valid code, but outputs are unexecuted. | Execute all notebooks with real UCI data and store cell outputs and visualizations. |
| F-18 | Automated Test Suite | `tests/test_core.py` | PARTIAL | High | Only 13 tests, running only on demo fixture; no tests for provenance, real dataset verification, CLI fallbacks, or manifest invariants. | Expand test suite to comprehensive coverage (>25 tests) verifying real data, provenance, all 20 models, and bounds. |
| F-19 | Mac Execution Scripts | `run_mac_real.sh`, `run_mac_demo.sh` | PASS | Low | Simple bash scripts exist. | Update `run_mac_real.sh` to prevent fallback and ensure fail-safe execution. |
| F-20 | Project Documentation | `docs/` | PARTIAL | Low | Existing docs have project report guides and viva QA, but lack the complete final audit and readiness documentation. | Produce all required Phase 12 reports and update README. |

---

## 3. Next Steps Action Plan

1. **Phase 2:** Upgrade `src/data.py` with cryptographic SHA-256 provenance checking, explicit 3-tier dataset classification, and eliminate silent fallback in `src/train.py` and `src/experiments.py`.
2. **Phase 3 & 4:** Enhance evaluation diagnostics (residual distribution, calibration analysis) and verify all 20 model pipelines.
3. **Phase 5, 6 & 7:** Update `src/experiments.py` to evaluate depth 5, expand L1/L2 boosting regularization, and re-run all experiments against verified real data.
4. **Phase 8:** Upgrade `app.py` with clear 3-tier status indicators, residual plots, correlation heatmaps, and dataset provenance details.
5. **Phase 9:** Execute and update all 4 Jupyter notebooks.
6. **Phase 10:** Expand automated tests in `tests/` and verify complete test suite execution.
7. **Phase 11 & 12:** Produce all required final audit and handover documents.
