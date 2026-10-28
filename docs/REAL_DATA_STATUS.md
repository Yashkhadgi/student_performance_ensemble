# Real vs Synthetic Dataset Status & Provenance Report

**Project:** Student Performance Prediction and Academic Risk Analysis Using Ensemble Learning  
**Verification Date:** 2026-10-14  
**Auditor:** Senior ML Engineer & Technical Auditor (Antigravity)  

---

## 1. Definitive Dataset Provenance Statement

| Attribute | Verified Value |
|---|---|
| **Active Dataset Name** | `student-mat.csv` |
| **Active Dataset Filepath** | `data/raw/student-mat.csv` (mirrored to `data/active_dataset.csv`) |
| **Dataset Classification Tier** | **`VERIFIED REAL UCI DATA`** |
| **Dataset Authenticity** | **CRYPTOGRAPHICALLY VERIFIED** against official UC Irvine Machine Learning Repository release |
| **Cryptographic Hash (SHA-256)** | `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80` |
| **Total Observations (Rows)** | **395** |
| **Feature Dimensions (Columns)** | **33** (30 features + G1, G2, G3) |
| **Target Distribution (G3 Grade)** | Range: $[0, 20]$, Mean: $10.42$, Median: $11.0$, Std: $4.58$ |
| **Academic Risk Prevalence ($G3 < 10$)** | **130 students (32.91%) At Risk**, **265 students (67.09%) Not At Risk** |
| **Missing Values / Imputations** | **0 missing values** across all columns |
| **Duplicate Complete Records** | **0 duplicates** |

---

## 2. Verification Against Official UCI Records

The dataset `data/raw/student-mat.csv` was downloaded directly from the official UC Irvine Machine Learning Repository archive:
- **Archive URL:** `https://archive.ics.uci.edu/static/public/320/student+performance.zip`
- **Landing Page:** `https://archive.ics.uci.edu/dataset/320/student+performance`
- **Official Citation:** Cortez, P. (2008). *Student Performance.* UCI Machine Learning Repository. DOI: `10.24432/C5TG7T` (CC BY 4.0).

### Checksum Verification Matrix:
- **Official UCI Archive `student-mat.csv` SHA-256:**  
  `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`
- **Local Active File `data/raw/student-mat.csv` SHA-256:**  
  `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`
- **Match Status:** **EXACT MATCH (100% BYTE-FOR-BYTE IDENTICAL)**

---

## 3. Deployed Model Artifact & Experiment Provenance

| Component | Status | Details |
|---|---|---|
| **Active Models Trained On** | `data/raw/student-mat.csv` | All 20 `.joblib` pipelines in `models/` were trained directly on the verified UCI Mathematics dataset. |
| **Model Manifest** | `models/manifest.json` | Records `source.status = 'VERIFIED REAL UCI DATA'`, SHA-256 fingerprint, and active training timestamp. |
| **Synthetic Artifacts Status** | **NONE IN USE** | Synthetic demo files (`data/demo/student-demo-SYNTHETIC.csv`) remain isolated in `data/demo/` for offline unit testing only. |
| **Experiment Metrics Alignment** | **100% MATCH** | All CSV artifacts in `results/` (`model_metrics.csv`, `cv_candidates.csv`, `test_predictions.csv`, `feature_importance.csv`, `bias_variance_depth.csv`, `tree_regularization.csv`, `boosting_regularization.csv`, `bagging_estimators.csv`) were generated from the active real dataset. |
| **Streamlit UI Provenance** | **UPDATED** | Dashboard displays the green `VERIFIED REAL UCI DATA` badge, showing 395 rows, SHA-256 short hash, and exact class balance. |

---

## 4. Elimination of Silent Fallbacks

Previously, the codebase contained heuristic filename matching (`'synthetic' in path.name.lower()`) and silently fell back to synthetic demo data when `student-mat.csv` was missing.

**Permanent Architectural Fixes Applied:**
1. **Cryptographic Provenance:** Replaced filename string inspection with SHA-256 hash calculation and verification against known UCI and synthetic hashes.
2. **Strict CLI Contract:** `src/train.py` and `src/experiments.py` now reject non-existent datasets with informative error messages. Running synthetic demo mode requires an explicit `--demo` flag.
3. **Reproducible Active Copy:** `train_all()` uses `shutil.copy2` to mirror `data/raw/student-mat.csv` directly into `data/active_dataset.csv`, preserving file attributes and guaranteeing identical checksums.
