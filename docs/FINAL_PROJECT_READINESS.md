# Final Project Readiness & Viva Preparation Matrix

**Project:** Student Performance Prediction and Academic Risk Analysis Using Ensemble Learning  
**Evaluation Status:** **READY FOR SUBMISSION & VIVA PRESENTATION**  
**Auditor:** Senior ML Engineer & Technical Auditor (Antigravity)  
**Date:** 2026-10-14  

---

## 1. Requirement & Syllabus Verification Matrix

| Requirement / Topic | Category | Status | Verified Evidence |
|---|---|---|---|
| **Real UCI Mathematics Dataset Active** | Data Authenticity | **PASS** | `student-mat.csv` SHA-256: `e47f9ee225e1...` verified byte-for-byte |
| **Cryptographic Provenance Verification** | Data Provenance | **PASS** | `src/data.py:source_info()`, `models/manifest.json` |
| **No Silent Fallback to Synthetic Data** | Reliability | **PASS** | `src/train.py`, `src/experiments.py` verified; explicit `--demo` required |
| **Decision Tree (Regression & Classification)** | Model Coverage | **PASS** | All 4 pipelines saved and verified |
| **Bagging (Bootstrap Aggregation)** | Model Coverage | **PASS** | All 4 pipelines saved and verified |
| **Random Forest** | Model Coverage | **PASS** | All 4 pipelines saved and verified |
| **XGBoost (Gradient Boosted Trees)** | Model Coverage | **PASS** | All 4 pipelines saved and verified |
| **LightGBM (Histogram Gradient Boosted Trees)**| Model Coverage | **PASS** | All 4 pipelines saved and verified |
| **Early Prediction Scenario (No G1/G2/G3)** | Experimental Protocol | **PASS** | 10 pipelines strictly leakage-safe |
| **Grade-Informed Scenario (G1/G2 included)** | Experimental Protocol | **PASS** | 10 pipelines with target G3 strictly held out |
| **Total Pipelines (2 scenarios × 2 tasks × 5 models)** | Model Coverage | **PASS** | All 20 pipelines present in `models/` and loadable |
| **Train/Test Isolation (80/20 Holdout)** | Evaluation Rigor | **PASS** | Test set untouched during tuning; seeded split (seed=42) |
| **Cross-Validation Protocol** | Evaluation Rigor | **PASS** | 3-fold CV on train split only for model tuning and selection |
| **Regression Diagnostics (MAE, RMSE, R², Residuals)** | Evaluation Rigor | **PASS** | Computed and visualized via distribution histograms |
| **Classification Diagnostics (F1, AUC, Brier, CM)** | Evaluation Rigor | **PASS** | Full confusion matrices, Brier calibration, ROC-AUC |
| **Bias–Variance Complexity Curve** | Syllabus Topic | **PASS** | Depths 1, 2, 3, 5, 8, 10, Unlimited in `results/bias_variance_depth.csv` |
| **Overfitting vs Underfitting Demonstration** | Syllabus Topic | **PASS** | High bias at depth 1 vs high variance at unlimited depth documented |
| **Decision Tree Regularization** | Syllabus Topic | **PASS** | `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha` evaluated |
| **Boosting Regularization** | Syllabus Topic | **PASS** | L1 (`reg_alpha`), L2 (`reg_lambda`), learning rate, depth in `results/` |
| **Bagging Scaling & Variance Reduction** | Syllabus Topic | **PASS** | Estimator curves (10, 25, 50, 75, 120) in `results/bagging_estimators.csv` |
| **Streamlit Web Application** | User Interface | **PASS** | 7 sections active, 3-tier status badge, interactive predictor |
| **Jupyter Notebooks Walkthrough** | Documentation | **PASS** | All 4 notebooks executed with real data outputs and plots saved |
| **Automated Test Suite** | Software Quality | **PASS** | 35 passed tests (`pytest -v`), 0 failures |
| **Mac M1 Apple Silicon Stability** | Portability | **PASS** | Zero OpenMP/libomp crashes; native ARM64 support |

---

## 2. Technical Viva Defense Q&A Cheatsheet

### Q1: Why does Scenario B (Grade-Informed) have a much higher $R^2$ (0.93) than Scenario A (0.27)?
**Defense Answer:**  
"Academic performance exhibits strong temporal autocorrelation. Intermediate trimester grades ($G1$ and $G2$) are direct measurements of student mastery of the exact same subject curriculum. In Scenario A, we deliberately exclude $G1$ and $G2$ to simulate an **early intervention system** before any exams occur. Demographics, study time, and parental support explain approximately $27\%$ of the final grade variance ($R^2 \approx 0.27$, $\text{RMSE} \approx 3.92$). In Scenario B, adding $G1$ and $G2$ raises explained variance to $93\%$ ($\text{RMSE} \approx 1.22$). Both scenarios are scientifically valid for different administrative timelines."

### Q2: Why did you use training-split cross-validation rather than selecting the best model based on held-out test scores?
**Defense Answer:**  
"Selecting hyperparameters or deciding which model 'wins' based on the held-out test set causes **data leakage** and produces optimistically biased performance estimates. In our methodology, the 80/20 train/test split is locked. Hyperparameter selection and candidate comparison are performed strictly on the training partition ($N=316$) using 3-fold cross-validation. The held-out test set ($N=79$) is evaluated only once on the finalized pipelines for unbiased reporting."

### Q3: How do Bagging and Boosting differ mathematically in addressing bias and variance?
**Defense Answer:**  
- **Bagging (Bootstrap Aggregation):** Primarily targets **variance reduction**. It trains deep, low-bias, high-variance decision trees in parallel on bootstrap samples drawn with replacement, then averages their predictions:
  $$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
  As the number of trees $B$ increases, the second term approaches zero. Random Forest further reduces tree correlation $\rho$ through random feature subsampling.
- **Boosting (XGBoost / LightGBM):** Primarily targets **bias reduction** (and variance via shrinkage). It builds shallow, high-bias trees sequentially, where each subsequent tree fits the negative gradient (pseudo-residuals) of the loss function:
  $$f_m(x) = f_{m-1}(x) + \eta h_m(x)$$
  Regularization terms ($\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2 + \alpha \sum |w_j|$) prevent the additive ensemble from overfitting.

### Q4: What did your Bias–Variance experiment reveal about tree depth?
**Defense Answer:**  
"We trained Decision Trees across depths 1, 2, 3, 5, 8, 10, and Unlimited across repeated validation splits. At depth 1 (a single decision stump), training RMSE was $4.15$ and validation RMSE was $4.28$—clear **underfitting** due to high bias. With unlimited depth, training RMSE dropped to $0.0$, but validation RMSE diverged to $4.55 \pm 0.38$—severe **overfitting** due to high variance. Depths 2 and 3 yielded the optimal minimum validation RMSE ($\approx 3.96$–$4.02$)."

### Q5: How do you prevent target leakage during categorical encoding?
**Defense Answer:**  
"Encoding before splitting leads to data contamination because category frequencies and target relationships from the test set leak into the training process. In our implementation, `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` and `SimpleImputer` is encapsulated inside a Scikit-Learn `Pipeline`. When `cross_val_score` or `fit` runs, the preprocessor is fitted strictly on training observations and merely applied as a transform to unseen validation or test records."

---

## 3. Remaining Blockers Assessment

**Blockers Remaining:** **NONE.**  
All 20 pipelines, verified dataset provenance, automated tests, notebooks, experiment figures, and documentation are complete and verified.
