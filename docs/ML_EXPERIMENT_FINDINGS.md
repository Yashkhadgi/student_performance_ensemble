# Machine Learning Experiment Findings & Scientific Analysis

**Project:** Student Performance Prediction and Academic Risk Analysis Using Ensemble Learning  
**Dataset:** Official UCI Student Performance (`student-mat.csv`, $N=395$, Verified SHA-256)  
**Evaluation Protocol:** 80/20 Stratified Holdout Test ($N_{\text{train}}=316$, $N_{\text{test}}=79$), 3-fold Cross-Validation on Training Split Only  
**Random Seed:** 42  

---

## 1. Regression Analysis: Predicting Final Grade ($G3 \in [0, 20]$)

### Performance Matrix:

| Scenario | Model Family | Training CV RMSE (Score) | Training CV Std | Train RMSE | Held-Out Test RMSE | Held-Out Test MAE | Held-Out Test $R^2$ | Max Error |
|---|---|---|---|---|---|---|---|---|
| **Early** | **Decision Tree** | 4.073 | 0.317 | 3.946 | 3.970 | 2.924 | 0.255 | 12.28 |
| **Early** | **Bagging** | **3.902** | 0.336 | 2.333 | 3.956 | 2.900 | 0.261 | 13.18 |
| **Early** | **Random Forest** | **3.895** | 0.310 | 2.104 | 3.972 | 2.915 | 0.255 | 13.67 |
| **Early** | **XGBoost** | 4.039 | 0.362 | 3.364 | 3.929 | 3.063 | 0.271 | 12.17 |
| **Early** | **LightGBM** | 4.024 | 0.476 | 3.043 | **3.919** | 3.092 | **0.274** | 11.80 |
| **Grade-Informed** | **Decision Tree** | 1.981 | 0.153 | 1.160 | 1.218 | 0.777 | 0.930 | 6.38 |
| **Grade-Informed** | **Bagging** | **1.675** | 0.240 | 0.812 | 1.238 | 0.794 | 0.928 | 5.80 |
| **Grade-Informed** | **Random Forest** | **1.668** | 0.231 | 0.798 | 1.253 | 0.812 | 0.926 | 5.64 |
| **Grade-Informed** | **XGBoost** | 1.721 | 0.227 | 1.498 | 1.377 | 0.913 | 0.910 | 6.03 |
| **Grade-Informed** | **LightGBM** | 1.685 | 0.174 | 0.872 | **1.216** | 0.818 | **0.930** | 4.68 |

### Key Regression Takeaways:
1. **The Power of Temporal Grades ($R^2$ jump from 0.27 to 0.93):**  
   Excluding earlier period grades (Scenario A: Early) limits predictive power to $R^2 \approx 0.27$, reflecting that demographic, study time, and parental support indicators explain only modest variance in academic grades. Incorporating intermediate grades $G1$ and $G2$ (Scenario B) increases explained variance to over $93\%$, with RMSE dropping from $\approx 3.92$ to $\approx 1.22$ points.
2. **Model Selection Integrity:**  
   According to the strict rule of selecting the best model based **only on training-split cross-validation** (never held-out test data):
   - In Early Regression, **Random Forest** ($\text{CV RMSE} = 3.895$) and **Bagging** ($\text{CV RMSE} = 3.902$) are the winning algorithms.
   - In Grade-Informed Regression, **Random Forest** ($\text{CV RMSE} = 1.668$) and **Bagging** ($\text{CV RMSE} = 1.675$) are selected.
   - Both ensemble bagging variants outperform the single decision tree baseline by over $0.18$–$0.31$ points in CV RMSE.

---

## 2. Classification Analysis: Predicting Academic Risk ($G3 < 10$)

### Performance Matrix:

| Scenario | Model Family | Training CV F1 (Score) | Training CV Std | Train F1 | Held-Out Test F1 | Test Recall | Test Precision | Test Accuracy | Test ROC-AUC | Test Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **Early** | **Decision Tree** | 0.420 | 0.036 | 0.619 | 0.410 | 0.308 | 0.615 | 0.709 | 0.606 | 0.242 |
| **Early** | **Bagging** | **0.448** | 0.104 | 0.863 | 0.462 | 0.346 | 0.692 | 0.734 | 0.747 | 0.188 |
| **Early** | **Random Forest** | 0.290 | 0.084 | 0.870 | 0.294 | 0.192 | 0.625 | 0.696 | 0.705 | 0.198 |
| **Early** | **XGBoost** | 0.419 | 0.057 | 0.939 | **0.558** | 0.462 | 0.706 | **0.759** | 0.731 | 0.188 |
| **Early** | **LightGBM** | **0.446** | 0.052 | 0.816 | 0.533 | 0.462 | 0.632 | 0.734 | **0.745** | **0.185** |
| **Grade-Informed** | **Decision Tree** | **0.892** | 0.031 | 0.892 | 0.852 | 0.885 | 0.821 | 0.899 | 0.919 | 0.079 |
| **Grade-Informed** | **Bagging** | 0.880 | 0.035 | 0.995 | 0.868 | 0.885 | 0.852 | 0.911 | 0.964 | 0.069 |
| **Grade-Informed** | **Random Forest** | 0.868 | 0.056 | 0.981 | **0.885** | 0.885 | 0.885 | **0.924** | 0.967 | 0.083 |
| **Grade-Informed** | **XGBoost** | 0.882 | 0.035 | 0.908 | 0.852 | 0.885 | 0.821 | 0.899 | **0.972** | 0.069 |
| **Grade-Informed** | **LightGBM** | 0.885 | 0.042 | 0.925 | **0.885** | 0.885 | 0.885 | **0.924** | 0.969 | **0.069** |

### Key Classification Takeaways:
1. **Best Model Selection via Training CV:**
   - In Early Risk Classification: **Bagging** ($\text{CV F1} = 0.448$) and **LightGBM** ($\text{CV F1} = 0.446$).
   - In Grade-Informed Risk Classification: **Decision Tree** ($\text{CV F1} = 0.892$) and **LightGBM** ($\text{CV F1} = 0.885$).
2. **ROC-AUC & Ranking Quality:**
   Even when Early F1-score is moderate ($0.45$–$0.56$) due to conservative binary thresholds, **ROC-AUC reaches 0.745–0.747**, confirming strong relative ranking ability for academic prioritization.
3. **Probability Calibration:**
   Brier scores in Grade-Informed classification drop to $\approx 0.069$, indicating sharp, well-separated class probability assignments.

---

## 3. Syllabus Experiments: Deep Empirical Observations

### A. Bias–Variance Tradeoff & Tree Depth
From `results/bias_variance_depth.csv` across 4 repeated validation splits:
- **Depth 1 (Decision Stump):** High training error ($\text{RMSE} \approx 4.15$) and high validation error ($\text{RMSE} \approx 4.28$). **Diagnosis: Severe Underfitting (High Bias).** The model cannot represent the multi-variable interactions in student achievement.
- **Depths 2–3:** Optimal balance in Early scenario. Validation RMSE reaches its minimum ($\approx 3.96$–$4.02$).
- **Depth 5–8:** Training RMSE drops steadily to $3.0$–$2.2$, while validation RMSE begins creeping upward to $4.2$–$4.4$.
- **Unlimited Depth:** Training RMSE collapses to $\approx 0.0$ (memorization of leaf nodes), but validation RMSE worsens significantly to $4.55 \pm 0.38$. **Diagnosis: Severe Overfitting (High Variance).**

### B. Decision Tree Regularization
From `results/tree_regularization.csv`:
- **Unrestricted Baseline:** Train RMSE = $0.0$, Validation RMSE = $4.55$. Large generalization gap ($\Delta = 4.55$).
- **Constrained Depth (`max_depth=4`):** Train RMSE = $3.24$, Validation RMSE = $4.11$. Generalization gap reduced by over $70\%$.
- **Min Leaf Size (`min_samples_leaf=8`):** Train RMSE = $3.45$, Validation RMSE = $4.05$. Highly effective at preventing isolated outlier leaves.
- **Cost-Complexity Pruning (`ccp_alpha=0.08`):** Train RMSE = $3.91$, Validation RMSE = $4.02$. Prunes non-statistically significant subtree branches, achieving the lowest validation variance.

### C. Boosting Regularization (XGBoost & LightGBM)
From `results/boosting_regularization.csv`:
- Less constrained boosting models without L1/L2 penalties overfit the training partition ($\text{Train RMSE} = 1.85$, $\text{Val RMSE} = 4.22$).
- Adding L1 penalty (`reg_alpha = 4.0`) forces sparsity on irrelevant demographic survey questions.
- Adding L2 penalty (`reg_lambda = 10.0`) and setting `learning_rate = 0.045` with row/column subsampling ($0.80$) stabilizes validation loss at $\approx 3.94$.

### D. Bagging vs Single Tree vs Random Forest Scaling
From `results/bagging_estimators.csv`:
- Increasing ensemble size from $N=10$ to $N=75$ estimators progressively smooths out individual tree decision boundaries, reducing validation error variance by $18\%$.
- Beyond $75$ estimators, validation gain exhibits diminishing returns, demonstrating the asymptotic variance bound predicted by ensemble theory.
