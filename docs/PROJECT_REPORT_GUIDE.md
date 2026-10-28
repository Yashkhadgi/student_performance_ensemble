# Mini-project report: Student Performance & Academic Risk Prediction Using Ensemble Learning

**Important:** All numeric results in the packaged default artifacts use **SYNTHETIC demo data**, not UCI. Re-run training/experiments with the real CSV before filling in final metrics below.

## Abstract

This project implements and evaluates tree-based machine-learning techniques for predicting students' final grades and identifying a possible risk of failing. The core syllabus topics are Bagging, Random Forest, Boosting (XGBoost and LightGBM), overfitting, underfitting, the bias–variance tradeoff, and regularization. A consistent held-out split and cross-validation process are used to compare models. An interactive Streamlit application visualizes the results and permits demonstrations with hypothetical student profiles. The tool is intended only for educational decision-support analysis.

## Problem and objectives

Some students may benefit from additional help before the end of a school period. This project studies (1) final grade regression using G3 and (2) risk classification using the target G3 below 10. The objective is not to make definitive personal judgments; it is to understand how different ML algorithms generalize.

## Data description

Use the UCI Student Performance dataset. Select Mathematics, `student-mat.csv` (395 rows), **or** Portuguese, `student-por.csv` (649 rows). Identify the chosen subject, number of usable rows, feature types, class balance, and limitations. Do not mix dataset versions in the same reported experiment without additional methodology.

## Preprocessing and experimental method

1. Validate grade columns are numeric in the 0–20 range.
2. Create two feature scenarios: early excludes G1/G2; later includes them. Always exclude G3.
3. Encode categorical inputs using `OneHotEncoder` inside a `Pipeline`; impute with per-training-fold medians or modes.
4. Split all observations once into stratified 80/20 training/held-out test partitions (stratify on risk label).
5. For each task/scenario/model run 3-fold training-only CV over three candidate parameter settings.
6. Train the best-CV candidate on the full training portion and report metrics once on the held-out test partition.
7. Measure regression MAE/RMSE/R²; measure classification precision/recall/F1/AUC. Show confusion matrix and actual vs predicted scatter.
8. Use further splits of **training data only** for bias–variance and regularization analysis.

## Model details

- **Decision Tree:** Interpretability and under/overfit baseline.
- **Bagging:** Aggregates bootstrap-trained tree predictions to stabilize estimates.
- **Random Forest:** Bagging plus random feature selection within decision trees.
- **XGBoost:** Gradient boosting with shrinkage, subsampling and L1/L2 regularization.
- **LightGBM:** Efficient gradient boosting with configurable leaf complexity and L1/L2 regularization.

## Results to insert after REAL UCI training

Paste rows from `results/model_metrics.csv` filtered by scenario and task. Select the model by `cv_score` (higher is better; grade CV score is *negative RMSE*). Do NOT select the "winner" by lowest test error. Explain when complex boosted models underperform simpler models on a small dataset.

Insert figures from `results/figures/` and screenshots of the Streamlit comparison/prediction pages. Use `bias_variance_depth.csv`, `tree_regularization.csv`, `boosting_regularization.csv` to interpret what changed.

## Interpreting bias–variance

A shallow tree can fail to capture relationships (high bias); an unconstrained tree can memorize the training data (high variance and an error gap on validation data). The depth curves show practical train/validation evidence, **not** an exact decomposition into squared bias, variance, and irreducible noise. Repeated splits show sensitivity to data partitions. Discuss RMSE trends, not just broad theoretical claims.

## Limitations and future scope

The source population is Portuguese secondary schools. The sample is small and not necessarily representative of another country or university setting; grades, surveys and class definitions may differ. Risk probabilities are not calibrated and support bands are heuristic. Correlations are not causal, and some attributes could be unfair proxies. Future work could explore proper probability calibration, time-based evaluation, sensitive-attribute auditing (with approval), external validation and simpler reduced-feature variants.

## References

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. CC BY 4.0.

Cortez, P. & Silva, A. (2008). Using data mining to predict secondary school student performance. *Proceedings of 5th Future Business Technology Conference*.
