# Suggested PPT layout (8 slides)

1. **Title:** Student Performance & Academic Risk Prediction Using Ensemble Learning
2. **Problem/Objective:** Identify broad support needs; grade regression plus binary risk classification.
3. **Dataset and preprocessing:** UCI source/citation, label G3, leakage-safe scenarios and imputation/encoding pipeline.
4. **Algorithms:** Baseline Decision Tree, Bagging, Random Forest, XGBoost, LightGBM.
5. **Week 1:** Screenshot of depth vs train/validation RMSE and explain bias–variance and tree pruning.
6. **Week 2:** Real UCI model comparison table, boosting parameter experiment and feature importance.
7. **Live demo:** Streamlit Overview, Compare, Predict Student, Learning Lab.
8. **Conclusions/limitations:** Results are sample-dependent, avoid unjustified claims, improvements and fairness.

**Reminder:** Obtain screenshot/results only after `python -m src.train --data data/raw/student-mat.csv` and `python -m src.experiments --data data/raw/student-mat.csv`.
