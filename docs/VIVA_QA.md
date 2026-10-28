# Viva preparation — 20 likely questions

1. **What is an ensemble?** Combining outputs of multiple estimators to improve robustness or prediction quality.
2. **What is bagging?** Train base learners on bootstrap-resampled observations, then vote/average their predictions.
3. **What is a bootstrap sample?** A same-size sample drawn with replacement; some original rows may be repeated and others omitted.
4. **Bagging vs Random Forest?** Random Forest uses bagging plus feature subsampling when splitting each tree.
5. **What is boosting?** Sequentially add weak learners to improve an ensemble according to a loss-function objective.
6. **XGBoost vs LightGBM?** Both are gradient-boosted tree systems. LightGBM typically employs leaf-wise tree growth with histogram optimization; XGBoost supports several tree-building methods and explicit regularization.
7. **Bias?** Error arising from a restrictive learning assumption that fails to capture the pattern (high bias can underfit).
8. **Variance?** Sensitivity of fitted model predictions to changes in sampled training data (high variance can overfit).
9. **How do you show overfitting?** Compare training and validation RMSE; a very small training error with worse validation RMSE is evidence.
10. **How do you show underfitting?** A shallow model has poor performance on both train and validation sets.
11. **Why repeated splits?** They reveal model sensitivity; one split does not mathematically identify variance.
12. **What is regularization?** Penalizing model complexity or restricting how complex predictions can become.
13. **What is `ccp_alpha`?** A penalty on decision-tree complexity used for cost-complexity pruning.
14. **What are `reg_alpha` and `reg_lambda`?** L1 and L2 regularization weights in XGBoost/LightGBM.
15. **What is the learning rate?** The shrinkage multiplier for each added boosting tree.
16. **How did you avoid leakage?** G3 is excluded from all inputs, and encoding/imputation are inside fold-fitted pipelines. Earlier grades are excluded in the early scenario.
17. **Why use cross-validation?** To select parameters based on development-set generalization rather than a single training score.
18. **Why precision, recall, F1?** For risk labels, accuracy alone can hide missed at-risk cases; recall tracks captured positives, precision tracks false alarms, F1 balances them.
19. **What is ROC-AUC?** How well model probabilities rank positive vs negative cases across thresholds; it does not establish calibrated probabilities.
20. **Is the output ready for schools?** No; real deployment needs representative data, external validation, fairness and privacy review, monitoring and human oversight.
