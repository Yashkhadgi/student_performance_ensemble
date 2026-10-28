# 3–5 minute faculty demo

1. Open **Overview**. State that this uses the verified UCI Mathematics dataset: 395 Portuguese secondary-school records, not live or Indian college data.
2. Open **Model Comparison**. Explain that all five model families use the same 80/20 holdout, while candidate selection uses only 3-fold CV on the training portion. Show both `early` (without G1/G2) and `grade_informed` (with G1/G2) scenarios; G3 is never an input.
3. Open **Learning Lab**. Use the tree-depth plot to identify the train-versus-validation gap as an empirical underfit/overfit indicator. Then show tree and booster regularization results; they were produced on development splits, not the final test set.
4. Open **Syllabus Evidence**. Point to each result file and implementation location. This distinguishes evidence from explanations.
5. Open **Neural Network Lab**. Show the supplied Colab export's SGD-versus-Adam loss history and validation table, then the final NN's 79-row held-out metrics and confusion matrix. Explain that the NN is an early-intervention risk classifier, so its comparison belongs only with risk metrics on the matching split.
6. Open **Predict Student**. Enter a hypothetical profile and explain that it is decision support, not a grading or disciplinary system.

Before the demo run `./scripts/verify_demo.sh`, then start with `source .venv/bin/activate && streamlit run app.py`.
