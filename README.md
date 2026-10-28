# 🎓 Student Ensemble Lab
### Student Performance & Academic Risk Prediction using Bagging, XGBoost, LightGBM and Bias–Variance Experiments

A complete, **local-first, Python + scikit-learn + Streamlit** Semester 5 Machine Learning mini-project. It contains two prediction tasks, two leakage-safe scenarios, five model families, training-only cross-validation, comparative error diagnostics, bias-variance and regularization studies, four walkthrough notebooks, and a multi-page web dashboard.

> **Dataset Status:** Fully migrated to the **VERIFIED REAL UCI Student Performance dataset** (`student-mat.csv`, $N=395$). The dataset is cryptographically verified against the official UC Irvine Machine Learning Repository archive (SHA-256: `e47f9ee225e1...`). All 20 model pipelines and experiment CSVs are trained and evaluated on authentic real observations. Silent fallbacks to synthetic data have been completely eliminated.

## 1) Prerequisites

- **Python 3.9, 3.11, or 3.12** supported. MacBook Air (M1/M2/M3) or Windows/Linux.
- Internet connection for initial package installation.
- No paid API keys, no GPU, no TensorFlow, **no manual pre-training** required.
- Roughly 500 MB free disk space for the local virtual environment.

## 2) Quick start on Mac (Apple Silicon)

Open **Terminal** and navigate to the project directory:

```bash
cd ~/Downloads/student_performance_ensemble_project
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app should open in your browser at **http://localhost:8501**. The package already has **demo-trained models**, so the dashboard opens immediately after package installation.

> If `lightgbm` complains about **libomp.dylib** on an Apple Silicon Mac, install [Homebrew](https://brew.sh/) if necessary and run `brew install libomp`. Then try `python -c "import lightgbm; print(lightgbm.__version__)"`.
>
> If `python3` points to an unsupported version, use `python3.12` for the virtual environment instead. Do not use a global `sudo pip install`.

### Windows equivalent (PowerShell)

```powershell
cd "$HOME\Downloads\student_performance_ensemble_project"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell prevents activation, use Command Prompt instead: `.venv\Scripts\activate.bat`.

## 3) Replace demo models with REAL data (recommended for submission)

Use these commands from the project root **after installing requirements**:

```bash
# 1. Downloads the real Mathematics AND Portuguese UCI CSVs into data/raw/
python -m src.data --download

# 2. Fits 20 pipelines, with train-only 3-fold CV and 3 parameter candidates/model
python -m src.train --data data/raw/student-mat.csv

# 3. Runs repeated bias–variance and tree/booster regularization experiments
python -m src.experiments --data data/raw/student-mat.csv

# 4. Launch dashboard with real-data results
streamlit run app.py
```

Restart the Streamlit server after retraining to refresh cached models. For the larger Portuguese dataset, replace `student-mat.csv` with `student-por.csv` in **both** training commands. Do **not** combine the two course datasets blindly, since the same student may appear in both.

### If the UCI download fails

1. Visit https://archive.ics.uci.edu/dataset/320/student+performance
2. Download and unzip `student.zip` (sometimes inside an outer ZIP).
3. Place `student-mat.csv` or `student-por.csv` in `data/raw/`.
4. Run steps 2–4 above. Files must retain their **semicolon-separated** CSV format.

### Offline demo training (purely synthetic)

```bash
python -m src.data --demo
python -m src.train --data data/demo/student-demo-SYNTHETIC.csv --fast
python -m src.experiments --data data/demo/student-demo-SYNTHETIC.csv
streamlit run app.py
```

The `--fast` flag runs a **single prespecified candidate/model** with CV instead of selecting from three candidates. It is ideal for sanity checks. For your presentation, **omit `--fast`** while training on UCI data.

### Run automated tests
Run the focused unit and integration suite:

```bash
python -m pytest -v
```

### Faculty-ready verification and current artifact run

The bundled historical artifacts were trained on the verified UCI Mathematics dataset and are preserved as archived evidence. Do not retrain them during the demonstration. Run the complete local verification before presenting:

```bash
bash scripts/verify_demo.sh
source .venv/bin/activate
streamlit run app.py
```

`scripts/smoke_inference.py` loads each of the 20 trusted historical pipelines, checks the UCI SHA-256 fingerprint and performs real predictions. The dashboard needs scikit-learn `1.6.1`, XGBoost `2.1.4`, and LightGBM `4.6.0`, matching the artifacts verified in this repository.

### Google Colab neural-network handoff

The **Neural Network Lab** is intentionally a viewer until verified Colab outputs are copied to `colab_exports/`. It displays no invented NN metrics, charts, predictions, or trained status. Export `experiment_manifest.json` first, then genuine `nn_history.csv` and `nn_metrics.csv`; add `nn_predictions.csv` to show a held-out confusion matrix. The exact validation rules and columns are in [docs/COLAB_CONTRACT.md](docs/COLAB_CONTRACT.md). After that handoff, Codex can inspect the trusted model format and run a real local inference check before enabling NN prediction.

The supplied `hack_o_week_exports.zip` has now been integrated under `colab_exports/` as a validated legacy export. Its manifest declares the real UCI Mathematics 316/79 split with seed 42; all 20 ensemble pipelines and the NumPy NN passed local inference. The Neural Network Lab shows the genuine SGD/Adam history, their validation comparison, and a real 79-row held-out confusion matrix generated from the exported NN and preprocessing artifact.

### Complete Project Audit & Reports in `docs/`
- [docs/FINAL_AUDIT_REPORT.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/FINAL_AUDIT_REPORT.md) — Comprehensive requirement-by-requirement audit
- [docs/REAL_DATA_STATUS.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/REAL_DATA_STATUS.md) — Official UCI dataset cryptographic verification report
- [docs/IMPLEMENTATION_LOG.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/IMPLEMENTATION_LOG.md) — Detailed code and engineering changelog
- [docs/TEST_RESULTS.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/TEST_RESULTS.md) — Pytest test execution logs and validation evidence
- [docs/ML_EXPERIMENT_FINDINGS.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/ML_EXPERIMENT_FINDINGS.md) — In-depth syllabus findings on Ensembles, Bias-Variance, and Regularization
- [docs/FINAL_PROJECT_READINESS.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/FINAL_PROJECT_READINESS.md) — Viva defense Q&A matrix and readiness assessment
- [docs/NEXT_TASK_READINESS.md](file:///Users/yashkhadgi/Downloads/student_performance_ensemble_project/docs/NEXT_TASK_READINESS.md) — Final clearance and optional next steps

## 4) Implemented ML syllabus topics

| Topic | Working implementation | Output evidence |
|---|---|---|
| Underfitting | Tree depths 1 and 2 compared with deeper trees | `results/bias_variance_depth.csv` and depth plots |
| Overfitting | Unrestricted trees, train-vs-validation gap | Same CSV and Learning Lab graphs |
| Bias–variance tradeoff | Repeated 80/20 *development* splits, depth/error stability | Four repeats by default; no claims of exact bias decomposition |
| Bagging | `BaggingRegressor` / `BaggingClassifier` with bootstrap trees | Separate bagging model metrics and test predictions |
| Random Forest | Bootstrap + per-split feature sampling | RF models and comparison |
| Boosting | `XGBRegressor`, `XGBClassifier`, `LGBMRegressor`, `LGBMClassifier` | All models trained and compared |
| Regularization | Tree `max_depth`, `min_samples_leaf`, `ccp_alpha`; booster leaf/depth, L1/L2 | `tree_regularization.csv`, `boosting_regularization.csv` |
| Model selection | 3-fold cross-validation on 80% training data | `cv_candidates.csv` |
| Regression | Predict `G3` (0 to 20) | MAE, RMSE, R²; actual-versus-predicted scatter |
| Classification | Predict `G3 < 10` | Precision, recall, F1, ROC-AUC, confusion matrix |
| Interpretation | One-hot-aware aggregated tree feature importance | `feature_importance.csv`, Explainability page |

### Scientific setup and data leakage

- **Early intervention** uses all columns except `G1`, `G2`, `G3`.
- **Grade-informed** uses `G1` and `G2`, but never `G3` as an input.
- Labels are: numerical `G3` for regression; `1` if `G3 < 10`, otherwise `0` for risk classification.
- The same seeded, stratified **80% train / 20% held-out test** indexes are used for every model and scenario.
- A `ColumnTransformer` and `Pipeline` fit imputation and one-hot encoding *inside the training folds*, not globally.
- CV selects candidates; the test set is reported **only after selection**. This is not nested CV, so CV model-selection scores can be optimistic and test scores are noisy on small samples.
- Bias–variance and regularization experiments are restricted to further splits of the development set.
- Academic support bands from risk probabilities are descriptive only and **not probability-calibrated**.

## 5) Streamlit dashboard

**Overview:** Project, population size, syllabus mapping, scenario differences.

**Predict Student:** Hypothetical academic profile; predicted grade and failure-risk probability from chosen models. Unedited inputs are filled by median or most-common values from the training dataset; not a real assessment.

**Model Comparison:** Train/test/CV metrics and fit times for all five models, confusion matrix, actual/predicted plots and cross-validation candidates.

**Learning Lab:** Decision-tree depth curves, pruning/leaf-size comparison, XGBoost versus LightGBM regularization effects.

**Explainability:** Tree-based feature importance for all five algorithms and RF/boosting comparison.

**Data Explorer:** Grades and class distributions, summary statistics.

**Methodology:** Clear leakage rules and limitations.

## 6) Folder structure

```text
student_performance_ensemble_project/
├── app.py                     # Streamlit app (7 pages)
├── src/
│   ├── data.py                # UCI downloader, validation, synthetic fallback
│   ├── models.py              # 5 model factories + preprocessing + tunable configs
│   ├── train.py               # 20 pipelines, CV selection, holdout metrics
│   └── experiments.py         # Depth, repeated splits, pruning, XGB/LGBM regularization
├── data/
│   ├── raw/                   # Put downloaded real UCI CSV files here
│   ├── demo/                  # Clearly marked synthetic test CSV
│   └── active_dataset.csv     # Exact dataset last used in training
├── models/                    # 20 trained demo artifacts + manifest.json
├── results/                   # CV, metrics, predictions, importances, experiment CSVs
│   └── figures/               # Generated experiment PNG plots
├── notebooks/                 # 4 reproducible educational walkthroughs
├── docs/                      # Dataset citation, report guide and viva Q&A
├── tests/                     # 13 automated checks
├── requirements.txt
└── README.md
```

**Notebooks** can be run from the project root using `jupyter lab` (JupyterLab included in requirements). They reference saved results so you can learn the concepts without rerunning every model. Training scripts remain the single source of truth for experiment generation.

## 7) What to say during your viva

> "We trained a Decision Tree, Bagging, Random Forest, XGBoost and LightGBM on the same development and held-out test split. Two separate experiments evaluate numerical final grades and binary academic risk. We measured train/validation error for different tree depths to demonstrate underfitting, overfitting, and the bias–variance tradeoff. Then we compared tree pruning and L1/L2-style boosting regularization, and used training-only cross-validation for selection. The UCI results and their generalization limits are reported transparently."

## 8) Dataset attribution and ethics

**Cortez, P. (2008). Student Performance.** UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. It is derived from two Portuguese secondary schools; it **is not** a study of modern Indian university students. Metadata: https://archive.ics.uci.edu/dataset/320/student+performance.

This is an educational support demonstrator. The data include demographic and socioeconomic variables, and the system may embed bias. Do not use this app for real student grading, selection, automated disciplinary decisions, or interventions without privacy review, external validation, fairness testing and human oversight.

## 9) Troubleshooting

| Error | Fix |
|---|---|
| `zsh: command not found: streamlit` | Activate `.venv`, then `pip install -r requirements.txt`, or run `python -m streamlit run app.py` |
| `No module named src` | Execute commands from the root folder containing `app.py` |
| `libomp.dylib not loaded` | On Apple Silicon: `brew install libomp`; reopen terminal and activate `.venv` |
| `UCI download failed` | Download CSV manually from UCI and place in `data/raw/` |
| `model not found` | Run `python -m src.train --data ...` and restart Streamlit |
| `Streamlit port already in use` | `streamlit run app.py --server.port 8502` |
| Everything says synthetic | You are using included demo models. Train on `data/raw/student-mat.csv` then restart. |
| Python module install errors | Use Python 3.11 or 3.12, upgrade pip, use fresh venv |

---

**No pre-trained foundation models, LLM API calls, or external databases** are required. Training is entirely local and CPU-friendly.

## 10) Faculty evidence map

| Week/topic | What to show | Code / authentic evidence |
|---|---|---|
| 13–14 ensembles | Bagging, Random Forest, XGBoost and LightGBM on common scenarios | `src/models.py`, `results/model_metrics.csv`, `results/test_predictions.csv` |
| Bias–variance / overfit | Train vs development-validation error at multiple tree depths | `src/experiments.py`, `results/bias_variance_depth.csv` |
| L1/L2 / constraints | Booster penalties and tree pruning/leaf constraints | `results/boosting_regularization.csv`, `results/tree_regularization.csv` |
| Deployment | Serialized historical pipelines and real inference | `app.py`, `models/manifest.json`, `scripts/smoke_inference.py` |
| 15–16 NumPy NN | Perceptron, ReLU/sigmoid, BCE, backprop, SGD vs Adam | Neural Network Lab; awaiting verified Colab training evidence |

Use [docs/FACULTY_DEMO.md](docs/FACULTY_DEMO.md) for the concise speaking sequence. The dashboard’s **Syllabus Evidence** page states what each concept means, where it is implemented, how it was evaluated, why it is useful, and the available evidence.
