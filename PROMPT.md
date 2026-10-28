# CODEX TASK PROMPT — Hack O Week | Student Ensemble Lab

> **Use this file as your sole implementation brief in Codex. Work inside the EXISTING `student_performance_ensemble_project` repository.**
> **Division of labor is mandatory:** YOU (Codex) own repository engineering, application/dashboard, integrations, compatibility, tests, documentation and presentation flow. The student + ChatGPT own Google Colab, **all new ML experiments/training**, the from-scratch NumPy neural network, actual graphs/metrics, and exporting trained artifacts. Do not duplicate their work.

## 0. Objective / context

Build a convincing **faculty-evaluable Hack O Week demonstration** for **Weeks 13–16**, where each curriculum requirement can be proved by **what / where in code / how / why / actual evidence**. The app must remain usable even before the new Colab files arrive, but must never display invented experimental results.

User environment: MacBook Air M1, Python, Streamlit, repo opened in Codex. Student has a short live demo window. Preserve current functionality rather than redesigning or starting from zero.

### Existing repo (verified from supplied ZIP)

- Root: `app.py`, `requirements.txt`, `README.md`, `models/manifest.json`, `models/*.joblib`, `results/*.csv`, `src/{data,models,train,experiments}.py`, `notebooks/01_data_exploration.ipynb` through `04_boosting_comparison.ipynb`, `tests/test_core.py`, `scripts/smoke_inference.py`.
- Existing app navigation: **Overview**, **Predict Student**, **Model Comparison**, **Learning Lab**, **Explainability**, **Data Explorer** (inspect actual repo for any additional screens before editing).
- Mathematics UCI data: `data/raw/student-mat.csv`, **395 rows, 33 columns**, expected SHA-256: `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`.
- 20 serialized sklearn model pipelines (2 scenarios `early`/`grade_informed` × 2 tasks `grade`/`risk` × 5 model families) already exist. `early` excludes `G1`, `G2`, `G3`; `grade_informed` excludes `G3`; risk label is `G3 < 10`.
- Existing evaluation files include: `results/model_metrics.csv`, `bias_variance_depth.csv`, `tree_regularization.csv`, `boosting_regularization.csv`, `bagging_estimators.csv`, `cv_candidates.csv`, `feature_importance.csv`, etc.
- `scikit-learn==1.6.1`, `xgboost==2.1.4`, `lightgbm==4.6.0` are specified in reviewed `requirements.txt`; check the repo before assuming. Saved joblib files were previously incompatible with scikit-learn 1.8.0. Do **not** blindly upgrade sklearn.
- An existing `scripts/smoke_inference.py` tests saved-model predictions; improve/reuse it, do not duplicate it unnecessarily.
- Do not treat past reports of test passes as proof of model-loading success on this machine.

## 1. Ownership boundaries (STRICT)

| CODEX owns and must work on now | ChatGPT + STUDENT via COLAB own |
|---|---|
| Inspect source, fix real bugs and portability issues | Load and verify UCI dataset in live Colab |
| Fix dependency / serialized artifact loading without changing model results | Train/retrain ensemble models in Colab |
| Improve Streamlit experience, evidence navigation, data explanations | Train-vs-validation, bias–variance, over/underfit experiments |
| Prepare integration paths for exported CSV/JSON/joblib artifacts | L1/L2 experiments and actual plots |
| Make app show evidence from existing **real** results, with provenance labels | Implement perceptron, activations, forward/backprop in NumPy |
| Build `Neural Network Lab` UI shell driven by **external verified files** | Train NumPy NN with SGD & Adam, compute real metrics and save models |
| Tests, docs, scripts, reproducibility, optional local Git prep | Export Colab outputs and report filenames/columns, explain findings |
| After artifacts arrive: integrate, verify loading and demo | Decide which exported model results should appear in final comparison |

**DO NOT** implement/train NumPy neural networks, train ensembles, run hyperparameter searches to create replacement metrics, author Colab training notebooks, synthesize outcome CSVs, make up demo results, or overwrite existing trained artifacts as a substitute for missing Colab outputs. Read the existing `src/train.py` only to understand the prediction interfaces. Exception: fixing a small inference/serialization code bug is within scope; if old models cannot load under pinned compatible dependencies, clearly report the blocker instead of secretly retraining.

## 2. Milestone A — audit, reproducibility, model loading (START NOW)

1. Inspect `README.md`, `app.py`, `src/`, `models/manifest.json`, `requirements.txt`, `.gitignore`, `tests/`, and key result files. Make a short factual audit; preserve existing features.
2. Verify current interpreter + installed package versions. Prefer project `.venv` where compatible; if needed, give exact Mac M1 commands to create a supported clean environment. Keep dependencies compatible with existing joblib files. If LightGBM requires `libomp`, report the macOS-specific remedy.
3. Run `python -m pytest -q` and `python scripts/smoke_inference.py`; report the **actual** results, not assumed status. If a command cannot run due to missing dependencies, document exactly which package / version is blocking it, with remediation. Do not claim model compatibility until tested.
4. Inspect loading source: ensure all paths resolve from repo root (not old `/Users/...` paths in archived metadata). Treat `manifest.source.path` as historical metadata; replace UI dependency on it with a relative dataset path if needed.
5. Verify bundled data SHA-256, explain this is **real UCI Portuguese school data**, not Indian college records, not live college data. Never activate synthetic data for academic demonstration. Existing explicitly-marked synthetic fixture may remain for unit tests.
6. Preserve all historical results: no overwriting or retitling them as if trained in the new Colab session.

**Checkpoint A output:** commands executed, actual tests passed/failed, loaded model count out of 20, blocking issues, modifications made.

## 3. Milestone B — faculty-facing evidence UI (WORK NOW, independent of Colab)

Improve/extend the existing Streamlit app minimally. Add a prominent **"Syllabus Evidence / Implementation Proof"** area, either as one new page or a dedicated tab, with clear Week 13–14 and Week 15–16 sections. Present a compact table for each concept:

- **Concept:** Bagging / Random Forest / XGBoost / LightGBM / bias–variance / overfitting & underfitting / L1 & L2 / Joblib deployment / perceptron / ReLU & sigmoid / forward/backprop / BCE loss / SGD vs Adam / NumPy neural network.
- **What:** plain-language explanation (2–3 short sentences).
- **Where:** exact existing code file and Python class/function when available; provide a repo-relative link/label, not a nonexistent line number. For neural net items before Colab integration, explicitly mark **"Awaiting Colab training evidence"** rather than claiming code exists.
- **How:** training/evaluation protocol or brief actionable code pointer (no imaginary training steps).
- **Why:** benefit for performance/generalization.
- **Evidence:** relevant already-generated CSV chart/model, or clear "Not imported yet" state. Show authentic figures only.

For *existing* Week 13–14 results, reuse actual CSVs and saved plots from `results/`: model comparisons, training vs held-out scores, tree depth vs errors, L1/L2 regularization and bootstrap/bagging estimator trends. Label train vs validation vs test accurately; don't misrepresent training plots or validation artifacts. Show both `early` and `grade_informed` scenarios with warning that previous grades G1/G2 are not early predictors.

Preserve original prediction page, options, sidebar navigation, theme, charts and existing exploratory pages. Don't add heavyweight libraries or external APIs. All screens must gracefully handle missing optional results.

## 4. Milestone C — **Colab integration contract** (SCAFFOLD NOW; fill ONLY when data arrives)

The student will export verified artifacts from Google Colab. Prepare the application to accept them without training them yourself.

Suggested incoming destination (you may use current `models/` and `results/` if more maintainable, but document the choice):

```
colab_exports/
  experiment_manifest.json        # data hash, seed, split and package versions
  ensemble_metrics.csv            # scenario/task/model/actual metrics, optional
  bias_variance.csv               # complexity, train/validation errors, optional
  regularization_metrics.csv      # model/alpha/lambda/train/validation metrics, optional
  nn_metrics.csv                  # scenario, optimizer, genuine holdout metrics
  nn_history.csv                  # scenario, optimizer, epoch, train_loss, val_loss
  nn_models/                      # pretrained NN + fitted preprocessing, joblib
```

These are **interfaces/examples, NOT claims that artifacts exist**. Actual filenames/schema will be agreed with the student and ChatGPT when their Colab work finishes. Build a small reusable `src/colab_artifacts.py` (or similar) to:

- Validate existence and basic column schema; load CSV/JSON safely; display actionable file-specific missing-data messages.
- Verify dataset hash, feature schema and labels match this repo before mixing comparisons.
- Check reported training environment, note when Colab and local library versions differ; provide informative joblib errors (no unsafe patching).
- Keep **original archived results** visibly separate from **new Colab-verified results**, including provenance tags and training timestamp/source (when provided).
- Never auto-download arbitrary serialized models from the web or unpickle untrusted external artifacts. Only load user-exported, trusted joblib files.
- Do not automatically switch the Predictor to a newly exported model until genuine load + inference checks pass.
- Make the app work normally if `colab_exports/` does not exist.

### Neural Network Lab (UI now; data later)

Create a **Neural Network Lab** screen or tabs with a lightweight architecture diagram (input → hidden ReLU → output sigmoid), an educational summary of perceptron/activation/forward/backward/BCE/SGD/Adam, a placeholder-ready optimizer selector, and clearly labelled slots for **real SGD vs Adam history**, actual metrics and confusion matrix. When files are absent, show "Colab models not imported yet" and **do not** show invented charts, placeholder scores or a simulated "trained" model. After files arrive, render plots from the exported CSVs. Test selector behavior with temporary **test fixtures only**, never present fixtures as scientific evidence.

## 5. Milestone D — model inference handoff (AFTER Colab export)

When the user adds actual files from Colab:

1. Inspect real artifact structure (NumPy weights + fitted `ColumnTransformer`, sklearn pipeline, or explicitly documented alternative). Don't assume a saved joblib object supports `.predict()` if it's a plain dict; write a small explicit trusted-model adapter matching its actual structure.
2. Verify scenario-specific input fields and exact training feature order using manifest/schema. Never use `G3` as input; `G1/G2` only in `grade_informed`.
3. Perform one real prediction with **existing** saved ensemble pipeline and one with an **imported** NN artifact. Test probability finite/in [0,1], correct shape, correct feature mapping, and correct classification threshold.
4. Compare NN with ensemble **only when scenario/task/test split/metric are comparable**. Explain any differing CV or selection methodology and avoid declaring a winner from invalid comparisons.
5. If library versions differ between Colab and Mac, prefer matching the proven environment or a safe interchange format; do not silently modify scores or regenerate synthetic models.

**Do not idle waiting for new exports:** complete A/B/C first, then tell the user exactly how to hand off the Colab outputs.

## 6. Milestone E — tests, docs, repo hygiene, final run

- Add focused tests for proof-page rendering logic/data loaders, graceful missing Colab files, no bogus "trained" claims, scenario feature schemas, the existing inference smoke path, and imported fixture parsing. Use small fixtures under `tests/` when needed and label them as fixtures.
- Ensure `python -m pytest -q` and `python scripts/smoke_inference.py` work when deps are available; add `scripts/verify_demo.sh` or similarly simple one-command verification. A README command should match a real existing script.
- README additions: prerequisite Python + exact dependency versions as tested; UCI citation and sample limitations; Colab→repo integration instructions; Week 13–16 what/where/how/why/evidence matrix; live Streamlit launch instructions; faculty demo flow.
- Create `docs/FACULTY_DEMO.md` with a **3–5-minute evidence sequence**: UCI source → sample rows → trained algorithms/evaluation → bias/variance → over/underfit → L1/L2 → Colab NN evidence when imported → saved-model live prediction. Include short spoken explanations of why each method was used and disclose when a requirement is still pending.
- Repo hygiene: exclude `.venv`, `.DS_Store`, caches, tokens, credentials; keep small public licensed UCI data & demonstrably usable model artifacts if the demo requires offline playback (respect attribution). Do not run `git push` or publish publicly without user's instruction. Check `.gitignore` before changing it.
- Optionally produce a screenshot-ready evidence matrix in the app, but spend no time on unrelated UI overhauls.

## 7. Working style & non-negotiable quality bar

- **Immediate work is independent of Colab**; don't edit the ML notebook, don't train models, don't create NN implementation. Changes must avoid merge conflicts with the Colab team's forthcoming training code/files.
- At each milestone, print a concise status report with **DONE / PENDING / BLOCKED**, exact repo paths changed, commands run, error traces if unresolved and verification evidence. Continue to the next independent milestone.
- Aim for a fast student demo, not enterprise architecture. Use current Streamlit stack and plotting libraries. Add no paid services, secret tokens, databases, external LLM calls or Docker.
- Do not fabricate graphs, metrics, confusion matrices, experiments, saved model files, dataset hashes, screenshots or successful terminal runs.
- Do not pretend integration is complete before Colab export arrives.
- Avoid needless clarifying questions. Use facts in repo and make sensible decisions; if an unknown input schema genuinely blocks final integration, finish other work and report one clearly scoped handoff request.

## 8. Definition of done (Codex side)

- Existing dashboard runs on local Mac and unchanged existing features work.
- 20 historical model artifacts actually load and predict, or precise environment blocker is identified.
- Syllabus evidence UI covers Week 13–16 honestly; *only verified* evidence appears.
- Colab integration loader and NN Lab are ready **without** new model files; clear pending state is visible.
- Test suite and smoke checks run with results recorded.
- Documentation provides commands, concept mapping, evidence locations and speaking order.
- You hand off a short **`CODEX_HANDOFF.md`** listing: (a) modified files, (b) tested versions and outputs, (c) which new Colab filenames/schema you need, (d) what must be done after export, and (e) exact app start command.

**START: inspect the existing repository, run tests/smoke, then implement A → B → C → E. Save D for when the student provides trained Colab artifacts.**
