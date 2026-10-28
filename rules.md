# CODEX EXECUTION RULES — HACK O WEEK

Read `PROMPT.md` completely before starting. Follow it as the main project specification, with the execution rules below taking priority for workflow, efficiency, and stopping conditions.

## 1. Main objective

Complete ALL Codex-owned work in **one focused execution session**, minimizing token usage, tool calls, repeated checks, and unnecessary modifications.

Work inside the existing `student_performance_ensemble_project` repository. Do not rebuild the project.

This is a faculty-evaluated ML project. Every syllabus topic needs evidence showing WHAT, WHERE, HOW, WHY, and actual results.

## 2. Mandatory execution order

**STAGE A — Repository Audit and Fixes**

1. Inspect the existing project, dependencies, models, results, tests, and Streamlit application.
2. Identify genuine problems and fix only what is necessary.
3. Verify model loading and Python/package compatibility.
4. Preserve existing trained models, data, and results.
5. Do not rebuild features that already work.

**STAGE B — Dashboard and Faculty Evidence**

1. Improve the existing Streamlit dashboard without unnecessary redesign.
2. Add a Syllabus Evidence section covering Weeks 13–16.
3. Show existing authentic model results, graphs, algorithm details, and implementation references.
4. Every concept must explain its purpose, source-code location, implementation, and available evidence.
5. Preserve all existing working functionality.

**STAGE C — Prepare Colab Integration**

1. Prepare the loader and integration interfaces for future Colab exports.
2. Create the Neural Network Lab UI.
3. Support SGD/Adam graphs, neural-network metrics, and predictions when real artifacts arrive.
4. Show a clear "Awaiting Colab artifacts" state wherever required files are missing.
5. Make the dashboard run normally without new Colab exports.

**STAGE D — Testing and Documentation**

1. Fix relevant bugs and add focused tests.
2. Verify the dashboard, dataset loading, and existing saved-model inference.
3. Add a simple one-command verification script.
4. Update README with setup, execution, dataset source, methodology, and demonstration steps.
5. Prepare the faculty demo guide and Codex handoff summary.

## 3. STRICT DO's

- Work autonomously through Stages A → B → C → D.
- Follow existing architecture and coding conventions.
- Prefer modifying existing modules over creating unnecessary files.
- Use the actual UCI dataset and real existing results.
- Maintain clear separation between historical results and upcoming Colab results.
- Provide honest success/failure reporting.
- Prioritize functionality over UI decoration.
- Keep source-code changes small, maintainable, and easy to demonstrate.
- Report genuine blockers rather than hiding failures.

## 4. STRICT DON'Ts

- DO NOT train or retrain ML models.
- DO NOT implement the NumPy neural network training code.
- DO NOT generate SGD/Adam training results.
- DO NOT create fake datasets, model metrics, plots, or predictions.
- DO NOT modify the Colab notebooks or upcoming ML experiments.
- DO NOT overwrite existing Joblib models.
- DO NOT introduce unnecessary frameworks, databases, APIs, Docker, or paid services.
- DO NOT repeatedly ask me for confirmation.
- DO NOT redesign the entire application.
- DO NOT push to GitHub without authorization.
- DO NOT claim an integration is complete before real Colab artifacts are provided.

## 5. When to wait for ML work

ChatGPT and I are independently performing all ML training in Google Colab.

**DO NOT wait for ML artifacts during Stages A, B, C, or D.**

Finish every task that does not depend on new ML outputs.

If an artifact is unavailable, prepare the integration interface, mark the feature as pending, and continue working on other tasks.

Only AFTER all independent Codex work is completed should you stop and request the following handoff information:

- Exported trained model files
- Real model evaluation metrics
- Neural-network training histories
- Preprocessing and feature schema
- Dataset and experiment manifest
- Experiment results and graphs, where applicable

**Stop at this handoff checkpoint. Do not repeatedly check for files, poll directories, or consume tokens waiting for Colab.**

When I supply those artifacts in a later interaction, complete final integration and inference verification.

## 6. Testing rules

We have limited execution budget. Be efficient.

**Initial checks:** Run the existing tests and model inference smoke test once to identify baseline failures.

**During implementation:** Use focused tests only for modified components.

**Final checks:** Run the complete test suite and inference smoke test once after completing independent implementation.

Required commands, where supported:

```bash
python -m pytest -q
python scripts/smoke_inference.py
```

Also perform a lightweight Streamlit startup/import check where possible.

If a check fails, investigate and fix the actual issue. Rerun only the necessary failed checks, followed by final verification.

Never claim "all tests passed" unless they actually executed successfully.

## 7. Efficiency and context-limit rules

Everything possible should be completed within this single Codex execution budget.

Priority order:

**P0:** Existing app works, model compatibility, authentic evidence, integration interfaces, tests.

**P1:** Neural Network Lab UI, faculty documentation, handoff instructions.

**P2:** Cosmetic improvements, refactoring, optional enhancements.

Do not spend execution budget on P2 while P0 or P1 tasks remain.

Avoid:
- Repeated full-repository scans
- Reading irrelevant files
- Unnecessary package installations
- Duplicated implementations
- Repeated full test-suite executions
- Long progress explanations
- Unnecessary planning loops

If execution limits become tight, complete P0 first, preserve working changes, and clearly report any remaining tasks.

## 8. Final response format

When your independent work is finished, give ONE concise report containing:

1. **COMPLETED:** All implemented tasks.
2. **FILES CHANGED:** Exact repository paths.
3. **TEST RESULTS:** Commands executed and actual results.
4. **PENDING COLAB:** Exactly what ML artifacts are required.
5. **BLOCKERS:** Anything still failing.
6. **RUN COMMAND:** Exact command to open the project.
7. **READY FOR HANDOFF:** Whether I can safely start the integration phase.

Also save the essential handoff information in `CODEX_HANDOFF.md`.

Do not start Colab-dependent work without the actual exports.

## 9. Final instruction

**Start immediately. Execute all independent tasks in one focused run. Do not stop midway to ask routine questions. Finish testing, documentation, and handoff preparation before stopping. Never fabricate ML evidence.**

Your stopping point is:

**"Codex-side implementation complete. Ready for verified Google Colab artifacts."**

Only use that statement if the independent work and required verification are genuinely complete.
