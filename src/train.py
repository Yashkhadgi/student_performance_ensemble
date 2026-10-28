"""Train 2 scenarios × 2 tasks × 5 models with training-only CV and final holdout testing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, max_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss)
from .data import ROOT, RAW, DEMO_DIR, load_data, feature_frame, source_info
from .models import MODELS, make_pipeline, configurations, SEED


def slug(name: str) -> str:
    return name.lower().replace(' ', '_')


def safe_auc(y, prob):
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float('nan')


def evaluate(task: str, y, prediction, probability=None) -> dict:
    if task == 'grade':
        return {
            'mae': round(float(mean_absolute_error(y, prediction)), 5),
            'rmse': round(float(np.sqrt(mean_squared_error(y, prediction))), 5),
            'r2': round(float(r2_score(y, prediction)), 5),
            'max_error': round(float(max_error(y, prediction)), 5)
        }
    res = {
        'accuracy': round(float(accuracy_score(y, prediction)), 5),
        'precision': round(float(precision_score(y, prediction, zero_division=0)), 5),
        'recall': round(float(recall_score(y, prediction, zero_division=0)), 5),
        'f1': round(float(f1_score(y, prediction, zero_division=0)), 5),
        'roc_auc': round(safe_auc(y, probability), 5) if probability is not None else None
    }
    if probability is not None:
        res['brier_score'] = round(float(brier_score_loss(y, probability)), 5)
    return res


def get_feature_importances(pipeline, original_columns: list[str]) -> list[dict]:
    model = pipeline.named_steps['model']
    transformed = pipeline.named_steps['prep'].get_feature_names_out()
    if hasattr(model, 'feature_importances_'):
        imp = np.asarray(model.feature_importances_)
    elif hasattr(model, 'estimators_'):
        imp = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    else:
        return []
    if len(imp) != len(transformed):
        return []
    # Aggregate one-hot encoded levels back to original input fields.
    aggregated = {name: 0.0 for name in original_columns}
    for name, value in zip(transformed, imp):
        field = name.split('__', 1)[-1]
        origin = next((c for c in original_columns if field == c or field.startswith(c + '_')), None)
        if origin is not None:
            aggregated[origin] += float(value)
    return [{'feature': c, 'importance': round(v, 8)} for c, v in
            sorted(aggregated.items(), key=lambda item: item[1], reverse=True)]


def train_all(data_path: Path | str, output=ROOT, fast=False, cv_folds=3) -> dict:
    output = Path(output)
    data_path = Path(data_path)
    model_dir = output / 'models'
    result_dir = output / 'results'
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    source = source_info(data_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    manifest = {
        'source': source,
        'trained_at': now_iso,
        'rows': len(df),
        'columns': len(df.columns),
        'class_balance': {
            'at_risk': int((df.G3 < 10).sum()),
            'not_at_risk': int((df.G3 >= 10).sum()),
            'risk_rate': round(float((df.G3 < 10).mean()), 4)
        },
        'seed': SEED,
        'test_fraction': 0.2,
        'cv_folds': cv_folds,
        'fast': fast,
        'models': {},
        'schema': {},
        'notes': 'CV uses training data only. Held-out test metrics are for final reporting, not model selection.'
    }

    all_results, all_cv, all_preds, all_imp = [], [], [], []

    for scenario in ('early', 'grade_informed'):
        x = feature_frame(df, scenario)
        y_grade = df.G3.astype(float)
        y_risk = (df.G3 < 10).astype(int)
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, stratify=y_risk, random_state=SEED
        )
        xt, xv = x.iloc[train_idx], x.iloc[test_idx]

        manifest['schema'][scenario] = {
            'fields': list(x.columns),
            'defaults': {
                c: (float(x[c].median()) if pd.api.types.is_numeric_dtype(x[c])
                    else str(x[c].mode(dropna=True).iloc[0])) for c in x.columns
            },
            'options': {
                c: sorted([str(v) for v in x[c].dropna().unique()]) for c in x.select_dtypes(exclude='number').columns
            }
        }

        for task in ('grade', 'risk'):
            y = y_grade if task == 'grade' else y_risk
            yt, yv = y.iloc[train_idx], y.iloc[test_idx]
            cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
                  if task == 'risk' else KFold(n_splits=cv_folds, shuffle=True, random_state=SEED))
            scoring = 'neg_root_mean_squared_error' if task == 'grade' else 'f1'

            for name in MODELS:
                label = f'{scenario}__{task}__{slug(name)}'
                candidate_list = configurations(name, fast)
                print(f'[{label}] fitting {len(candidate_list)} candidate(s)...', flush=True)
                started = time.perf_counter()
                best_score = -np.inf
                best_params, best_cv_sd = {}, np.nan

                for i, params in enumerate(candidate_list):
                    pipe = make_pipeline(name, task, xt, params)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        scores = cross_val_score(
                            pipe, xt, yt, scoring=scoring, cv=cv, n_jobs=1, error_score='raise'
                        )
                    score = float(scores.mean())
                    all_cv.append({
                        'scenario': scenario,
                        'task': task,
                        'model': name,
                        'candidate': i + 1,
                        'params': json.dumps(params),
                        'cv_mean_score': score,
                        'cv_std': float(scores.std()),
                        'scoring': scoring
                    })
                    if score > best_score:
                        best_score, best_params, best_cv_sd = score, params, float(scores.std())

                best = make_pipeline(name, task, xt, best_params)
                best.fit(xt, yt)
                ptr, pte = best.predict(xt), best.predict(xv)

                if task == 'grade':
                    ptr, pte = np.clip(ptr, 0, 20), np.clip(pte, 0, 20)
                    train_scores = evaluate(task, yt, ptr)
                    test_scores = evaluate(task, yv, pte)
                    proba = [None] * len(xv)
                    residuals = [round(float(a - p), 5) for a, p in zip(yv.to_numpy(), pte)]
                else:
                    ptr_proba = best.predict_proba(xt)[:, 1]
                    pte_proba = best.predict_proba(xv)[:, 1]
                    train_scores = evaluate(task, yt, ptr, ptr_proba)
                    test_scores = evaluate(task, yv, pte, pte_proba)
                    proba = pte_proba
                    residuals = [round(float(a - pr), 5) for a, pr in zip(yv.to_numpy(), pte_proba)]

                elapsed = time.perf_counter() - started
                model_file = f'{label}.joblib'
                joblib.dump(best, model_dir / model_file, compress=3)

                imp = get_feature_importances(best, list(xt.columns))
                for item in imp:
                    all_imp.append(dict(scenario=scenario, task=task, model=name, **item))

                row = {
                    'scenario': scenario,
                    'task': task,
                    'model': name,
                    'cv_score': round(float(best_score), 5),
                    'cv_std': round(float(best_cv_sd), 5) if not np.isnan(best_cv_sd) else 0.0,
                    'cv_scoring': scoring,
                    'tuning_params': json.dumps(best_params),
                    'train_seconds': round(elapsed, 3),
                    **{f'train_{k}': v for k, v in train_scores.items()},
                    **{f'test_{k}': v for k, v in test_scores.items()}
                }
                all_results.append(row)

                for ix, actual, pred, prob, res in zip(test_idx, yv.to_numpy(), pte, proba, residuals):
                    all_preds.append({
                        'scenario': scenario,
                        'task': task,
                        'model': name,
                        'row_index': int(ix),
                        'actual': float(actual),
                        'predicted': float(pred),
                        'risk_probability': float(prob) if prob is not None else np.nan,
                        'residual': float(res)
                    })

                manifest['models'][label] = {
                    'scenario': scenario,
                    'task': task,
                    'name': name,
                    'file': model_file,
                    'params': best_params,
                    'cv_mean': best_score,
                    'cv_std': best_cv_sd,
                    'test_metrics': test_scores
                }
                print(f'  -> CV={best_score:.3f}; held-out={test_scores}; {elapsed:.1f}s', flush=True)

    pd.DataFrame(all_results).to_csv(result_dir / 'model_metrics.csv', index=False)
    pd.DataFrame(all_cv).to_csv(result_dir / 'cv_candidates.csv', index=False)
    pd.DataFrame(all_preds).to_csv(result_dir / 'test_predictions.csv', index=False)
    pd.DataFrame(all_imp).to_csv(result_dir / 'feature_importance.csv', index=False)

    # Copy the exact dataset file so active_dataset.csv retains the identical SHA-256 fingerprint
    target_active = output / 'data' / 'active_dataset.csv'
    target_active.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_path, target_active)

    (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nTraining complete: {len(all_results)} models across both tasks & scenarios.')
    print(f'Dataset provenance: {source["status"]} (SHA-256: {source["sha256_short"]})')
    print(f'Results saved to: {result_dir / "model_metrics.csv"}')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train all 20 models and CV comparisons')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--data', type=Path, default=None, help='Explicit path to semicolon CSV dataset')
    group.add_argument('--demo', action='store_true', help='Explicitly train on SYNTHETIC demo data')
    ap.add_argument('--fast', action='store_true', help='Train standard configurations without CV tuning')
    ap.add_argument('--folds', type=int, default=3, help='Cross-validation folds (default: 3)')
    args = ap.parse_args()

    if args.folds < 2:
        ap.error('folds must be 2 or more')

    if args.demo:
        dataset_path = DEMO_DIR / 'student-demo-SYNTHETIC.csv'
        if not dataset_path.exists():
            print('Synthetic demo dataset missing. Generating now...')
            from .data import make_demo_dataset
            dataset_path = make_demo_dataset()
    elif args.data:
        dataset_path = args.data
        if not dataset_path.exists():
            ap.error(f'Specified dataset not found: {dataset_path}')
    else:
        # Default to official real UCI dataset
        dataset_path = RAW / 'student-mat.csv'
        if not dataset_path.exists():
            ap.error(
                f'Real dataset not found at {dataset_path}.\n'
                'The training workflow does NOT silently fall back to synthetic data.\n'
                'To download the official UCI dataset, run:\n'
                '    python -m src.data --download\n'
                'To explicitly train on synthetic demo data instead, run:\n'
                '    python -m src.train --demo'
            )

    train_all(dataset_path, fast=args.fast, cv_folds=args.folds)
