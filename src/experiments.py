"""Experiments based entirely on the training split, not the held-out test set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from .data import ROOT, RAW, DEMO_DIR, load_data, feature_frame
from .models import SEED, make_pipeline


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, np.clip(p, 0, 20))))


def run_experiments(data_path: Path | str, output=ROOT, repeats=4):
    df = load_data(data_path)
    result_dir = Path(output) / 'results'
    figures = result_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    depth_rows, reg_rows, boost_rows, bag_rows = [], [], [], []

    for scenario in ('early', 'grade_informed'):
        x = feature_frame(df, scenario)
        y = df.G3.astype(float)
        train_idx, _ = train_test_split(
            np.arange(len(df)), test_size=0.2, stratify=(df.G3 < 10).astype(int), random_state=SEED
        )
        xx, yy = x.iloc[train_idx], y.iloc[train_idx]

        # 1. Bias-Variance & Tree Depth complexity curve (Depths: 1, 2, 3, 5, 8, 10, Unlimited)
        for seed in range(repeats):
            xtr, xval, ytr, yval = train_test_split(xx, yy, test_size=0.25, random_state=SEED + seed)
            for depth in (1, 2, 3, 5, 8, 10, None):
                depth_label = str(depth) if depth is not None else 'Unlimited'
                model = make_pipeline(
                    'Decision Tree', 'grade', xtr,
                    {'max_depth': depth, 'min_samples_leaf': 1, 'ccp_alpha': 0.0}
                )
                model.fit(xtr, ytr)
                depth_rows.append({
                    'scenario': scenario,
                    'repeat': seed,
                    'depth': depth_label,
                    'train_rmse': rmse(ytr, model.predict(xtr)),
                    'validation_rmse': rmse(yval, model.predict(xval))
                })

            # 2. Decision Tree Regularization (Depth, Leaf, Split, CCP Alpha)
            for label, params in [
                ('Unrestricted Baseline', {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'ccp_alpha': 0.0}),
                ('Constrained Depth (depth=4)', {'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'ccp_alpha': 0.0}),
                ('Min Leaf Size (leaf=8)', {'max_depth': None, 'min_samples_leaf': 8, 'min_samples_split': 2, 'ccp_alpha': 0.0}),
                ('Min Sample Split (split=12)', {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 12, 'ccp_alpha': 0.0}),
                ('Cost-Complexity Pruned (α=0.08)', {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'ccp_alpha': 0.08}),
                ('Cost-Complexity Pruned (α=0.25)', {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'ccp_alpha': 0.25}),
            ]:
                model = make_pipeline('Decision Tree', 'grade', xtr, params)
                model.fit(xtr, ytr)
                reg_rows.append({
                    'scenario': scenario,
                    'repeat': seed,
                    'regularization': label,
                    'train_rmse': rmse(ytr, model.predict(xtr)),
                    'validation_rmse': rmse(yval, model.predict(xval))
                })

        # 3. Boosting Regularization (XGBoost & LightGBM: L1, L2, Depth, Leaves, Subsample)
        xtr, xval, ytr, yval = train_test_split(xx, yy, test_size=0.25, random_state=SEED)
        for family, configurations in [
            ('XGBoost', [
                ('Less constrained', {'n_estimators': 180, 'max_depth': 7, 'learning_rate': 0.15,
                                     'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}),
                ('L1 Regularized (reg_alpha=4.0)', {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.06,
                                                   'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 4.0, 'reg_lambda': 0.0}),
                ('L2 Regularized (reg_lambda=10.0)', {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.06,
                                                    'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.0, 'reg_lambda': 10.0}),
                ('Conservative Regularized', {'n_estimators': 110, 'max_depth': 3, 'learning_rate': 0.045,
                                             'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 12.0})
            ]),
            ('LightGBM', [
                ('Less constrained', {'n_estimators': 180, 'num_leaves': 31, 'max_depth': -1,
                                     'min_child_samples': 2, 'reg_alpha': 0.0, 'reg_lambda': 0.0}),
                ('L1 Regularized (reg_alpha=4.0)', {'n_estimators': 120, 'num_leaves': 12, 'max_depth': 4,
                                                   'min_child_samples': 12, 'reg_alpha': 4.0, 'reg_lambda': 0.0}),
                ('L2 Regularized (reg_lambda=10.0)', {'n_estimators': 120, 'num_leaves': 12, 'max_depth': 4,
                                                    'min_child_samples': 12, 'reg_alpha': 0.0, 'reg_lambda': 10.0}),
                ('Conservative Regularized', {'n_estimators': 110, 'num_leaves': 7, 'max_depth': 3,
                                             'min_child_samples': 20, 'reg_alpha': 2.0, 'reg_lambda': 12.0})
            ])
        ]:
            for label, params in configurations:
                model = make_pipeline(family, 'grade', xtr, params)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(xtr, ytr)
                boost_rows.append({
                    'scenario': scenario,
                    'model': family,
                    'configuration': label,
                    'train_rmse': rmse(ytr, model.predict(xtr)),
                    'validation_rmse': rmse(yval, model.predict(xval)),
                    'params': json.dumps(params)
                })

        # 4. Bagging vs Single Tree vs Random Forest Scaling
        dt_base = make_pipeline('Decision Tree', 'grade', xtr, {'max_depth': 7, 'min_samples_leaf': 2})
        dt_base.fit(xtr, ytr)
        dt_train_rmse = rmse(ytr, dt_base.predict(xtr))
        dt_val_rmse = rmse(yval, dt_base.predict(xval))

        for n_est in [10, 25, 50, 75, 120]:
            bag = make_pipeline('Bagging', 'grade', xtr, {'n_estimators': n_est})
            bag.fit(xtr, ytr)
            rf = make_pipeline('Random Forest', 'grade', xtr, {'n_estimators': n_est})
            rf.fit(xtr, ytr)

            bag_rows.append({
                'scenario': scenario, 'model': 'Decision Tree (Single)', 'n_estimators': n_est,
                'train_rmse': dt_train_rmse, 'validation_rmse': dt_val_rmse
            })
            bag_rows.append({
                'scenario': scenario, 'model': 'Bagging Ensemble', 'n_estimators': n_est,
                'train_rmse': rmse(ytr, bag.predict(xtr)), 'validation_rmse': rmse(yval, bag.predict(xval))
            })
            bag_rows.append({
                'scenario': scenario, 'model': 'Random Forest', 'n_estimators': n_est,
                'train_rmse': rmse(ytr, rf.predict(xtr)), 'validation_rmse': rmse(yval, rf.predict(xval))
            })

    depths = pd.DataFrame(depth_rows)
    regs = pd.DataFrame(reg_rows)
    boosts = pd.DataFrame(boost_rows)
    bags = pd.DataFrame(bag_rows)

    depths.to_csv(result_dir / 'bias_variance_depth.csv', index=False)
    regs.to_csv(result_dir / 'tree_regularization.csv', index=False)
    boosts.to_csv(result_dir / 'boosting_regularization.csv', index=False)
    bags.to_csv(result_dir / 'bagging_estimators.csv', index=False)

    # Generate Figures
    for scenario in ('early', 'grade_informed'):
        part = depths[depths.scenario == scenario]
        plot = part.groupby('depth', sort=False)[['train_rmse', 'validation_rmse']].mean()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(plot.index, plot.train_rmse, marker='o', linewidth=2, label='Training RMSE (Bias proxy)')
        ax.plot(plot.index, plot.validation_rmse, marker='s', linewidth=2, label='Validation RMSE (Generalization)')
        ax.set_title(f'Bias–Variance Tradeoff: Decision Tree Complexity ({scenario})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Maximum Tree Depth (Complexity)', fontsize=10)
        ax.set_ylabel('RMSE (Lower is Better)', fontsize=10)
        ax.legend(frameon=True)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / f'depth_{scenario}.png', dpi=150)
        plt.close(fig)

        r = regs[regs.scenario == scenario].groupby('regularization', sort=False)[['train_rmse', 'validation_rmse']].mean()
        fig, ax = plt.subplots(figsize=(10, 5.2))
        xs = np.arange(len(r))
        ax.bar(xs - 0.17, r.train_rmse, width=0.34, label='Training RMSE', color='#4a90e2')
        ax.bar(xs + 0.17, r.validation_rmse, width=0.34, label='Validation RMSE', color='#50e3c2')
        ax.set_xticks(xs)
        ax.set_xticklabels(r.index, rotation=18, ha='right', fontsize=9)
        ax.set_ylabel('RMSE (Lower is Better)', fontsize=10)
        ax.set_title(f'Decision Tree Regularization Comparison ({scenario})', fontsize=12, fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(alpha=0.2, axis='y')
        fig.tight_layout()
        fig.savefig(figures / f'tree_regularization_{scenario}.png', dpi=150)
        plt.close(fig)

        # Boosting comparison figure
        b = boosts[boosts.scenario == scenario]
        fig, ax = plt.subplots(figsize=(10, 5.2))
        b_plot = b.set_index(['model', 'configuration'])[['train_rmse', 'validation_rmse']]
        labels = [f'{m}\n{c}' for m, c in b_plot.index]
        xs = np.arange(len(labels))
        ax.bar(xs - 0.17, b_plot.train_rmse, width=0.34, label='Training RMSE', color='#f5a623')
        ax.bar(xs + 0.17, b_plot.validation_rmse, width=0.34, label='Validation RMSE', color='#bd10e0')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_title(f'Boosting Regularization (XGBoost & LightGBM) ({scenario})', fontsize=12, fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(alpha=0.2, axis='y')
        fig.tight_layout()
        fig.savefig(figures / f'boosting_regularization_{scenario}.png', dpi=150)
        plt.close(fig)

    print('Saved bias_variance_depth.csv, tree_regularization.csv, boosting_regularization.csv, bagging_estimators.csv and figures.')
    return depths, regs, boosts, bags


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run ML syllabus experiments (Bias-Variance, Regularization, Bagging)')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--data', type=Path, default=None, help='Explicit dataset path')
    group.add_argument('--demo', action='store_true', help='Explicitly run on synthetic demo data')
    ap.add_argument('--repeats', type=int, default=4, help='Repeated validation splits (default: 4)')
    args = ap.parse_args()

    if args.demo:
        dataset_path = DEMO_DIR / 'student-demo-SYNTHETIC.csv'
        if not dataset_path.exists():
            from .data import make_demo_dataset
            dataset_path = make_demo_dataset()
    elif args.data:
        dataset_path = args.data
        if not dataset_path.exists():
            ap.error(f'Dataset not found: {dataset_path}')
    else:
        dataset_path = RAW / 'student-mat.csv'
        if not dataset_path.exists():
            ap.error(
                f'Real dataset not found at {dataset_path}.\n'
                'The experiment pipeline does NOT silently fall back to synthetic data.\n'
                'To download real UCI data, run: python -m src.data --download\n'
                'To explicitly run on synthetic demo data, specify: --demo'
            )

    run_experiments(dataset_path, repeats=args.repeats)
