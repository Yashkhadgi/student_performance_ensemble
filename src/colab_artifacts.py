"""Read-only contract for future user-exported Colab evidence; never unpickles files.

Schema validation establishes compatibility, not scientific authenticity. Export provenance
and split membership must still be reviewed with the student before final integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from .data import ROOT, EXPECTED_COLUMNS, SCENARIOS, UCI_MAT_SHA256

SCHEMAS = {
    'nn_history.csv': ['scenario', 'optimizer', 'epoch', 'train_loss', 'val_loss'],
    'nn_metrics.csv': ['scenario', 'optimizer', 'task', 'split', 'metric', 'value'],
    'nn_predictions.csv': ['scenario', 'optimizer', 'row_index', 'actual', 'predicted', 'risk_probability'],
    'ensemble_metrics.csv': ['scenario', 'task', 'model', 'split', 'metric', 'value'],
    'bias_variance.csv': ['scenario', 'complexity', 'train_error', 'validation_error'],
    'regularization_metrics.csv': ['scenario', 'model', 'alpha', 'lambda', 'train_error', 'validation_error'],
}
METRICS = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score', 'mae', 'rmse', 'r2', 'max_error'}


@dataclass
class ColabArtifacts:
    manifest: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    environment_notes: list = field(default_factory=list)
    compatible: bool = False


def validate_manifest(manifest):
    """Reject incompatible labels/features/data; require reproducible held-out membership."""
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be a JSON object')
    if manifest.get('dataset_sha256') != UCI_MAT_SHA256:
        raise ValueError('dataset_sha256 must match bundled student-mat.csv')
    if manifest.get('risk_label') != 'G3 < 10':
        raise ValueError('risk_label must be exactly "G3 < 10" (class 1 = at risk)')
    if not isinstance(manifest.get('seed'), int) or isinstance(manifest['seed'], bool):
        raise ValueError('seed must be an integer')
    for key in ('source', 'trained_at'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f'{key} must identify the training source/time')
    schema = manifest.get('feature_schema', {})
    if not isinstance(schema, dict) or not schema or set(schema) - set(SCENARIOS):
        raise ValueError('feature_schema must map supported scenarios to ordered raw input fields')
    for scenario, fields in schema.items():
        expected = [c for c in EXPECTED_COLUMNS if c not in SCENARIOS[scenario]]
        if fields != expected:
            raise ValueError(f'feature_schema.{scenario} must match exact ordered repo fields; no G3 leakage')
    split = manifest.get('split', {})
    if not isinstance(split, dict) or split.get('name') != 'held_out_test':
        raise ValueError('split.name must be held_out_test')
    indices = split.get('test_row_indices')
    if (not isinstance(indices, list) or not indices or
        any(type(i) is not int or i < 0 or i >= 395 for i in indices) or
        len(indices) != len(set(indices)) or len(indices) >= 395):
        raise ValueError('split.test_row_indices must be unique zero-based UCI row positions, a proper subset of 395 rows')
    packages = manifest.get('packages')
    if not isinstance(packages, dict) or not packages or any(not isinstance(v, str) or not v for v in packages.values()):
        raise ValueError('packages must contain training package versions')
    return manifest


def _numeric(table, columns, nonnegative=True):
    for column in columns:
        values = pd.to_numeric(table[column], errors='raise')
        if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
            raise ValueError(f'{column} must contain finite' + (' nonnegative' if nonnegative else '') + ' numbers')
        table[column] = values


def validate_table(name, table, manifest):
    missing = set(SCHEMAS[name]) - set(table.columns)
    if missing:
        raise ValueError(f'missing columns: {", ".join(sorted(missing))}')
    if table.empty or table[SCHEMAS[name]].isna().any().any():
        raise ValueError('required cells must be nonempty')
    if not set(table.scenario).issubset(manifest['feature_schema']):
        raise ValueError('scenario has no compatible feature_schema')
    if 'optimizer' in table and not set(table.optimizer).issubset({'SGD', 'Adam'}):
        raise ValueError('optimizer must be SGD or Adam')
    if 'task' in table:
        if not set(table.task).issubset({'grade', 'risk'}):
            raise ValueError('task must be grade or risk')
        if name == 'nn_metrics.csv' and not table.task.eq('risk').all():
            raise ValueError('sigmoid/BCE NN contract supports risk classification only')
        if not table.split.eq('held_out_test').all() or not set(table.metric).issubset(METRICS):
            raise ValueError('metrics require split=held_out_test and supported metric names')
        _numeric(table, ['value'], nonnegative=False)
        bounded = table.metric.isin({'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score'})
        if not table.loc[bounded, 'value'].between(0, 1).all():
            raise ValueError('classification metrics must lie in [0, 1]')
        if (table.loc[~table.metric.eq('r2'), 'value'] < 0).any():
            raise ValueError('only r2 may be negative')
        group = 'optimizer' if 'optimizer' in table else 'model'
        if table.duplicated(['scenario', group, 'task', 'split', 'metric']).any():
            raise ValueError('duplicate metric rows')
    if name == 'nn_history.csv':
        _numeric(table, ['epoch', 'train_loss', 'val_loss'])
        if (table.epoch < 1).any() or (table.epoch % 1 != 0).any():
            raise ValueError('epoch must be a positive integer')
        if table.duplicated(['scenario', 'optimizer', 'epoch']).any():
            raise ValueError('duplicate optimizer epochs')
    if name in {'bias_variance.csv', 'regularization_metrics.csv'}:
        _numeric(table, ['train_error', 'validation_error'])
        if name == 'regularization_metrics.csv':
            _numeric(table, ['alpha', 'lambda'])
    if name == 'nn_predictions.csv':
        _numeric(table, ['row_index', 'actual', 'predicted', 'risk_probability'])
        if not set(table.actual).issubset({0, 1}) or not set(table.predicted).issubset({0, 1}):
            raise ValueError('actual/predicted must be binary risk labels')
        if not table.risk_probability.between(0, 1).all():
            raise ValueError('risk_probability must lie in [0, 1]')
        if not set(table.row_index).issubset(manifest['split']['test_row_indices']):
            raise ValueError('predictions must use manifest test_row_indices')
        for _, part in table.groupby(['scenario', 'optimizer']):
            if part.row_index.duplicated().any() or set(part.row_index) != set(manifest['split']['test_row_indices']):
                raise ValueError('each prediction group must cover the exact held-out rows once')
    return table


def load_colab_artifacts(directory=None):
    """Load compatible evidence only. Missing/invalid files have actionable messages.

    Joblib is intentionally deferred until trust, format and inference are reviewed.
    """
    directory = Path(directory) if directory is not None else ROOT / 'colab_exports'
    result = ColabArtifacts()
    path = directory / 'experiment_manifest.json'
    if not path.is_file():
        result.messages.append('Awaiting Colab artifacts: experiment_manifest.json missing. See docs/COLAB_CONTRACT.md.')
        return result
    try:
        result.manifest = validate_manifest(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.messages.append(f'experiment_manifest.json: {exc}. Evidence withheld; fix the manifest.')
        return result
    result.compatible = True
    for package, exported in result.manifest['packages'].items():
        try:
            local = version(package)
        except PackageNotFoundError:
            local = 'not installed'
        if local != exported:
            result.environment_notes.append(f'{package}: Colab {exported}; local {local}. Match versions before trusted model loading.')
    for name in SCHEMAS:
        path = directory / name
        if not path.is_file():
            result.messages.append(f'Awaiting Colab artifacts: {name} missing (optional until exported).')
            continue
        try:
            result.tables[name] = validate_table(name, pd.read_csv(path), result.manifest)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result.messages.append(f'{name}: {exc}. This file is not displayed.')
    return result


LEGACY_METRICS = {
    'early_risk.csv': ['Model', 'CV F1', 'Test Accuracy', 'Test Recall', 'Test F1'],
    'informed_risk.csv': ['Model', 'CV F1', 'Test Accuracy', 'Test Recall', 'Test F1'],
    'grade_regression.csv': ['Scenario', 'Model', 'CV RMSE', 'Test MAE', 'Test RMSE', 'Test R2'],
    'bias_variance.csv': ['Max Depth', 'Train RMSE', 'Validation RMSE', 'Gap'],
    'regularization.csv': ['Model', 'L1 Alpha', 'L2 Lambda', 'Train RMSE', 'Validation RMSE', 'Gap'],
    'optimizer_comparison.csv': ['Optimizer', 'Best Epoch', 'Train Loss', 'Val Loss', 'Val Accuracy', 'Val Recall', 'Val F1'],
    'optimizer_histories.csv': ['epoch', 'sgd_train_loss', 'sgd_val_loss', 'adam_train_loss', 'adam_val_loss'],
}


def load_legacy_colab_exports(directory=None):
    """Validate the student-provided Hack O Week Colab export format.

    This is intentionally separate from the forward-looking `experiment_manifest.json`
    contract. The legacy export has no dataset checksum, so it is accepted only when
    its declared UCI dataset, row count, split, model schemas and local UCI hash agree.
    """
    directory = Path(directory) if directory is not None else ROOT / 'colab_exports'
    result = ColabArtifacts()
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        result.messages.append('No legacy Colab manifest found.')
        return result
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('dataset') != 'UCI Student Performance - Mathematics' or manifest.get('dataset_rows') != 395:
            raise ValueError('declared dataset is not the expected 395-row UCI Mathematics dataset')
        if manifest.get('risk_definition') != 'G3 < 10' or manifest.get('random_state') != 42:
            raise ValueError('risk definition or random_state does not match the declared experiment')
        train, test = manifest.get('train_indices'), manifest.get('test_indices')
        if not isinstance(train, list) or not isinstance(test, list) or len(train) != 316 or len(test) != 79:
            raise ValueError('expected the declared 316/79 train/held-out split')
        if set(train) & set(test) or set(train) | set(test) != set(range(395)):
            raise ValueError('train/test indices must be disjoint and cover all 395 UCI rows')
        models = manifest.get('models', [])
        if len(models) != 20 or manifest.get('model_count') != 21:
            raise ValueError('expected 20 ensemble entries plus one NumPy NN artifact')
        expected = {s: [c for c in EXPECTED_COLUMNS if c not in dropped] for s, dropped in SCENARIOS.items()}
        for entry in models:
            scenario = 'grade_informed' if entry.get('scenario') == 'informed' else entry.get('scenario')
            artifact = directory / entry.get('artifact', '')
            if scenario not in expected or entry.get('input_columns') != expected[scenario] or not artifact.is_file():
                raise ValueError(f"invalid schema or missing model: {entry.get('artifact')}")
        nn = manifest.get('numpy_nn', {})
        if nn.get('scenario') != 'early' or nn.get('optimizer') not in {'SGD', 'Adam'} or not (directory / nn.get('artifact', '')).is_file():
            raise ValueError('missing or invalid NumPy NN declaration')
        result.manifest = manifest
        result.compatible = True
        for name, columns in LEGACY_METRICS.items():
            path = directory / 'metrics' / name
            if not path.is_file():
                result.messages.append(f'Colab metric file missing: metrics/{name}')
                continue
            table = pd.read_csv(path)
            missing = set(columns) - set(table.columns)
            if missing or table.empty:
                result.messages.append(f'metrics/{name}: missing {sorted(missing)} or empty; not displayed.')
            else:
                result.tables[name] = table
        exported = manifest.get('versions', {})
        local_names = {'numpy': 'numpy', 'pandas': 'pandas', 'scikit_learn': 'scikit-learn', 'xgboost': 'xgboost', 'lightgbm': 'lightgbm', 'joblib': 'joblib'}
        for exported_name, package in local_names.items():
            try:
                local = version(package)
            except PackageNotFoundError:
                local = 'not installed'
            if exported.get(exported_name) != local:
                result.environment_notes.append(f'{package}: Colab {exported.get(exported_name)}; local {local}.')
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result.messages.append(f'Legacy Colab manifest rejected: {exc}')
    return result


def load_legacy_nn(directory=None):
    """Load the user-provided NN only after legacy manifest validation."""
    artifacts = load_legacy_colab_exports(directory)
    if not artifacts.compatible:
        raise ValueError('; '.join(artifacts.messages))
    directory = Path(directory) if directory is not None else ROOT / 'colab_exports'
    model = joblib.load(directory / artifacts.manifest['numpy_nn']['artifact'])
    required = {'preprocessor', 'weights', 'input_columns', 'hidden_activation', 'output_activation', 'threshold', 'optimizer', 'epochs'}
    if not required.issubset(model) or model['input_columns'] != [c for c in EXPECTED_COLUMNS if c not in SCENARIOS['early']]:
        raise ValueError('NN artifact has an incompatible schema')
    weights = model['weights']
    if (model['hidden_activation'], model['output_activation'], model['threshold']) != ('relu', 'sigmoid', .5):
        raise ValueError('NN architecture or threshold does not match the declared export')
    if set(weights) != {'W1', 'b1', 'W2', 'b2'} or weights['W1'].shape[1] != 16 or weights['W2'].shape != (16, 1):
        raise ValueError('NN weight shapes are invalid')
    return model, artifacts


def legacy_nn_probabilities(frame, directory=None):
    """Run trusted exported NumPy NN inference using its exported preprocessor."""
    model, artifacts = load_legacy_nn(directory)
    x = frame[model['input_columns']]
    transformed = np.asarray(model['preprocessor'].transform(x), dtype=float)
    weights = model['weights']
    hidden = np.maximum(0, transformed @ weights['W1'] + weights['b1'])
    output = hidden @ weights['W2'] + weights['b2']
    probabilities = 1 / (1 + np.exp(-np.clip(output, -500, 500)))
    probabilities = probabilities.reshape(-1)
    if not np.isfinite(probabilities).all() or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError('NN inference produced invalid probabilities')
    return probabilities, model, artifacts
