import json

import pandas as pd

from src.colab_artifacts import legacy_nn_probabilities, load_colab_artifacts, load_legacy_colab_exports
from src.data import ROOT, SCENARIOS, UCI_MAT_SHA256, EXPECTED_COLUMNS, feature_frame, load_data


def manifest():
    return {
        'dataset_sha256': UCI_MAT_SHA256,
        'risk_label': 'G3 < 10',
        'seed': 42,
        'source': 'Google Colab verified export',
        'trained_at': '2026-10-21T08:00:00Z',
        'feature_schema': {s: [c for c in EXPECTED_COLUMNS if c not in dropped] for s, dropped in SCENARIOS.items()},
        'split': {'name': 'held_out_test', 'test_row_indices': [1, 3, 5]},
        'packages': {'numpy': '2.0.2'},
    }


def test_missing_colab_folder_is_a_clear_pending_state(tmp_path):
    artifact = load_colab_artifacts(tmp_path)
    assert artifact.compatible is False
    assert 'Awaiting Colab artifacts' in artifact.messages[0]
    assert not artifact.tables


def test_valid_fixture_history_and_metrics_are_loaded(tmp_path):
    (tmp_path / 'experiment_manifest.json').write_text(json.dumps(manifest()))
    pd.DataFrame([
        {'scenario': 'early', 'optimizer': 'SGD', 'epoch': 1, 'train_loss': .7, 'val_loss': .8},
        {'scenario': 'early', 'optimizer': 'Adam', 'epoch': 1, 'train_loss': .6, 'val_loss': .7},
    ]).to_csv(tmp_path / 'nn_history.csv', index=False)
    pd.DataFrame([
        {'scenario': 'early', 'optimizer': 'SGD', 'task': 'risk', 'split': 'held_out_test', 'metric': 'f1', 'value': .5},
        {'scenario': 'early', 'optimizer': 'Adam', 'task': 'risk', 'split': 'held_out_test', 'metric': 'f1', 'value': .6},
    ]).to_csv(tmp_path / 'nn_metrics.csv', index=False)
    artifact = load_colab_artifacts(tmp_path)
    assert artifact.compatible is True
    assert set(artifact.tables) == {'nn_history.csv', 'nn_metrics.csv'}


def test_incompatible_schema_does_not_publish_fixture(tmp_path):
    bad = manifest()
    bad['feature_schema']['early'] = ['G3']
    (tmp_path / 'experiment_manifest.json').write_text(json.dumps(bad))
    artifact = load_colab_artifacts(tmp_path)
    assert artifact.compatible is False
    assert not artifact.tables


def test_user_colab_export_manifest_and_nn_inference():
    artifact = load_legacy_colab_exports(ROOT / 'colab_exports')
    assert artifact.compatible is True
    assert len(artifact.manifest['models']) == 20
    assert {'optimizer_histories.csv', 'optimizer_comparison.csv'}.issubset(artifact.tables)
    data = load_data(ROOT / 'data/raw/student-mat.csv')
    probability, model, _ = legacy_nn_probabilities(feature_frame(data, 'early').iloc[:4], ROOT / 'colab_exports')
    assert model['input_columns'] == list(feature_frame(data, 'early').columns)
    assert probability.shape == (4,)
    assert ((probability >= 0) & (probability <= 1)).all()
