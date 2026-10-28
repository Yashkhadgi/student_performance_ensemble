import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import (
    ROOT, RAW, DEMO_DIR,
    UCI_MAT_SHA256, UCI_POR_SHA256,
    STATUS_VERIFIED_REAL, STATUS_SYNTHETIC, STATUS_UNVERIFIED,
    make_demo_dataset, load_data, feature_frame, source_info, compute_sha256
)
from src.models import make_pipeline, MODELS
from src.train import evaluate


@pytest.fixture(scope='module')
def demo_frame(tmp_path_factory):
    file = make_demo_dataset(tmp_path_factory.mktemp('demo') / 'sample.csv', n=120, seed=12)
    return load_data(file)


@pytest.fixture(scope='module')
def real_mat_path():
    path = RAW / 'student-mat.csv'
    if not path.is_file():
        pytest.skip("Real student-mat.csv not downloaded; run python -m src.data --download")
    return path


# --- Provenance & Data Ingestion Tests ---

def test_real_data_provenance_and_hash(real_mat_path):
    sha = compute_sha256(real_mat_path)
    assert sha == UCI_MAT_SHA256, f"Expected UCI mat hash {UCI_MAT_SHA256}, got {sha}"
    info = source_info(real_mat_path)
    assert info['status'] == STATUS_VERIFIED_REAL
    assert info['verified'] is True
    assert info['synthetic'] is False
    assert 'mat' in info['subject']


def test_synthetic_data_detection(tmp_path):
    demo_file = make_demo_dataset(tmp_path / 'demo_test.csv', n=80, seed=99)
    info = source_info(demo_file)
    assert info['status'] == STATUS_SYNTHETIC
    assert info['synthetic'] is True
    assert info['verified'] is False


def test_unverified_custom_data_detection(tmp_path):
    custom_dir = tmp_path / 'custom_dir'
    custom_dir.mkdir()
    custom_file = custom_dir / 'custom_students.csv'
    # Copy from raw student-mat.csv and modify values so it's a realistic custom dataset without UCI hash
    mat_file = RAW / 'student-mat.csv'
    df_data = pd.read_csv(mat_file, sep=';')
    df_data.loc[0, 'absences'] = 91
    df_data.to_csv(custom_file, sep=';', index=False)

    info = source_info(custom_file)
    assert info['status'] == STATUS_UNVERIFIED
    assert info['verified'] is False
    assert info['synthetic'] is False


def test_schema_validation_missing_columns(tmp_path):
    bad_file = tmp_path / 'bad_schema.csv'
    bad_file.write_text("school;sex;age;G3\nGP;F;17;12\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_data(bad_file)


def test_schema_validation_invalid_grade_bounds(tmp_path):
    df_file = make_demo_dataset(tmp_path / 'temp_bounds.csv', n=60, seed=1)
    df = pd.read_csv(df_file, sep=';')
    df.loc[0, 'G3'] = 25  # Invalid: > 20
    bad_file = tmp_path / 'bad_grades.csv'
    df.to_csv(bad_file, sep=';', index=False)
    with pytest.raises(ValueError, match="must contain grades between 0 and 20"):
        load_data(bad_file)


def test_nonexistent_dataset_raises_error():
    with pytest.raises(FileNotFoundError):
        load_data("data/raw/definitely_nonexistent_dataset_12345.csv")


# --- Leakage & Feature Framing Tests ---

def test_no_target_leakage(demo_frame):
    early = feature_frame(demo_frame, 'early')
    late = feature_frame(demo_frame, 'grade_informed')
    assert not {'G1', 'G2', 'G3'}.intersection(early.columns)
    assert 'G3' not in late
    assert {'G1', 'G2'}.issubset(late.columns)


def test_grade_bounds_and_risk_balance(demo_frame):
    assert demo_frame.G3.between(0, 20).all()
    assert demo_frame.G3.lt(10).nunique() == 2


# --- Model Pipelines & Inference Tests ---

@pytest.mark.parametrize('name', MODELS)
@pytest.mark.parametrize('scenario', ['early', 'grade_informed'])
def test_all_five_regressors(demo_frame, name, scenario):
    x = feature_frame(demo_frame, scenario)
    y = demo_frame.G3
    model = make_pipeline(name, 'grade', x, {'n_estimators': 5} if name != 'Decision Tree' else {})
    model.fit(x, y)
    preds = model.predict(x.iloc[:5])
    assert len(preds) == 5
    # Predictions should be numeric and reasonably bounded
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize('name', MODELS)
@pytest.mark.parametrize('scenario', ['early', 'grade_informed'])
def test_all_five_classifiers(demo_frame, name, scenario):
    x = feature_frame(demo_frame, scenario)
    y = (demo_frame.G3 < 10).astype(int)
    model = make_pipeline(name, 'risk', x, {'n_estimators': 5} if name != 'Decision Tree' else {})
    model.fit(x, y)
    probs = model.predict_proba(x.iloc[:5])
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_unknown_categorical_handled(demo_frame):
    x = feature_frame(demo_frame, 'early')
    y = demo_frame.G3
    model = make_pipeline('Decision Tree', 'grade', x).fit(x, y)
    altered = x.iloc[:2].copy()
    altered['Mjob'] = 'unseen_exotic_job'
    altered['school'] = 'UNKNOWN_SCHOOL'
    preds = model.predict(altered)
    assert len(preds) == 2
    assert np.all(np.isfinite(preds))


def test_model_serialization_and_roundtrip(demo_frame, tmp_path):
    x = feature_frame(demo_frame, 'early')
    y = demo_frame.G3
    model = make_pipeline('Random Forest', 'grade', x, {'n_estimators': 10})
    model.fit(x, y)
    preds_before = model.predict(x.iloc[:10])

    saved_file = tmp_path / 'rf_test.joblib'
    joblib.dump(model, saved_file)
    loaded_model = joblib.load(saved_file)
    preds_after = loaded_model.predict(x.iloc[:10])

    assert np.allclose(preds_before, preds_after)


# --- Evaluation Metrics & Diagnostic Tests ---

def test_evaluation_metrics_regression():
    y_true = np.array([10.0, 15.0, 8.0, 12.0])
    y_pred = np.array([11.0, 14.0, 8.5, 11.5])
    metrics = evaluate('grade', y_true, y_pred)
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'max_error' in metrics
    assert metrics['max_error'] == 1.0


def test_evaluation_metrics_classification():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.6, 0.85])
    metrics = evaluate('risk', y_true, y_pred, y_prob)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'brier_score' in metrics


# --- Manifest & Deployment Integrity Tests ---

def test_manifest_structure_and_coverage():
    manifest_path = ROOT / 'models' / 'manifest.json'
    assert manifest_path.is_file(), "models/manifest.json must exist"
    data = json.loads(manifest_path.read_text())
    assert 'source' in data
    assert 'trained_at' in data or 'models' in data
    assert 'class_balance' in data
    assert len(data['models']) == 20, f"Expected 20 trained models in manifest, got {len(data['models'])}"

    for label, m in data['models'].items():
        assert 'scenario' in m
        assert 'task' in m
        assert 'name' in m
        assert 'file' in m
        assert (ROOT / 'models' / m['file']).is_file(), f"Model file {m['file']} missing on disk"


def test_results_artifacts_exist_and_non_empty():
    res_dir = ROOT / 'results'
    required_files = [
        'model_metrics.csv',
        'cv_candidates.csv',
        'test_predictions.csv',
        'feature_importance.csv',
        'bias_variance_depth.csv',
        'tree_regularization.csv',
        'boosting_regularization.csv',
        'bagging_estimators.csv'
    ]
    for rf in required_files:
        p = res_dir / rf
        assert p.is_file(), f"Missing required result file: {rf}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Result file {rf} is empty"


def test_test_predictions_residuals():
    preds_file = ROOT / 'results' / 'test_predictions.csv'
    assert preds_file.is_file()
    df = pd.read_csv(preds_file)
    assert 'residual' in df.columns
    reg_df = df[df.task == 'grade']
    if not reg_df.empty:
        diff = reg_df['actual'] - reg_df['predicted']
        assert np.allclose(reg_df['residual'], diff, atol=1e-4)
