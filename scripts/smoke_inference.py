"""Read-only smoke check of all bundled, trusted historical pipelines. Never fits models."""
from pathlib import Path
import json
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from src.data import ROOT, UCI_MAT_SHA256, compute_sha256, load_data, feature_frame


def check_models():
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    path = ROOT / 'data/raw/student-mat.csv'
    assert compute_sha256(path) == UCI_MAT_SHA256, 'Bundled UCI dataset hash mismatch'
    data = load_data(path)
    assert data.shape == (395, 33)
    assert len(manifest['models']) == 20, 'Expected 20 historical pipelines'
    failures = []
    for key, entry in manifest['models'].items():
        try:
            x = feature_frame(data, entry['scenario']).iloc[:3]
            assert list(x.columns) == manifest['schema'][entry['scenario']]['fields']
            with warnings.catch_warnings():
                warnings.simplefilter('error', category=__import__('sklearn.exceptions', fromlist=['InconsistentVersionWarning']).InconsistentVersionWarning)
                model = joblib.load(ROOT / 'models' / entry['file'])
                assert list(model.feature_names_in_) == list(x.columns)
                prediction = np.asarray(model.predict(x))
                assert prediction.shape == (3,) and np.isfinite(prediction).all()
                if entry['task'] == 'risk':
                    assert set(prediction).issubset({0, 1})
                    probability = model.predict_proba(x)
                    assert probability.shape == (3, 2)
                    assert np.isfinite(probability).all()
                    assert ((probability >= 0) & (probability <= 1)).all()
                    assert np.allclose(probability.sum(axis=1), 1)
                    assert list(model.classes_) == [0, 1]
            print(f'PASS {key}')
        except Exception as exc:
            failures.append(f'{key}: {type(exc).__name__}: {exc}')
    for failure in failures:
        print(f'FAIL {failure}', file=sys.stderr)
    print(f'{20 - len(failures)}/20 historical pipelines loaded and predicted; UCI shape/hash verified.')
    return not failures


if __name__ == '__main__':
    raise SystemExit(0 if check_models() else 1)
