"""Read-only verification for the user-provided Google Colab export."""
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np

from src.colab_artifacts import legacy_nn_probabilities, load_legacy_colab_exports
from src.data import ROOT, feature_frame, load_data


def check_colab_exports():
    artifacts = load_legacy_colab_exports()
    if not artifacts.compatible:
        raise RuntimeError('; '.join(artifacts.messages))
    data = load_data(ROOT / 'data/raw/student-mat.csv')
    failures = []
    for entry in artifacts.manifest['models']:
        scenario = 'grade_informed' if entry['scenario'] == 'informed' else entry['scenario']
        try:
            x = feature_frame(data, scenario).iloc[:3]
            assert list(x.columns) == entry['input_columns']
            with warnings.catch_warnings():
                warnings.simplefilter('error', category=__import__('sklearn.exceptions', fromlist=['InconsistentVersionWarning']).InconsistentVersionWarning)
                model = joblib.load(ROOT / 'colab_exports' / entry['artifact'])
                prediction = np.asarray(model.predict(x))
                assert prediction.shape == (3,) and np.isfinite(prediction).all()
                if '_risk_' in entry['artifact']:
                    probability = np.asarray(model.predict_proba(x))
                    assert probability.shape == (3, 2) and np.isfinite(probability).all()
                    assert np.allclose(probability.sum(axis=1), 1)
            print(f"PASS {entry['artifact']}")
        except Exception as exc:
            failures.append(f"{entry['artifact']}: {type(exc).__name__}: {exc}")
    try:
        x = feature_frame(data, 'early').iloc[artifacts.manifest['test_indices']]
        probability, model, _ = legacy_nn_probabilities(x)
        assert len(probability) == 79 and np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
        print(f"PASS {artifacts.manifest['numpy_nn']['artifact']} ({model['optimizer']}, {model['epochs']} epochs)")
    except Exception as exc:
        failures.append(f"numpy_neural_network.joblib: {type(exc).__name__}: {exc}")
    for error in failures:
        print(f'FAIL {error}', file=sys.stderr)
    print(f"{21 - len(failures)}/21 Colab artifacts loaded and inferred; declared 316/79 UCI split verified.")
    return not failures


if __name__ == '__main__':
    raise SystemExit(0 if check_colab_exports() else 1)
