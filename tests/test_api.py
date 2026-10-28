"""Unit tests for the REST API endpoints and prediction service."""
import json
import threading
from urllib.request import urlopen, Request
import pytest

from api import create_server, run_prediction, load_service_resources


@pytest.fixture(scope="module")
def live_api_server():
    """Start API server on a test port in a background thread."""
    load_service_resources()
    server = create_server(host='127.0.0.1', port=8999)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield 'http://127.0.0.1:8999'
    server.shutdown()
    server.server_close()


def test_api_prediction_functions():
    """Test direct Python prediction wrapper."""
    load_service_resources()
    grade = run_prediction('early', 'grade', 'Random Forest', {'studytime': 3, 'absences': 2})
    assert 0.0 <= grade <= 20.0

    prob = run_prediction('early', 'risk', 'LightGBM', {'studytime': 1, 'absences': 16})
    assert 0.0 <= prob <= 1.0


def test_api_health_endpoint(live_api_server):
    """Test GET /health."""
    req = Request(f"{live_api_server}/health")
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data['status'] == 'healthy'
        assert data['models_loaded'] == 20
        assert 'student-mat.csv' in data['dataset']


def test_api_predict_grade_endpoint(live_api_server):
    """Test POST /predict/grade."""
    payload = {
        'scenario': 'early',
        'model': 'Random Forest',
        'profile': {'studytime': 3, 'absences': 4, 'failures': 0}
    }
    req = Request(
        f"{live_api_server}/predict/grade",
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data['status'] == 'success'
        assert 0.0 <= data['predicted_grade_G3'] <= 20.0


def test_api_predict_risk_endpoint(live_api_server):
    """Test POST /predict/risk with custom threshold."""
    payload = {
        'scenario': 'early',
        'model': 'XGBoost',
        'threshold': 0.35,
        'profile': {'studytime': 1, 'absences': 14, 'failures': 2}
    }
    req = Request(
        f"{live_api_server}/predict/risk",
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data['status'] == 'success'
        assert 0.0 <= data['risk_probability'] <= 1.0
        assert data['threshold_used'] == 0.35


def test_api_simulate_intervention_endpoint(live_api_server):
    """Test POST /simulate/intervention counterfactual analysis."""
    payload = {
        'scenario': 'early',
        'model': 'LightGBM',
        'profile': {'studytime': 1, 'absences': 18, 'failures': 1},
        'intervention': {'studytime': 3, 'absences': 4, 'schoolsup': 'yes'}
    }
    req = Request(
        f"{live_api_server}/simulate/intervention",
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data['status'] == 'success'
        assert 'impact' in data
        assert 'grade_improvement' in data['impact']
        assert 'risk_reduction_percentage' in data['impact']
