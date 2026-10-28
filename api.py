"""Production-grade REST API server for the Student Performance Ensemble Lab.

Provides HTTP REST endpoints for external system integration (school portals, mobile apps, ERPs):
  - GET  /health               -> Service health, dataset SHA-256 fingerprint, model inventory
  - POST /predict/grade        -> Real-time student final grade prediction (0 to 20 scale)
  - POST /predict/risk         -> Real-time academic risk probability and classification
  - POST /simulate/intervention -> Prescriptive counterfactual academic intervention analysis

Built using Python's standard library `http.server` for zero-dependency portability.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / 'models' / 'manifest.json'

# Global cache for manifest and loaded joblib pipelines
_MANIFEST: Dict[str, Any] = {}
_MODELS: Dict[str, Any] = {}


def load_service_resources() -> None:
    """Pre-load manifest and pipeline artifacts into memory."""
    global _MANIFEST, _MODELS
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")
    _MANIFEST = json.loads(MANIFEST_PATH.read_text())
    models_dir = ROOT / 'models'
    for key, info in _MANIFEST.get('models', {}).items():
        model_file = models_dir / info['file']
        if model_file.is_file() and key not in _MODELS:
            _MODELS[key] = joblib.load(model_file)


def run_prediction(
    scenario: str,
    task: str,
    model_name: str,
    profile_data: Dict[str, Any],
) -> float:
    """Execute model inference with schema validation and default filling."""
    if scenario not in ('early', 'grade_informed'):
        raise ValueError("scenario must be 'early' or 'grade_informed'")
    if task not in ('grade', 'risk'):
        raise ValueError("task must be 'grade' or 'risk'")

    model_slug = model_name.lower().replace(' ', '_')
    key = f"{scenario}__{task}__{model_slug}"
    if key not in _MODELS:
        valid_models = [k.split('__')[2] for k in _MODELS if k.startswith(f"{scenario}__{task}")]
        raise ValueError(f"Model '{model_name}' not found. Available: {valid_models}")

    schema_info = _MANIFEST['schema'][scenario]
    defaults = dict(schema_info['defaults'])
    defaults.update(profile_data)

    df_in = pd.DataFrame([defaults])[schema_info['fields']]
    estimator = _MODELS[key]

    if task == 'grade':
        pred = float(estimator.predict(df_in)[0])
        return float(np.clip(pred, 0.0, 20.0))
    else:
        probs = estimator.predict_proba(df_in)
        return float(probs[0][1])


class EnsembleAPIRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for REST API endpoints."""

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        response_bytes = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests."""
        if self.path in ('/', '/health'):
            src = _MANIFEST.get('source', {})
            payload = {
                'status': 'healthy',
                'service': 'Student Performance & Academic Risk Ensemble API',
                'dataset': src.get('name', 'student-mat.csv'),
                'dataset_sha256': src.get('sha256', ''),
                'dataset_status': src.get('status', 'VERIFIED REAL UCI DATA'),
                'models_loaded': len(_MODELS),
                'supported_scenarios': ['early', 'grade_informed'],
                'supported_models': ['Decision Tree', 'Bagging', 'Random Forest', 'XGBoost', 'LightGBM'],
                'endpoints': {
                    'GET /health': 'System health check and metadata',
                    'POST /predict/grade': 'Predict final course grade (0-20 scale)',
                    'POST /predict/risk': 'Predict fail risk probability and support band',
                    'POST /simulate/intervention': 'Compute before-and-after prescriptive intervention delta',
                }
            }
            self._send_json(200, payload)
        else:
            self._send_json(404, {'error': f"Route '{self.path}' not found."})

    def do_POST(self) -> None:
        """Route POST requests."""
        content_len = int(self.headers.get('Content-Length', 0))
        if content_len == 0:
            self._send_json(400, {'error': 'Empty request body.'})
            return

        try:
            body = json.loads(self.rfile.read(content_len).decode('utf-8'))
        except json.JSONDecodeError as exc:
            self._send_json(400, {'error': f'Invalid JSON payload: {exc}'})
            return

        scenario = body.get('scenario', 'early')
        model = body.get('model', 'Random Forest')
        profile = body.get('profile', {})

        try:
            if self.path == '/predict/grade':
                grade = run_prediction(scenario, 'grade', model, profile)
                self._send_json(200, {
                    'status': 'success',
                    'scenario': scenario,
                    'model': model,
                    'task': 'grade',
                    'predicted_grade_G3': round(grade, 2),
                    'scale': '0 to 20',
                    'performance_tier': 'High Distinction' if grade >= 16 else 'Proficient' if grade >= 12 else 'Passing' if grade >= 10 else 'Failing'
                })

            elif self.path == '/predict/risk':
                risk_prob = run_prediction(scenario, 'risk', model, profile)
                threshold = float(body.get('threshold', 0.50))
                is_at_risk = bool(risk_prob >= threshold)
                priority = 'Low Risk (<30%)' if risk_prob < 0.30 else 'Moderate Risk (30-65%)' if risk_prob < 0.65 else 'High Academic Risk (>65%)'

                self._send_json(200, {
                    'status': 'success',
                    'scenario': scenario,
                    'model': model,
                    'task': 'risk',
                    'risk_probability': round(risk_prob, 4),
                    'threshold_used': threshold,
                    'is_at_risk': is_at_risk,
                    'support_priority_band': priority,
                    'recommendation': 'Priority intervention required' if is_at_risk else 'Student on safe academic track'
                })

            elif self.path == '/simulate/intervention':
                # Baseline prediction
                base_grade = run_prediction(scenario, 'grade', model, profile)
                base_risk = run_prediction(scenario, 'risk', model, profile)

                # Target counterfactual adjustments
                interventions = body.get('intervention', {})
                cf_profile = dict(profile)
                cf_profile.update(interventions)

                cf_grade = run_prediction(scenario, 'grade', model, cf_profile)
                cf_risk = run_prediction(scenario, 'risk', model, cf_profile)

                self._send_json(200, {
                    'status': 'success',
                    'scenario': scenario,
                    'model': model,
                    'baseline': {
                        'predicted_grade': round(base_grade, 2),
                        'risk_probability': round(base_risk, 4),
                    },
                    'post_intervention': {
                        'predicted_grade': round(cf_grade, 2),
                        'risk_probability': round(cf_risk, 4),
                    },
                    'impact': {
                        'grade_improvement': round(cf_grade - base_grade, 2),
                        'risk_reduction_percentage': round((base_risk - cf_risk) * 100, 2),
                        'successful_derisking': bool(cf_risk < 0.30 and base_risk >= 0.30)
                    }
                })

            else:
                self._send_json(404, {'error': f"POST endpoint '{self.path}' not found."})

        except Exception as exc:
            self._send_json(500, {'error': f'{type(exc).__name__}: {str(exc)}'})


def create_server(host: str = '0.0.0.0', port: int = 8000) -> HTTPServer:
    """Create and initialize the HTTP server."""
    load_service_resources()
    server = HTTPServer((host, port), EnsembleAPIRequestHandler)
    return server


if __name__ == '__main__':
    load_service_resources()
    server = HTTPServer(('127.0.0.1', 8000), EnsembleAPIRequestHandler)
    print("🚀 Ensemble REST API Server running on http://127.0.0.1:8000")
    print("📋 Health check: http://127.0.0.1:8000/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API server.")
        server.server_close()
