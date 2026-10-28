"""Dataset ingestion, cryptographic validation, deterministic synthetic demo data and leakage-safe features."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
DEMO_DIR = ROOT / 'data' / 'demo'
UCI_URL = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'
DATASET_URL = 'https://archive.ics.uci.edu/dataset/320/student+performance'

# Official UCI Student Performance Archive SHA-256 Hashes
UCI_MAT_SHA256 = 'e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80'
UCI_POR_SHA256 = 'a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f'

STATUS_VERIFIED_REAL = 'VERIFIED REAL UCI DATA'
STATUS_SYNTHETIC = 'SYNTHETIC DEMO DATA'
STATUS_UNVERIFIED = 'UNVERIFIED CUSTOM DATA'

SCENARIOS = {'early': ['G1', 'G2', 'G3'], 'grade_informed': ['G3']}
EXPECTED_COLUMNS = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2', 'G3'
]


def compute_sha256(path: Path | str) -> str:
    """Compute reproducible SHA-256 hash of a file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Cannot compute SHA-256 for non-existent file: {path}')
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_uci(subject='mat', target_dir=RAW, overwrite=False) -> Path:
    """Download the official UCI archive, extract both CSVs, and verify SHA-256 checksums."""
    if subject not in ('mat', 'por'):
        raise ValueError('subject must be mat or por')
    target_dir = Path(target_dir)
    dest = target_dir / f'student-{subject}.csv'
    if dest.exists() and not overwrite:
        # Verify existing file hash
        sha = compute_sha256(dest)
        expected = UCI_MAT_SHA256 if subject == 'mat' else UCI_POR_SHA256
        if sha == expected:
            return dest
    target_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        UCI_URL,
        headers={'User-Agent': 'Mozilla/5.0 (StudentEnsembleLab/1.0; Academic Research Evaluation)'}
    )
    with urllib.request.urlopen(request, timeout=35) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        if 'student.zip' in outer.namelist():
            with zipfile.ZipFile(io.BytesIO(outer.read('student.zip'))) as inner:
                members = {name: inner.read(name) for name in ('student-mat.csv', 'student-por.csv')}
        else:
            members = {name: outer.read(name) for name in ('student-mat.csv', 'student-por.csv')}
    for name, data in members.items():
        out_file = target_dir / name
        out_file.write_bytes(data)

    # Validate extracted file checksums
    mat_sha = compute_sha256(target_dir / 'student-mat.csv')
    por_sha = compute_sha256(target_dir / 'student-por.csv')
    if mat_sha != UCI_MAT_SHA256:
        raise ValueError(f'student-mat.csv SHA-256 mismatch: expected {UCI_MAT_SHA256}, got {mat_sha}')
    if por_sha != UCI_POR_SHA256:
        raise ValueError(f'student-por.csv SHA-256 mismatch: expected {UCI_POR_SHA256}, got {por_sha}')
    return dest


def make_demo_dataset(path=None, n=540, seed=42) -> Path:
    """Generate an explicitly SYNTHETIC demonstration dataset; NEVER use for real academic evidence."""
    path = Path(path or ROOT / 'data' / 'demo' / 'student-demo-SYNTHETIC.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    randchoice = lambda seq: rng.choice(seq, size=n)
    d = pd.DataFrame({
        'school': randchoice(['GP', 'MS']), 'sex': randchoice(['F', 'M']),
        'age': rng.integers(15, 23, n), 'address': randchoice(['U', 'R']),
        'famsize': randchoice(['GT3', 'LE3']), 'Pstatus': randchoice(['A', 'T']),
        'Medu': rng.integers(0, 5, n), 'Fedu': rng.integers(0, 5, n),
        'Mjob': randchoice(['teacher', 'health', 'services', 'at_home', 'other']),
        'Fjob': randchoice(['teacher', 'health', 'services', 'at_home', 'other']),
        'reason': randchoice(['course', 'home', 'reputation', 'other']),
        'guardian': randchoice(['mother', 'father', 'other']),
        'traveltime': rng.integers(1, 5, n), 'studytime': rng.integers(1, 5, n),
        'failures': rng.choice([0, 1, 2, 3], n, p=[.76, .16, .06, .02]),
        'schoolsup': randchoice(['yes', 'no']), 'famsup': randchoice(['yes', 'no']),
        'paid': randchoice(['yes', 'no']), 'activities': randchoice(['yes', 'no']),
        'nursery': randchoice(['yes', 'no']), 'higher': randchoice(['yes', 'no']),
        'internet': randchoice(['yes', 'no']), 'romantic': randchoice(['yes', 'no']),
        'famrel': rng.integers(1, 6, n), 'freetime': rng.integers(1, 6, n),
        'goout': rng.integers(1, 6, n), 'Dalc': rng.integers(1, 6, n),
        'Walc': rng.integers(1, 6, n), 'health': rng.integers(1, 6, n),
        'absences': rng.poisson(5, n)
    })
    baseline = (5 + 1.8 * d['studytime'] - 2.15 * d['failures'] - .075 * d['absences']
                + .45 * d['Medu'] + .4 * (d['higher'] == 'yes') + rng.normal(0, 2.8, n))
    d['G1'] = np.clip(np.rint(baseline), 0, 20).astype(int)
    d['G2'] = np.clip(np.rint(.78 * d['G1'] + 2 + .3 * d['studytime'] + rng.normal(0, 2.4, n)), 0, 20).astype(int)
    d['G3'] = np.clip(np.rint(.25 * d['G1'] + .65 * d['G2'] + .5 * d['studytime']
                           - .45 * d['failures'] + rng.normal(0, 2.3, n)), 0, 20).astype(int)
    d = d[EXPECTED_COLUMNS]
    d.to_csv(path, sep=';', index=False)
    (path.parent / 'SYNTHETIC_DATA_WARNING.txt').write_text(
        'This dataset is SYNTHETIC and intended ONLY for offline unit testing.\n'
        'It contains no real student observations. Download UCI data for scientific experiments.\n'
    )
    return path


def load_data(path: Path | str) -> pd.DataFrame:
    """Load and validate semicolon-separated student performance dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Dataset missing: {path}.\n'
            'To download the official UCI dataset, run: python -m src.data --download\n'
            'To generate synthetic test data, run: python -m src.data --demo'
        )
    df = pd.read_csv(path, sep=';')
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f'Dataset missing required columns: {sorted(missing_cols)}')
    for c in ['G1', 'G2', 'G3']:
        df[c] = pd.to_numeric(df[c], errors='raise')
        if df[c].isna().any() or not df[c].between(0, 20).all():
            raise ValueError(f'{c} must contain grades between 0 and 20, with no missing values.')
    if len(df) < 50:
        raise ValueError(f'Dataset too small ({len(df)} rows); at least 50 observations required.')
    if df['G3'].lt(10).nunique() < 2:
        raise ValueError('Dataset must contain both at-risk (G3 < 10) and not-at-risk (G3 >= 10) students.')
    if df.isna().sum().sum() > 0:
        raise ValueError(f'Dataset contains unexpected missing values in feature columns: {df.isna().sum().to_dict()}')
    return df


def feature_frame(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Return feature DataFrame for a given scenario without target leakage."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Choose scenario from {list(SCENARIOS)}")
    return df.drop(columns=SCENARIOS[scenario], errors='ignore').copy()


def source_info(data_path: Path | str) -> dict:
    """Compute cryptographic provenance and classify dataset tier."""
    path = Path(data_path)
    if not path.is_file():
        return {
            'status': 'DATASET MISSING',
            'verified': False,
            'synthetic': False,
            'sha256': '',
            'sha256_short': '',
            'path': str(path),
            'name': path.name,
            'subject': 'unknown',
            'uci_url': DATASET_URL
        }
    sha = compute_sha256(path)
    if sha == UCI_MAT_SHA256:
        status = STATUS_VERIFIED_REAL
        subject = 'mat (Mathematics)'
        verified = True
        synthetic = False
        description = 'Official UCI Student Performance Mathematics dataset, verified by exact SHA-256 cryptographic match.'
    elif sha == UCI_POR_SHA256:
        status = STATUS_VERIFIED_REAL
        subject = 'por (Portuguese Language)'
        verified = True
        synthetic = False
        description = 'Official UCI Student Performance Portuguese dataset, verified by exact SHA-256 cryptographic match.'
    else:
        # Check if explicitly marked synthetic demo data
        is_synthetic_path = 'synthetic' in path.name.lower() or 'demo' in path.name.lower()
        has_warning_file = (path.parent / 'SYNTHETIC_DATA_WARNING.txt').exists()
        if is_synthetic_path or has_warning_file:
            status = STATUS_SYNTHETIC
            subject = 'synthetic_demo'
            verified = False
            synthetic = True
            description = 'Synthetic demonstration data generated for offline testing; not authentic academic observations.'
        else:
            status = STATUS_UNVERIFIED
            subject = 'custom_unverified'
            verified = False
            synthetic = False
            description = 'User-supplied custom CSV matching the Student Performance schema, but not matching official UCI release checksums.'

    return {
        'status': status,
        'verified': verified,
        'synthetic': synthetic,
        'subject': subject,
        'sha256': sha,
        'sha256_short': sha[:12],
        'path': str(path.resolve()),
        'name': path.name,
        'description': description,
        'uci_url': DATASET_URL,
        'citation': 'Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T'
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Fetch UCI real data or create marked synthetic demo data')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--download', action='store_true', help='Download and verify real UCI data (internet required)')
    group.add_argument('--demo', action='store_true', help='Create SYNTHETIC offline test data')
    group.add_argument('--verify', type=Path, help='Verify provenance of a specific dataset file')
    a = ap.parse_args()
    if a.demo:
        created = make_demo_dataset()
        info = source_info(created)
        print(f"Created synthetic demonstration: {created}")
        print(f"Status: {info['status']} (SHA-256: {info['sha256_short']})")
    elif a.verify:
        info = source_info(a.verify)
        print(f"Dataset: {a.verify}")
        print(f"Status: {info['status']}")
        print(f"SHA-256: {info['sha256']}")
        print(f"Description: {info['description']}")
    else:
        try:
            dest = fetch_uci()
            info = source_info(dest)
            print(f"Downloaded & verified official UCI dataset: {dest}")
            print(f"Status: {info['status']}")
            print(f"SHA-256: {info['sha256']}")
        except Exception as e:
            print(f"Download or verification unsuccessful: {e}")
            print("Manually place authentic student-mat.csv in data/raw/.")
            raise SystemExit(1)
