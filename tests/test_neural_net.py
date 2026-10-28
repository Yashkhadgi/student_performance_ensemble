"""Unit tests for the from-scratch NumPy Neural Network."""
from pathlib import Path
import numpy as np
import pytest

from src.data import ROOT, load_data, feature_frame
from src.models import make_preprocessor
from src.neural_net import (
    NumPyNeuralNetwork,
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    binary_cross_entropy,
)


def test_activation_functions_and_derivatives():
    z = np.array([-5.0, 0.0, 5.0])
    s = sigmoid(z)
    assert np.all((0.0 <= s) & (s <= 1.0))
    assert pytest.approx(s[1], rel=1e-5) == 0.5
    ds = sigmoid_derivative(s)
    assert pytest.approx(ds[1], rel=1e-5) == 0.25

    r = relu(z)
    assert np.array_equal(r, np.array([0.0, 0.0, 5.0]))
    dr = relu_derivative(z)
    assert np.array_equal(dr, np.array([0.0, 0.0, 1.0]))


def test_binary_cross_entropy():
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([0.999, 0.001, 0.999, 0.001])
    y_pred_bad = np.array([0.1, 0.9, 0.1, 0.9])

    loss_perfect = binary_cross_entropy(y_true, y_pred_perfect)
    loss_bad = binary_cross_entropy(y_true, y_pred_bad)
    assert loss_perfect < 0.01
    assert loss_bad > 1.5


def test_neural_network_gradient_check():
    """Verify analytical backpropagation math matches numerical finite differences (< 1e-6)."""
    np.random.seed(42)
    X_toy = np.random.randn(5, 6)
    y_toy = np.random.randint(0, 2, size=(5, 1))

    net = NumPyNeuralNetwork(input_dim=6, hidden_dim=4, output_dim=1, random_state=42)
    passed, max_err, errors = net.check_gradients(X_toy, y_toy, eps=1e-7)

    assert passed, f"Gradient check failed! Max relative error = {max_err}"
    assert max_err < 1e-6, f"Gradient relative error too high: {max_err}"


def test_neural_network_fit_sgd_and_adam():
    """Verify that both SGD and Adam reduce BCE loss over iterations."""
    np.random.seed(42)
    X = np.random.randn(50, 8)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Test Adam
    net_adam = NumPyNeuralNetwork(input_dim=8, hidden_dim=8, random_state=42)
    hist_adam = net_adam.fit(X, y, epochs=25, lr=0.05, optimizer='adam')
    assert hist_adam['train_loss'][-1] < hist_adam['train_loss'][0]

    # Test SGD
    net_sgd = NumPyNeuralNetwork(input_dim=8, hidden_dim=8, random_state=42)
    hist_sgd = net_sgd.fit(X, y, epochs=25, lr=0.05, optimizer='sgd')
    assert hist_sgd['train_loss'][-1] < hist_sgd['train_loss'][0]


def test_neural_network_on_uci_dataset():
    """Verify training and prediction on real UCI mathematics features."""
    df = load_data(ROOT / 'data' / 'raw' / 'student-mat.csv')
    X_raw = feature_frame(df, 'early')
    y = (df.G3 < 10).astype(int)

    preprocessor = make_preprocessor(X_raw)
    X_trans = preprocessor.fit_transform(X_raw)
    X_np = np.asarray(X_trans, dtype=float)
    y_np = y.to_numpy()

    net = NumPyNeuralNetwork(input_dim=X_np.shape[1], hidden_dim=16, random_state=42)
    hist = net.fit(X_np[:316], y_np[:316], X_val=X_np[316:], y_val=y_np[316:], epochs=15, lr=0.02, optimizer='adam')

    probs = net.predict_proba(X_np[316:])
    preds = net.predict(X_np[316:])

    assert len(probs) == len(X_np[316:])
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(np.unique(preds)).issubset({0, 1})
    assert hist['train_loss'][-1] < hist['train_loss'][0]
