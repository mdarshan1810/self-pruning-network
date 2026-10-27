"""Datasets. Spirals is generated locally; no external library is involved."""
import numpy as np


def make_spirals(n_per_class=300, n_classes=3, noise=0.2, seed=0):
    """K interleaved spiral arms in 2D. Returns X (N, 2) float64, y (N,) int."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for k in range(n_classes):
        idx = slice(k * n_per_class, (k + 1) * n_per_class)
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = k
    return X, y


def train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    te, tr = perm[:n_test], perm[n_test:]
    return X[tr], y[tr], X[te], y[te]
