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


def load_mnist(n_train=10000, n_test=2000, subset_seed=0):
    """MNIST via scikit-learn's OpenML fetcher (data loading ONLY; sklearn is never used
    for models, training or gradients). Falls back to sklearn's bundled 8x8 `load_digits`
    if the download fails (no network).

    MNIST: the canonical split is kept (first 60k rows = train, last 10k = test), then a
    fixed, seeded subset of `n_train` / `n_test` is drawn from each so the pure-NumPy
    engine stays fast. Pixels are scaled to [0, 1] (divide by 255).
    Digits fallback: 1797 images, 8x8, values 0..16 scaled to [0, 1]; seeded 80/20 split.

    The subset depends only on `subset_seed`, so every training seed sees the same data.
    Returns Xtr, ytr, Xte, yte, info (dict naming the dataset actually used).
    """
    rng = np.random.default_rng(subset_seed)
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        tr = rng.choice(60000, size=n_train, replace=False)
        te = 60000 + rng.choice(10000, size=n_test, replace=False)
        y = y.astype(np.int64)
        info = dict(dataset="mnist_784 (OpenML)", n_train=n_train, n_test=n_test,
                    n_features=784, subset_seed=subset_seed, scaling="x/255 -> [0,1]")
        return X[tr] / 255.0, y[tr], X[te] / 255.0, y[te], info
    except Exception as e:  # network unavailable etc.
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data.astype(np.float64) / 16.0, d.target.astype(np.int64)
        Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.2, seed=subset_seed)
        info = dict(dataset="sklearn load_digits (8x8) FALLBACK", fallback_reason=repr(e),
                    n_train=len(Xtr), n_test=len(Xte), n_features=64,
                    subset_seed=subset_seed, scaling="x/16 -> [0,1]")
        return Xtr, ytr, Xte, yte, info
