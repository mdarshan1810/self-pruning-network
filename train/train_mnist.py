"""MNIST replication of Part 3: train while pruning on real images.

Run: python -m train.train_mnist --sparsity 0.95 --criterion saliency

Same loop as train/train_prune.py (copied, not imported, so that file stays untouched):
same Pruner, Adam, mlp and softmax_cross_entropy; only the data and sizes differ.
Data: a fixed, seeded 10k-train / 2k-test subset of MNIST (see train/data.load_mnist),
identical for every training seed; the seed changes init and mini-batch order only.
"""
import argparse
import json

import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import mlp
from prune.pruner import Pruner
from train.data import load_mnist
from train.optim import Adam
from train.train_dense import accuracy

_DATA = {}


def get_data(n_train=10000, n_test=2000):
    """Load once per process (OpenML parsing takes a few seconds)."""
    key = (n_train, n_test)
    if key not in _DATA:
        _DATA[key] = load_mnist(n_train=n_train, n_test=n_test, subset_seed=0)
    return _DATA[key]


def run(seed=0, sparsity=0.95, criterion="saliency", epochs=8, batch=128, lr=1e-3,
        hidden=(128,), regrow_frac=0.0, verbose=False, n_train=10000, n_test=2000):
    Xtr, ytr, Xte, yte, info = get_data(n_train, n_test)
    n_cls = 10
    model = mlp([Xtr.shape[1], *hidden, n_cls], seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    steps_per_epoch = int(np.ceil(len(Xtr) / batch))
    total = epochs * steps_per_epoch
    # same schedule shape as train_prune: start after epoch 1, ramp over the first 60%
    # of training, last 40% fine-tunes at fixed sparsity
    pruner = Pruner(model, opt, criterion=criterion, s_final=sparsity, t0=steps_per_epoch,
                    t_end=int(0.6 * total), every=max(1, steps_per_epoch // 2),
                    regrow_frac=regrow_frac)
    rng = np.random.default_rng(seed)
    curve = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = softmax_cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
            loss.backward()
            pruner.accumulate()
            opt.step()
            pruner.maybe_prune()
            losses.append(loss.data.item())
        assert np.isfinite(np.mean(losses)), "NaN/inf loss"
        curve.append(dict(epoch=ep, loss=float(np.mean(losses)), sparsity=float(pruner.sparsity()),
                          test_acc=accuracy(model, Xte, yte)))
        if verbose:
            print(curve[-1], flush=True)
    for p in model.prunable():
        assert np.all(p.data[p.mask == 0] == 0.0)
    return model, curve, info


if __name__ == "__main__":
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.95)
    ap.add_argument("--criterion", default="saliency", choices=["saliency", "magnitude"])
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    get_data()
    t = time.perf_counter()
    _, curve, info = run(a.seed, a.sparsity, a.criterion, epochs=a.epochs,
                         regrow_frac=a.regrow, verbose=True)
    print(json.dumps(info), f"train time {time.perf_counter() - t:.1f}s")
