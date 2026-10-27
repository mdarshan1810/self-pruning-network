"""Part 3: train while pruning.   Run: python -m train.train_prune --sparsity 0.9"""
import argparse
import json

import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import mlp
from prune.pruner import Pruner
from train.data import make_spirals, train_test_split
from train.optim import Adam
from train.train_dense import accuracy


def run(seed=0, sparsity=0.9, criterion="saliency", epochs=80, batch=64, lr=1e-2,
        hidden=(64, 64), regrow_frac=0.0, verbose=False, data_kw=None):
    X, y = make_spirals(seed=seed, **(data_kw or {}))
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    n_cls = int(y.max()) + 1
    model = mlp([2, *hidden, n_cls], seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    steps_per_epoch = int(np.ceil(len(Xtr) / batch))
    total = epochs * steps_per_epoch
    # ramp over the first 60% of training; last 40% fine-tunes at fixed sparsity
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
        if verbose and (ep % 10 == 0 or ep == 1):
            print(curve[-1])
    # honesty check: pruned weights are exactly zero in the stored data
    for p in model.prunable():
        assert np.all(p.data[p.mask == 0] == 0.0)
    return model, curve, (Xte, yte)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", default="saliency", choices=["saliency", "magnitude"])
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _, curve, _ = run(a.seed, a.sparsity, a.criterion, regrow_frac=a.regrow, verbose=True)
    with open("results/prune_curve.json", "w") as f:
        json.dump(curve, f, indent=1)
