"""Part 2: train a dense MLP on spirals.   Run: python -m train.train_dense"""
import json

import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import mlp
from train.data import make_spirals, train_test_split
from train.optim import Adam


def accuracy(model, X, y):
    return float((model(Tensor(X)).data.argmax(axis=1) == y).mean())


def train(seed=0, epochs=60, batch=64, lr=1e-2, hidden=(64, 64), verbose=True):
    X, y = make_spirals(seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    model = mlp([2, *hidden, 3], seed=seed)
    opt = Adam(model.parameters(), lr=lr)
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
            opt.step()
            losses.append(loss.data.item())
        row = dict(epoch=ep, loss=float(np.mean(losses)),
                   train_acc=accuracy(model, Xtr, ytr), test_acc=accuracy(model, Xte, yte))
        assert np.isfinite(row["loss"]), "NaN/inf loss"
        curve.append(row)
        if verbose and (ep % 10 == 0 or ep == 1):
            print(row)
    return model, curve


if __name__ == "__main__":
    _, curve = train()
    with open("results/dense_curve.json", "w") as f:
        json.dump(curve, f, indent=1)
