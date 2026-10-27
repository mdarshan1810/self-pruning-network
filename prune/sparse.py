"""Sparse-aware inference. Work is proportional to the number of NONZERO weights.

Each Linear is stored in column-sorted COO form (rows, cols, vals). Forward:
    contrib[:, j] = x[:, rows[j]] * vals[j]        (gather: B x nnz multiplies)
    y[:, c]       = sum of contrib over entries with cols == c   (np.add.reduceat)
This is NOT dense-times-zero: pruned weights are not stored and not touched.

Honest caveat: NumPy gathers are slower per-FLOP than BLAS matmul, so wall-clock
may not beat the dense path until sparsity is high. FLOPs (counted exactly) are the
reliable cost metric; timing is reported as measured, whatever it shows.
"""
import numpy as np


class SparseLinear:
    def __init__(self, W, b):
        rows, cols = np.nonzero(W)
        order = np.argsort(cols, kind="stable")
        self.rows, self.cols = rows[order], cols[order]
        self.vals = W[self.rows, self.cols]
        self.b, self.n_out = b.copy(), W.shape[1]
        self.nnz = len(self.vals)
        self.present = np.unique(self.cols)
        self.starts = np.searchsorted(self.cols, self.present)

    def __call__(self, x):
        y = np.zeros((x.shape[0], self.n_out))
        if self.nnz:
            contrib = x[:, self.rows] * self.vals
            y[:, self.present] = np.add.reduceat(contrib, self.starts, axis=1)
        return y + self.b

    def flops(self, batch=1):
        return 2 * self.nnz * batch


class SparseMLP:
    """Built from a trained `Sequential` of Linear/ReLU."""

    def __init__(self, model):
        from nn.layers import Linear
        self.ops = []
        for layer in model.layers:
            if isinstance(layer, Linear):
                self.ops.append(SparseLinear(layer.W.data * layer.W.mask, layer.b.data))
            else:
                self.ops.append("relu")

    def __call__(self, x):
        for op in self.ops:
            x = np.maximum(x, 0.0) if op == "relu" else op(x)
        return x

    def flops(self, batch=1):
        return sum(o.flops(batch) for o in self.ops if o != "relu")

    def active_params(self):
        return sum(o.nnz for o in self.ops if o != "relu")


def dense_flops(model, batch=1):
    from nn.layers import Linear
    return sum(2 * l.W.data.size * batch for l in model.layers if isinstance(l, Linear))
