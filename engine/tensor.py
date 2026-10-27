"""Part 1: reverse-mode autodiff Tensor.

Each op builds a new Tensor that remembers its parents and a closure
(`_backward`) that pushes `out.grad` into the parents' `.grad`.
`backward()` runs those closures in reverse topological order.

Masking: `masked(w, mask)` multiplies by a constant 0/1 mask. The forward value
is a real zero. In backward the *update* gradient `w.grad` is masked, while the
unmasked ("dense") gradient is recorded in `w.dense_grad` so pruning/regrowth can
score pruned connections without them ever receiving an optimizer update.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape` (inverse of NumPy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    def __init__(self, data, parents=(), requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.dense_grad = None  # only filled for masked() outputs' source
        self.parents = tuple(parents)
        self._backward = lambda: None
        self.requires_grad = requires_grad or any(p.requires_grad for p in self.parents)

    # ---- plumbing -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None
        self.dense_grad = None

    def _acc(self, g):
        g = _unbroadcast(g, self.data.shape)
        self.grad = g.copy() if self.grad is None else self.grad + g

    def backward(self, grad=None):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                build(p)
            topo.append(t)

        build(self)
        # intermediate nodes start fresh each call (leaves keep accumulating, as in PyTorch)
        for t in topo:
            if t.parents:
                t.grad = None
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        for t in reversed(topo):
            if t.grad is not None:  # no gradient reached this node (e.g. constant branch)
                t._backward()

    # ---- element-wise ---------------------------------------------------
    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _b():
            if self.requires_grad: self._acc(out.grad)
            if other.requires_grad: other._acc(out.grad)
        out._backward = _b
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _b():
            if self.requires_grad: self._acc(out.grad * other.data)
            if other.requires_grad: other._acc(out.grad * self.data)
        out._backward = _b
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data / other.data, (self, other))

        def _b():
            if self.requires_grad: self._acc(out.grad / other.data)
            if other.requires_grad: other._acc(-out.grad * self.data / other.data ** 2)
        out._backward = _b
        return out

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    __radd__ = __add__
    __rmul__ = __mul__

    # ---- linear algebra -------------------------------------------------
    def matmul(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _b():
            if self.requires_grad: self._acc(out.grad @ other.data.T)
            if other.requires_grad: other._acc(self.data.T @ out.grad)
        out._backward = _b
        return out

    __matmul__ = matmul

    # ---- reductions -----------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _b():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.data.shape))
        out._backward = _b
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- non-linearities ------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def _b():
            self._acc(out.grad * (self.data > 0))
        out._backward = _b
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _b():
            self._acc(out.grad * (1.0 - t ** 2))
        out._backward = _b
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, (self,))

        def _b():
            self._acc(out.grad * s * (1.0 - s))
        out._backward = _b
        return out

    # ---- masking --------------------------------------------------------
    def masked(self, mask):
        """Forward: self * mask (real zeros). Backward: update grad is masked;
        the unmasked gradient is stashed in `self.dense_grad` (for regrowth scoring)."""
        mask = np.asarray(mask, dtype=np.float64)
        out = Tensor(self.data * mask, (self,))

        def _b():
            g = out.grad
            self.dense_grad = g.copy() if self.dense_grad is None else self.dense_grad + g
            self._acc(g * mask)
        out._backward = _b
        return out


def softmax_cross_entropy(logits, targets):
    """Mean cross-entropy of softmax(logits) vs integer class `targets`.
    Stable: subtract row max, use log-sum-exp. Fused backward: (softmax - onehot)/N."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logsumexp
    n = len(targets)
    loss = -logp[np.arange(n), targets].mean()
    out = Tensor(loss, (logits,))

    def _b():
        p = np.exp(logp)
        p[np.arange(n), targets] -= 1.0
        logits._acc(out.grad * p / n)
    out._backward = _b
    return out
