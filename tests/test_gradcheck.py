"""Part 1: finite-difference gradient checks for every op (incl. broadcasting)."""
import numpy as np
import pytest

from engine.tensor import Tensor, softmax_cross_entropy

rng = np.random.default_rng(0)


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(*x.shape):
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check(build, *arrays, tol=1e-5):
    """build(*Tensors) -> scalar Tensor. Compares analytic vs numeric grads."""
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    build(*ts).backward()
    for t, a in zip(ts, arrays):
        arr = a.copy()

        def f():
            return build(*[Tensor(arr if x is a else x) for x in arrays]).data.item()
        num = numeric_grad(f, arr)
        assert np.allclose(t.grad, num, atol=tol, rtol=1e-4), (t.grad, num)


A = lambda *s: rng.normal(size=s)


def test_add():        check(lambda a, b: (a + b).sum(), A(3, 4), A(3, 4))
def test_sub():        check(lambda a, b: ((a - b) * (a - b)).sum(), A(3, 4), A(3, 4))
def test_mul():        check(lambda a, b: (a * b).sum(), A(3, 4), A(3, 4))
def test_div():        check(lambda a, b: (a / b).sum(), A(3, 4), rng.uniform(1, 2, (3, 4)))
def test_matmul():     check(lambda a, b: (a @ b).sum(), A(3, 4), A(4, 5))
def test_sum_axis():   check(lambda a: (a.sum(axis=0) * a.sum(axis=0)).sum(), A(3, 4))
def test_mean():       check(lambda a: (a.mean(axis=1) * a.mean(axis=1)).sum(), A(3, 4))
def test_relu():       check(lambda a: (a.relu() * a.relu()).sum(), A(3, 4) + 0.05)
def test_tanh():       check(lambda a: a.tanh().sum(), A(3, 4))
def test_sigmoid():    check(lambda a: a.sigmoid().sum(), A(3, 4))


def test_broadcast_row_bias():
    """(N, D) + (D,) must sum the gradient over the batch axis."""
    check(lambda x, b: ((x + b) * (x + b)).sum(), A(5, 3), A(3))


def test_broadcast_column():
    check(lambda x, c: (x * c).sum(), A(4, 3), A(4, 1))


def test_reused_node_accumulates():
    check(lambda a: (a * a + a).sum(), A(3, 3))


def test_softmax_cross_entropy():
    y = np.array([0, 2, 1, 2])
    check(lambda z: softmax_cross_entropy(z, y), A(4, 3))


def test_softmax_cross_entropy_stable_for_huge_logits():
    z = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    loss = softmax_cross_entropy(z, np.array([0]))
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(z.grad))


def test_mlp_composition():
    y = np.array([0, 1, 2, 1, 0])
    check(lambda x, w1, b1, w2: softmax_cross_entropy(((x @ w1 + b1).relu() @ w2), y),
          A(5, 4), A(4, 6) * 0.5, A(6) + 0.05, A(6, 3) * 0.5)


def test_backward_twice_accumulates_only_at_leaves():
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    loss = (a * a).sum()
    loss.backward()
    loss.backward()
    assert np.allclose(a.grad, [4.0, 8.0])  # 2 x [2, 4], not double-counted intermediates


def test_constant_branch_through_nonlinearity():
    x = Tensor(np.array([[1.0, -2.0]]))           # no grad needed
    w = Tensor(np.array([[0.5], [0.3]]), requires_grad=True)
    (x.relu() @ w).sum().backward()
    assert np.allclose(w.grad, [[1.0], [0.0]])
