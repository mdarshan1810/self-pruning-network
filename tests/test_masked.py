"""Part 1: masked-weight correctness (gradient + optimizer state)."""
import numpy as np

from engine.tensor import Tensor
from nn.layers import Parameter
from train.optim import Adam, SGD


def _loss(w, x):
    out = Tensor(x) @ w.effective()
    return (out * out).sum() * 0.5


def test_masked_weight_contributes_zero_to_forward():
    w = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    w.mask = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = Tensor(np.eye(2)) @ w.effective()
    assert np.array_equal(out.data, [[1.0, 0.0], [0.0, 4.0]])  # exact zeros


def test_update_grad_is_masked_but_dense_grad_is_recorded():
    w = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    w.mask = np.array([[1.0, 0.0], [0.0, 1.0]])
    _loss(w, np.array([[1.0, 1.0]])).backward()
    assert np.all(w.grad[w.mask == 0] == 0)        # no update signal
    assert np.any(w.dense_grad[w.mask == 0] != 0)  # but regrowth can still score it


def _naive_adam_drift(steps=50):
    """The common bug: only mask the gradient, leave Adam moments alone."""
    w = Parameter(np.array([[1.0, 2.0]]))
    m = np.zeros_like(w.data); v = np.zeros_like(w.data)
    for t in range(1, 6):  # train both weights, building momentum
        g = w.data.copy()
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w.data -= 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    mask = np.array([[1.0, 0.0]])
    for t in range(6, 6 + steps):  # prune weight 1, zero its grad
        g = w.data * mask
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w.data -= 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w.data[0, 1]


def test_naive_masking_is_buggy_pruned_weight_drifts():
    assert abs(_naive_adam_drift()) > 1e-3  # documents WHY the fix is needed


def test_adam_keeps_pruned_weights_exactly_zero():
    w = Parameter(np.array([[1.0, 2.0, -3.0]]))
    opt = Adam([w], lr=0.1)
    for _ in range(5):
        opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
    opt.set_mask(w, np.array([[1.0, 0.0, 1.0]]))
    for _ in range(50):
        opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
        assert w.data[0, 1] == 0.0  # exact, every step
        # state itself must be clean, not just the weight re-zeroed afterwards
        assert opt.m[id(w)][0, 1] == 0.0 and opt.v[id(w)][0, 1] == 0.0


def test_revived_weight_starts_with_clean_optimizer_state():
    w = Parameter(np.array([[1.0, 2.0]]))
    opt = Adam([w], lr=0.1)
    for _ in range(5):
        opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
    opt.set_mask(w, np.array([[1.0, 0.0]]))
    for _ in range(5):
        opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
    opt.set_mask(w, np.array([[1.0, 1.0]]))  # revive
    k = id(w)
    assert opt.m[k][0, 1] == 0 and opt.v[k][0, 1] == 0 and opt.t[k][0, 1] == 0
    assert w.data[0, 1] == 0.0  # revived at exactly zero
    w.data[0, 1] = 1.0  # give it a non-zero gradient (loss = w^2/2 has g = 0 at w = 0)
    opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
    # fresh state => t=1, mhat=g, vhat=g^2 => first step is exactly lr * sign(g).
    # A stale global step count would give a different (smaller) step.
    assert abs(w.data[0, 1] - 0.9) < 1e-6


def test_sgd_momentum_state_cleared_on_prune():
    w = Parameter(np.array([[1.0, 2.0]]))
    opt = SGD([w], lr=0.1, momentum=0.9)
    for _ in range(5):
        opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
    opt.set_mask(w, np.array([[1.0, 0.0]]))
    assert opt.v[id(w)][0, 1] == 0.0  # velocity cleared at prune time
    for _ in range(20):
        opt.zero_grad(); _loss(w, np.ones((1, 1))).backward(); opt.step()
    assert w.data[0, 1] == 0.0
    opt.set_mask(w, np.array([[1.0, 1.0]]))  # revive at 0: first step must carry no old momentum
    w.data[0, 1] = 1.0
    opt.zero_grad(); _loss(w, np.ones((1, 1))).backward()
    g = w.grad[0, 1]
    opt.step()
    assert abs(w.data[0, 1] - (1.0 - 0.1 * g)) < 1e-12
