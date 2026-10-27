"""Part 2: SGD-momentum and Adam, mask-aware.

The trap: zeroing a masked weight's *gradient* is not enough. Adam's moments m, v
keep their old values, so m/(sqrt(v)+eps) is still non-zero and the "pruned"
weight drifts away from zero. Worse, a revived weight would resume with stale
moments and a stale bias-correction step count.

Correct handling (three parts):
  1. The update is multiplied by the mask, and weights are re-zeroed after the step.
  2. `set_mask` clears m, v and the per-element step count for every element
     whose mask value changed (pruned OR revived).
  3. The step count is per element, so a revived weight gets a fresh bias correction.
"""
import numpy as np


class _Base:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def set_mask(self, p, new_mask):
        raise NotImplementedError


class SGD(_Base):
    def __init__(self, params, lr=0.1, momentum=0.9):
        super().__init__(params)
        self.lr, self.mom = lr, momentum
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            v = self.v[id(p)]
            v *= self.mom
            v += p.grad * p.mask
            p.data -= self.lr * v * p.mask
            p.data *= p.mask

    def set_mask(self, p, new_mask):
        changed = p.mask != new_mask
        self.v[id(p)][changed] = 0.0
        p.mask = new_mask.astype(np.float64)
        p.data *= p.mask


class Adam(_Base):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = {id(p): np.zeros_like(p.data) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}
        self.t = {id(p): np.zeros_like(p.data) for p in self.params}  # per-element

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            m, v, t = self.m[id(p)], self.v[id(p)], self.t[id(p)]
            g = p.grad * p.mask
            alive = p.mask > 0
            t[alive] += 1.0
            m[:] = np.where(alive, self.b1 * m + (1 - self.b1) * g, 0.0)
            v[:] = np.where(alive, self.b2 * v + (1 - self.b2) * g * g, 0.0)
            tt = np.maximum(t, 1.0)
            mhat = m / (1 - self.b1 ** tt)
            vhat = v / (1 - self.b2 ** tt)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps) * p.mask
            p.data *= p.mask

    def set_mask(self, p, new_mask):
        changed = p.mask != new_mask
        for state in (self.m, self.v, self.t):
            state[id(p)][changed] = 0.0
        p.mask = new_mask.astype(np.float64)
        p.data *= p.mask
