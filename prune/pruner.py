"""Part 3: the self-pruning controller.

Per-layer (uniform) sparsity: every prunable weight matrix is pruned to the same
fraction. Global thresholds would let one layer collapse entirely, because score
scales differ per layer; per-layer is the simpler, safer choice here.

Every `every` steps: set the target s(t) from the schedule, keep the top-k alive
weights per layer by score, and (optionally) swap a small fraction for regrown
weights chosen by the largest |dense gradient| among pruned ones. Regrown
weights start at exactly 0 with cleared optimizer state (see train/optim.py).
Mask changes always go through optimizer.set_mask.
"""
import numpy as np

from prune.criteria import CRITERIA
from prune.schedule import cubic_sparsity


class Pruner:
    def __init__(self, model, optimizer, criterion="saliency", s_final=0.9,
                 t0=0, t_end=1000, every=50, ema=0.9, regrow_frac=0.0):
        self.params = model.prunable()
        self.opt = optimizer
        self.score_fn = CRITERIA[criterion]
        self.s_final, self.t0, self.t_end, self.every = s_final, t0, t_end, every
        self.ema, self.regrow_frac = ema, regrow_frac
        self.score = {id(p): np.zeros_like(p.data) for p in self.params}
        self.gabs = {id(p): np.zeros_like(p.data) for p in self.params}
        self.step_count = 0

    def accumulate(self):
        """Call after backward(), before opt.step(): update EMAs of the scores."""
        for p in self.params:
            if p.dense_grad is None:
                continue
            k = id(p)
            s = self.score_fn(p.data, g=p.dense_grad)
            self.score[k] = self.ema * self.score[k] + (1 - self.ema) * s
            self.gabs[k] = self.ema * self.gabs[k] + (1 - self.ema) * np.abs(p.dense_grad)

    def maybe_prune(self):
        """Call once per training step (after opt.step())."""
        self.step_count += 1
        t = self.step_count
        if t < self.t0 or t % self.every != 0 and t != self.t_end:
            return
        s = cubic_sparsity(t, self.s_final, self.t0, self.t_end)
        for p in self.params:
            self._prune_layer(p, s)

    def _prune_layer(self, p, s):
        k = id(p)
        n = p.data.size
        n_keep = max(1, int(round((1.0 - s) * n)))
        alive = p.mask.ravel() > 0
        new_mask = p.mask.ravel().copy()
        if alive.sum() > n_keep:
            sc = np.where(alive, self.score[k].ravel(), -np.inf)
            keep = np.argpartition(-sc, n_keep - 1)[:n_keep]
            new_mask[:] = 0.0
            new_mask[keep] = 1.0
        if self.regrow_frac > 0 and s > 0:
            n_swap = int(self.regrow_frac * new_mask.sum())
            if n_swap > 0:
                dead = new_mask == 0
                if dead.sum() >= n_swap:
                    g = np.where(dead, self.gabs[k].ravel(), -np.inf)
                    grow = np.argpartition(-g, n_swap - 1)[:n_swap]
                    sc = np.where(new_mask > 0, self.score[k].ravel(), np.inf)
                    drop = np.argpartition(sc, n_swap - 1)[:n_swap]
                    new_mask[drop] = 0.0
                    new_mask[grow] = 1.0
        self.opt.set_mask(p, new_mask.reshape(p.data.shape))

    def sparsity(self):
        tot = sum(p.mask.size for p in self.params)
        return 1.0 - sum(p.mask.sum() for p in self.params) / tot
