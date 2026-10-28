"""Part 3: the self-pruning controller.

Per-layer (uniform) sparsity: every prunable weight matrix is pruned to the same
fraction. Global thresholds would let one layer collapse entirely, because score
scales differ per layer; per-layer is the simpler, safer choice here.

Every `every` steps: set the target s(t) from the schedule, keep the top-k alive
weights per layer by score, and (optionally) swap a small fraction for regrown
weights chosen by the largest |dense gradient| among pruned ones. Regrown
weights start at exactly 0 with cleared optimizer state (see train/optim.py).
Mask changes always go through optimizer.set_mask.

Regrowth details (each fixes a measured failure, see results/REGROW.md):
- A revived weight starts at w = 0, so its score |w*g| (or |w|) is ~0 and it was
  the first thing dropped at the next prune event (<2% survived at fixed
  sparsity). Revived weights are therefore immune for `grace` prune events.
- The score EMA of a revived element is reset and bias-corrected (as in Adam:
  score / (1 - ema^n), n = updates since revival), so after the grace period it
  is compared fairly with long-lived weights instead of being shrunk towards 0.
  With no regrowth every element has the same n, so rankings are unchanged.
- Growth candidates are ranked by the summed |dense gradient| over the last
  prune interval only. A long EMA kept stale non-zero values for connections
  into dead units, whose gradient is exactly 0; reviving those is a no-op. Dead
  weights with zero gradient over the whole interval are never regrown, and
  neither is a weight pruned at the same event (that would skip the reset to 0).
"""
import numpy as np

from prune.criteria import CRITERIA
from prune.schedule import cubic_sparsity


class Pruner:
    def __init__(self, model, optimizer, criterion="saliency", s_final=0.9,
                 t0=0, t_end=1000, every=50, ema=0.9, regrow_frac=0.0, grace=1):
        self.params = model.prunable()
        self.opt = optimizer
        self.score_fn = CRITERIA[criterion]
        self.s_final, self.t0, self.t_end, self.every = s_final, t0, t_end, every
        self.ema, self.regrow_frac, self.grace = ema, regrow_frac, grace
        self.score = {id(p): np.zeros_like(p.data) for p in self.params}
        self.n_upd = {id(p): np.zeros_like(p.data) for p in self.params}  # EMA updates since (re)birth
        self.gabs = {id(p): np.zeros_like(p.data) for p in self.params}   # sum |dense_grad| this interval
        self.born = {id(p): np.full(p.data.shape, -np.inf) for p in self.params}  # event of last revival
        self.step_count = 0
        self.n_events = 0
        # regrowth bookkeeping: revived count, and at each cohort's first competitive event
        # how many were there, had moved off 0, and were kept
        self.stats = dict(revived=0, competed=0, moved=0, survived=0)

    def accumulate(self):
        """Call after backward(), before opt.step(): update EMAs of the scores."""
        for p in self.params:
            # While the mask is all ones, Parameter.effective() skips masked(), so dense_grad is
            # None; then p.grad IS dL/dW_eff exactly. Without this fallback every score stayed 0
            # until the first prune event, which then cut arbitrary tied weights.
            g = p.dense_grad if p.dense_grad is not None else p.grad
            if g is None:
                continue
            k = id(p)
            s = self.score_fn(p.data, g=g)
            self.score[k] = self.ema * self.score[k] + (1 - self.ema) * s
            self.n_upd[k] += 1
            self.gabs[k] += np.abs(g)

    def debiased_score(self, p):
        k = id(p)
        n = self.n_upd[k]
        return np.where(n > 0, self.score[k] / (1.0 - self.ema ** np.maximum(n, 1)), 0.0)

    def protected(self, p):
        """Revived weights that are still inside their grace period."""
        return self.n_events - self.born[id(p)] <= self.grace

    def maybe_prune(self):
        """Call once per training step (after opt.step())."""
        self.step_count += 1
        t = self.step_count
        if t < self.t0 or t % self.every != 0 and t != self.t_end:
            return
        self.n_events += 1
        s = cubic_sparsity(t, self.s_final, self.t0, self.t_end)
        for p in self.params:
            self._prune_layer(p, s)
            self.gabs[id(p)][:] = 0.0  # growth signal is per interval

    def _prune_layer(self, p, s):
        k = id(p)
        n = p.data.size
        n_keep = max(1, int(round((1.0 - s) * n)))
        alive = p.mask.ravel() > 0
        prot = self.protected(p).ravel() & alive
        score = self.debiased_score(p).ravel()
        new_mask = p.mask.ravel().copy()
        if alive.sum() > n_keep:
            sc = np.where(alive, score, -np.inf)
            sc[prot] = np.inf  # grace period: keep revived weights
            keep = np.argpartition(-sc, n_keep - 1)[:n_keep]
            new_mask[:] = 0.0
            new_mask[keep] = 1.0
        # cohorts revived grace+1 events ago compete for the first time at this event
        first = (self.n_events - self.born[k].ravel() == self.grace + 1) & alive
        self.stats["competed"] += int(first.sum())
        self.stats["moved"] += int((first & (p.data.ravel() != 0)).sum())
        if self.regrow_frac > 0 and s > 0:
            dead = ~alive  # never regrow what was pruned at this same event
            g = self.gabs[k].ravel()
            droppable = (new_mask > 0) & ~prot
            # only regrow weights that actually received gradient; drop as many as we grow
            n_swap = min(int(self.regrow_frac * new_mask.sum()),
                         int((dead & (g > 0)).sum()), int(droppable.sum()))
            if n_swap > 0:
                grow = np.argpartition(-np.where(dead, g, -np.inf), n_swap - 1)[:n_swap]
                drop = np.argpartition(np.where(droppable, score, np.inf), n_swap - 1)[:n_swap]
                new_mask[drop] = 0.0
                new_mask[grow] = 1.0
                self.score[k].flat[grow] = 0.0  # fresh, bias-corrected EMA for revived weights
                self.n_upd[k].flat[grow] = 0.0
                self.born[k].flat[grow] = self.n_events
                self.stats["revived"] += n_swap
        self.stats["survived"] += int((first & (new_mask > 0)).sum())  # after the swap
        self.opt.set_mask(p, new_mask.reshape(p.data.shape))

    def sparsity(self):
        tot = sum(p.mask.size for p in self.params)
        return 1.0 - sum(p.mask.sum() for p in self.params) / tot
