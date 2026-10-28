"""Regrowth (Pruner(regrow_frac>0)): exact sparsity, clean revival, and that regrowth is not a no-op.

Uses its own tiny synthetic problem so it does not depend on train/data.py.
"""
import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import mlp
from prune.pruner import Pruner
from prune.schedule import cubic_sparsity
from train.optim import Adam

LR = 1e-2


def _data(seed=0, n=512):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    r, a = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
    y = ((a + 2.0 * r) % (2 * np.pi) // (2 * np.pi / 3)).astype(int)  # 3-class swirl
    return X, y


def _train(regrow_frac=0.1, s_final=0.9, steps=400, seed=0, on_set_mask=None, after_event=None):
    X, y = _data(seed)
    model = mlp([2, 32, 32, 3], seed=seed)
    opt = Adam(model.parameters(), lr=LR)
    pr = Pruner(model, opt, criterion="saliency", s_final=s_final, t0=10, t_end=250, every=10,
                regrow_frac=regrow_frac)
    if on_set_mask is not None:
        orig = opt.set_mask

        def wrapped(p, new_mask):
            old = p.mask.copy()
            orig(p, new_mask)
            on_set_mask(pr, opt, p, old)
        opt.set_mask = wrapped
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(len(X), 64, replace=False)
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        pr.accumulate()
        opt.step()
        ev = pr.n_events
        pr.maybe_prune()
        if after_event is not None and pr.n_events > ev:
            after_event(pr)
    return model, opt, pr


def test_regrowth_keeps_target_sparsity_exact_at_every_event():
    for frac in (0.05, 0.1, 0.3):
        def check(pr):
            s = cubic_sparsity(pr.step_count, pr.s_final, pr.t0, pr.t_end)
            for p in pr.params:
                assert p.mask.sum() == max(1, int(round((1.0 - s) * p.mask.size)))
                assert set(np.unique(p.mask)) <= {0.0, 1.0}
        _, _, pr = _train(regrow_frac=frac, after_event=check)
        assert pr.stats["revived"] > 0  # the swaps really happened
        assert abs(pr.sparsity() - 0.9) < 0.005


def test_revived_weights_start_at_zero_with_cleared_state():
    seen = {"n": 0}
    first_step = {}

    def on_set_mask(pr, opt, p, old):
        rev = (old == 0) & (p.mask == 1)
        if not rev.any():
            return
        seen["n"] += int(rev.sum())
        k = id(p)
        assert np.all(p.data[rev] == 0.0)
        assert np.all(opt.m[k][rev] == 0) and np.all(opt.v[k][rev] == 0) and np.all(opt.t[k][rev] == 0)
        assert np.all(pr.score[k][rev] == 0) and np.all(pr.n_upd[k][rev] == 0)  # fresh score EMA
        assert np.all(pr.gabs[k][rev] > 0)  # only weights that actually received gradient regrow
        first_step[k] = (p, rev.copy())

    # 120 is a prune event (every=10), so the last revivals have not trained yet
    model, opt, pr = _train(on_set_mask=on_set_mask, steps=120)
    assert seen["n"] > 0
    # one more step: a freshly revived weight's first Adam step is ~lr (fresh bias correction)
    X, y = _data(0)
    opt.zero_grad()
    softmax_cross_entropy(model(Tensor(X[:64])), y[:64]).backward()
    opt.step()
    n_checked = 0
    for p, rev in first_step.values():
        just = rev & (opt.t[id(p)] == 1)
        n_checked += int(just.sum())
        assert np.all(np.abs(p.data[just]) <= LR * 1.0001)
    assert n_checked > 0


def test_revived_weights_get_grace_period():
    """Regression: revived weights (w=0, score~0) used to be dropped at the very next event."""
    prev = {}

    def on_set_mask(pr, opt, p, old):
        k = id(p)
        if k in prev:
            assert np.all(p.mask[prev[k]] == 1)  # last event's revivals survive this event
        prev[k] = (old == 0) & (p.mask == 1)

    _train(on_set_mask=on_set_mask)


def test_some_regrown_weights_survive_and_become_nonzero():
    ever, cohorts, kept = {}, {}, {"n": 0, "of": 0}

    def on_set_mask(pr, opt, p, old):
        k = id(p)
        rev = (old == 0) & (p.mask == 1)
        ever[k] = ever.get(k, np.zeros(p.mask.shape, bool)) | rev
        # independent count: cohort revived grace+1 events ago, kept after this event's swap?
        for ev, idx in cohorts.get(k, []):
            if pr.n_events - ev == pr.grace + 1:
                kept["n"] += int(p.mask[idx].sum()); kept["of"] += int(idx.sum())
        cohorts.setdefault(k, []).append((pr.n_events, rev))

    model, _, pr = _train(on_set_mask=on_set_mask)
    st = pr.stats
    assert st["competed"] > 0
    assert (st["survived"], st["competed"]) == (kept["n"], kept["of"])  # bookkeeping is right
    assert st["moved"] / st["competed"] > 0.9  # they train, they are not stuck at 0
    assert st["survived"] > 0  # some win against established weights on merit
    end = sum(int(np.sum((p.mask == 1) & ever[id(p)] & (p.data != 0))) for p in model.prunable())
    assert end > 0  # and are part of the final model, with non-zero values


def test_scores_are_nonzero_before_first_prune_and_first_event_keeps_top_scores():
    """Regression: with an all-ones mask dense_grad is None; scores used to stay 0 until the
    first prune, so that event cut arbitrary tied weights regardless of the criterion."""
    X, y = _data(0)
    for criterion in ("saliency", "magnitude"):
        model = mlp([2, 32, 32, 3], seed=0)
        opt = Adam(model.parameters(), lr=LR)
        pr = Pruner(model, opt, criterion=criterion, s_final=0.9, t0=0, t_end=250, every=10)
        rng = np.random.default_rng(0)
        for step in range(1, 11):  # first prune event is at step 10, with s(10) > 0
            idx = rng.choice(len(X), 64, replace=False)
            opt.zero_grad()
            softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
            pr.accumulate()
            opt.step()
            if step == 10:
                before = {id(p): pr.debiased_score(p).copy() for p in pr.params}
                for p in pr.params:
                    assert (before[id(p)] > 0).mean() > 0.5  # scored before any pruning (was 0%; dead ReLUs give exact 0)
            pr.maybe_prune()
        assert pr.n_events == 1
        for p in pr.params:
            sc = before[id(p)]
            kept, cut = sc[p.mask == 1], sc[p.mask == 0]
            assert cut.size > 0 and kept.min() >= cut.max()  # strictly top-k by score


def test_no_regrowth_means_no_revivals():
    _, _, pr = _train(regrow_frac=0.0)
    assert pr.stats["revived"] == 0 and abs(pr.sparsity() - 0.9) < 0.005
