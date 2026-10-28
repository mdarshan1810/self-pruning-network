"""MNIST replication of the Part 4 sweep (saliency vs magnitude).

Run: python -m train.sweep_mnist          (~10-15 min on 8 cores; writes results/mnist_*)

Purpose: the spirals sweep found, post hoc, that saliency beats magnitude at 95%
sparsity (results/CLAIM.md, "Exploratory"). Here 95% is the PRE-SPECIFIED
confirmatory comparison, on a different dataset and on fresh seeds (100..119) that
were never looked at during setup. The other levels are reported but secondary.

Outputs
  results/mnist_raw.json      every (criterion, sparsity, seed) run + run metadata
  results/mnist_summary.json  mean/std per cell, paired saliency-vs-magnitude stats
  results/mnist_pareto.png    accuracy vs achieved sparsity (mean +/- std over seeds)
  results/MNIST.md            report, generated FROM the numbers
"""
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from train.sweep import paired_bootstrap

SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95]
CRITERIA = ["magnitude", "saliency"]
SEEDS = list(range(100, 120))  # fresh seeds; seed 0 was used for smoke tests only
PRIMARY = 0.95                 # pre-specified confirmatory level
CFG = dict(epochs=12, batch=128, lr=1e-3, hidden=(128,), n_train=10000, n_test=2000)
PRUNER_PATH = os.path.join(os.path.dirname(__file__), "..", "prune", "pruner.py")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def pruner_preflight():
    """The pre-fix Pruner skipped accumulate() while dense_grad was None (i.e. before anything
    is pruned), so the first prune event ranked all-zero scores and cut arbitrary weights.
    Verify on an unpruned model that one accumulate() already gives non-zero scores."""
    from engine.tensor import Tensor, softmax_cross_entropy
    from nn.layers import mlp
    from prune.pruner import Pruner
    from train.optim import Adam
    rng = np.random.default_rng(0)
    m = mlp([5, 4, 3], seed=0)
    opt = Adam(m.parameters())
    ok = True
    for c in CRITERIA:
        pr = Pruner(m, opt, criterion=c)
        opt.zero_grad()
        softmax_cross_entropy(m(Tensor(rng.normal(size=(8, 5)))), rng.integers(0, 3, 8)).backward()
        pr.accumulate()
        ok &= all(np.any(pr.score[id(q)] != 0) for q in m.prunable())
    return bool(ok)


def _job(args):
    s, c, seed, cfg = args
    from train.train_mnist import run
    t = time.perf_counter()
    _, curve, info = run(seed=seed, sparsity=s, criterion=c, **cfg)
    return dict(criterion=c, target=s, seed=seed, achieved=curve[-1]["sparsity"],
                test_acc=curve[-1]["test_acc"], seconds=time.perf_counter() - t,
                dataset=info["dataset"])


def _init_worker(key, data):
    from train import train_mnist
    train_mnist._DATA[key] = data


def sweep(data, workers=None, seeds=SEEDS, cfg=CFG):
    jobs = [(s, c, seed, cfg) for s in SPARSITIES for c in CRITERIA for seed in seeds]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    raw = []
    with Pool(workers, initializer=_init_worker, initargs=((cfg["n_train"], cfg["n_test"]), data)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            raw.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} runs done", flush=True)
    return sorted(raw, key=lambda r: (r["target"], r["criterion"], r["seed"])), workers


def summarize(raw):
    out = {"cells": {}, "paired": {}}
    for s in SPARSITIES:
        acc = {}
        for c in CRITERIA:
            rows = sorted([r for r in raw if r["criterion"] == c and r["target"] == s], key=lambda r: r["seed"])
            a = np.array([r["test_acc"] for r in rows])
            acc[c] = a
            out["cells"][f"{c}@{s}"] = dict(mean=float(a.mean()), std=float(a.std(ddof=1)),
                                            achieved=float(np.mean([r["achieved"] for r in rows])), n=len(a))
        d = acc["saliency"] - acc["magnitude"]  # same seed -> same init & batch order: paired
        lo, hi = paired_bootstrap(d)
        out["paired"][str(s)] = dict(mean_diff=float(d.mean()), ci95=[lo, hi],
                                     saliency_wins=int((d > 0).sum()), ties=int((d == 0).sum()), n=len(d))
    return out


def plot(summary, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, col in zip(CRITERIA, ["tab:gray", "tab:blue"]):
        xs = [summary["cells"][f"{c}@{s}"]["achieved"] * 100 for s in SPARSITIES]
        m = np.array([summary["cells"][f"{c}@{s}"]["mean"] * 100 for s in SPARSITIES])
        sd = np.array([summary["cells"][f"{c}@{s}"]["std"] * 100 for s in SPARSITIES])
        ax.errorbar(xs, m, yerr=sd, marker="o", capsize=3, label=c, color=col)
    ax.set_xlabel("achieved sparsity (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"{dataset}: sparsity-accuracy (mean +/- std, {len(SEEDS)} seeds)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("results/mnist_pareto.png", dpi=150)


def write_report(summary, meta):
    k = str(PRIMARY)
    p = summary["paired"][k]
    sal, mag = summary["cells"][f"saliency@{PRIMARY}"], summary["cells"][f"magnitude@{PRIMARY}"]
    lo, hi = p["ci95"]
    if lo > 0:
        verdict = ("**Replicated.** The 95% CI excludes 0 in the same direction as on spirals "
                   "(saliency > magnitude).")
    elif hi < 0:
        verdict = ("**Reversed.** The 95% CI excludes 0 in the OPPOSITE direction to spirals "
                   "(magnitude > saliency). The spirals finding does not replicate.")
    else:
        verdict = ("**Not replicated (null).** The 95% CI includes 0, so this experiment does not "
                   "show a reliable saliency advantage at 95% sparsity on this dataset.")
    sp = json.load(open("results/summary.json"))["paired"].get("0.95") if os.path.exists("results/summary.json") else None
    spir = (f"spirals: {sp['mean_diff']*100:+.2f} pts, CI [{sp['ci95'][0]*100:+.2f}, {sp['ci95'][1]*100:+.2f}], "
            f"{sp['saliency_wins']}/{sp['n']} seeds" if sp else "spirals summary not found")
    pruner_note = ("unchanged during the sweep" if meta["pruner_sha_start"] == meta["pruner_sha_end"] else
                   "**CHANGED during the sweep** (another edit landed mid-run; worker processes imported "
                   "it at start-up, so runs may mix versions)")
    h = "-".join(str(x) for x in (meta["n_features"], *CFG["hidden"], 10))
    txt = f"""# MNIST replication of the saliency-vs-magnitude comparison

Generated by `python -m train.sweep_mnist` from `results/mnist_summary.json`.

**Dataset actually used: {meta['dataset']}.** {meta['n_train']} train / {meta['n_test']} test images, a fixed
subset drawn with seed {meta['subset_seed']} from the canonical split (train from the first 60k rows, test from
the last 10k), pixels scaled {meta['scaling']}. The same subset is used for every run; the training seed only
changes weight init and mini-batch order. scikit-learn is used only to download/parse the data.

Model: MLP {h} (ReLU), {meta['n_params']} prunable weights. Adam lr={CFG['lr']}, batch {CFG['batch']},
{CFG['epochs']} epochs. Same Pruner as spirals (per-layer uniform sparsity, cubic ramp from end of epoch 1 to 60%
of training, prune every half epoch, EMA-0.9 scores, no regrowth). Identical hyperparameters for both criteria;
nothing was tuned per criterion. {len(SEEDS)} paired seeds ({SEEDS[0]}..{SEEDS[-1]}), {meta['n_runs']} runs,
mean {meta['mean_seconds']:.1f} s per run ({meta['workers']} parallel worker processes, wall-clock
{meta['wall_minutes']:.1f} min).

## Pre-specified confirmatory test: 95% sparsity

The spirals sweep produced a post-hoc, exploratory finding: saliency beat magnitude at 95% sparsity
({spir}). **Before running this sweep, 95% was fixed as the single confirmatory comparison**, on a new dataset
and on fresh seeds. (Setup used seed 0 only, for timing and to pick the epoch count; seed 0 is not in this sweep.)

At {sal['achieved']*100:.1f}% achieved sparsity, saliency reaches {sal['mean']*100:.2f}% +/- {sal['std']*100:.2f}%
test accuracy versus {mag['mean']*100:.2f}% +/- {mag['std']*100:.2f}% for magnitude (mean paired difference
{p['mean_diff']*100:+.2f} pts, 95% paired-bootstrap CI [{lo*100:+.2f}, {hi*100:+.2f}], saliency better on
{p['saliency_wins']}/{p['n']} seeds, ties {p['ties']}).

{verdict}

## All levels

Secondary results, not corrected for multiple comparisons.

| Target sparsity | Achieved | Magnitude acc | Saliency acc | Mean diff (pts) | 95% CI | Saliency better |
|---|---|---|---|---|---|---|
"""
    for s in SPARSITIES:
        v = summary["paired"][str(s)]
        m_, s_ = summary["cells"][f"magnitude@{s}"], summary["cells"][f"saliency@{s}"]
        txt += (f"| {s*100:.0f}% | {s_['achieved']*100:.1f}% | {m_['mean']*100:.2f} +/- {m_['std']*100:.2f} | "
                f"{s_['mean']*100:.2f} +/- {s_['std']*100:.2f} | {v['mean_diff']*100:+.2f} | "
                f"[{v['ci95'][0]*100:+.2f}, {v['ci95'][1]*100:+.2f}] | {v['saliency_wins']}/{v['n']} |\n")
    txt += f"""
At 0% nothing is pruned, so both criteria produce identical runs (diff exactly 0) - a sanity check on pairing.

## Caveats

- Subset of MNIST (10k/2k), small one-hidden-layer MLP, short training; absolute accuracy is below
  full-MNIST numbers. Accuracy is final-epoch test accuracy, no validation-based selection.
- Uniform per-layer sparsity prunes the 128x10 output layer to the same fraction as the input layer;
  at 95% that layer keeps only {int(round(0.05 * CFG['hidden'][-1] * 10))} of {CFG['hidden'][-1] * 10} weights,
  which likely dominates the accuracy loss at that level for both criteria.
- One dataset subset; variability across seeds reflects init and batch order, not data sampling.
- Pruner version: `prune/pruner.py` sha256 prefix {meta['pruner_sha_start']}, which includes the fix where
  `Pruner.accumulate()` falls back to `p.grad` while nothing is pruned yet (preflight check that scores are
  non-zero before the first prune event: {'passed' if meta['pruner_preflight'] else 'FAILED'}). An earlier,
  aborted run of this sweep used the pre-fix pruner, whose first prune event cut arbitrary weights for both
  criteria; those numbers were discarded and are not reported. The file was {pruner_note}.
"""
    open("results/MNIST.md", "w").write(txt)
    return txt


if __name__ == "__main__":
    # one BLAS thread per worker; spawned workers inherit this env before importing NumPy
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from train.train_mnist import get_data
    data = get_data(CFG["n_train"], CFG["n_test"])
    info = data[4]
    sha0 = _sha(PRUNER_PATH)
    assert pruner_preflight(), "Pruner.accumulate() lacks the p.grad fallback; results would be invalid"
    t0 = time.perf_counter()
    raw, workers = sweep(data)
    wall = (time.perf_counter() - t0) / 60
    sha1 = _sha(PRUNER_PATH)
    n_params = info["n_features"] * CFG["hidden"][0] + sum(
        a * b for a, b in zip(CFG["hidden"], list(CFG["hidden"][1:]) + [10]))
    meta = dict(info, cfg={k: list(v) if isinstance(v, tuple) else v for k, v in CFG.items()},
                seeds=SEEDS, primary=PRIMARY, workers=workers, wall_minutes=wall, n_runs=len(raw),
                mean_seconds=float(np.mean([r["seconds"] for r in raw])), n_params=n_params,
                pruner_sha_start=sha0, pruner_sha_end=sha1, pruner_preflight=pruner_preflight())
    json.dump(dict(meta=meta, runs=raw), open("results/mnist_raw.json", "w"), indent=1)
    summary = summarize(raw)
    summary["meta"] = meta
    json.dump(summary, open("results/mnist_summary.json", "w"), indent=1)
    plot(summary, info["dataset"].split(" ")[0])
    print(write_report(summary, meta))
