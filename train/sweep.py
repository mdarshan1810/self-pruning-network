"""Part 4: Pareto sweep + baseline comparison + cost measurement.

Run: python -m train.sweep          (writes everything under results/)

Outputs
  results/sweep_raw.json   every (criterion, sparsity, seed) run
  results/summary.json     mean/std per cell, paired saliency-vs-magnitude stats
  results/pareto.png       accuracy vs achieved sparsity (mean +/- std over seeds)
  results/cost.json        FLOPs + wall-clock, sparse path vs dense path
  results/CLAIM.md         the falsifiable claim, generated FROM the numbers
"""
import json
import time

import numpy as np

from engine.tensor import Tensor
from prune.sparse import SparseMLP, dense_flops
from train.train_prune import run

SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95]
CRITERIA = ["magnitude", "saliency"]
SEEDS = list(range(20))
EPOCHS = 60
DATA_KW = dict(n_classes=5, noise=0.45, n_per_class=300)  # hard enough that pruning costs accuracy


def paired_bootstrap(diff, n_boot=10000, seed=0):
    """95% bootstrap CI of the mean paired difference (saliency - magnitude)."""
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sweep():
    raw = []
    for s in SPARSITIES:
        for c in CRITERIA:
            for seed in SEEDS:
                _, curve, _ = run(seed=seed, sparsity=s, criterion=c, epochs=EPOCHS, data_kw=DATA_KW)
                raw.append(dict(criterion=c, target=s, seed=seed,
                                achieved=curve[-1]["sparsity"], test_acc=curve[-1]["test_acc"]))
        print("done sparsity", s, flush=True)
    return raw


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
        d = acc["saliency"] - acc["magnitude"]  # same seed -> same data & init: paired
        lo, hi = paired_bootstrap(d)
        out["paired"][str(s)] = dict(mean_diff=float(d.mean()), ci95=[lo, hi],
                                     saliency_wins=int((d > 0).sum()), ties=int((d == 0).sum()), n=len(d))
    return out


def plot(summary):
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
    ax.set_title(f"Sparsity-accuracy Pareto (mean +/- std, {len(SEEDS)} seeds)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("results/pareto.png", dpi=150)


def cost():
    """Sparse-aware path vs dense forward, batch of 256, model from seed 0."""
    out = []
    for s in SPARSITIES:
        model, _, (Xte, _) = run(seed=0, sparsity=s, criterion="saliency", epochs=EPOCHS, data_kw=DATA_KW)
        sp = SparseMLP(model)
        x = Xte[:256]
        assert np.allclose(sp(x), model(Tensor(x)).data, atol=1e-9)

        def timeit(f, reps=200):
            f()
            t = time.perf_counter()
            for _ in range(reps):
                f()
            return (time.perf_counter() - t) / reps * 1e6

        # dense path = plain NumPy matmuls on the (zeroed) weight arrays, no autodiff overhead
        Ws = [(l.W.data * l.W.mask, l.b.data) for l in model.layers if hasattr(l, "W")]

        def dense_fwd():
            h = x
            for i, (W, b) in enumerate(Ws):
                h = h @ W + b
                if i < len(Ws) - 1:
                    h = np.maximum(h, 0.0)
            return h

        out.append(dict(target=s, active_params=sp.active_params(),
                        dense_params=int(sum(W.size for W, _ in Ws)),
                        sparse_flops=sp.flops(256), dense_flops=dense_flops(model, 256),
                        sparse_us=timeit(lambda: sp(x)), dense_zeros_us=timeit(dense_fwd)))
    return out


def write_claim(summary):
    s = "0.9"
    p, sal, mag = summary["paired"][s], summary["cells"]["saliency@0.9"], summary["cells"]["magnitude@0.9"]
    lo, hi = p["ci95"]
    verdict = ("The 95% CI excludes 0, so the difference is distinguishable from seed noise."
               if lo > 0 or hi < 0 else
               "The 95% CI includes 0, so this experiment does NOT show a reliable difference.")
    txt = (f"# Falsifiable claim (generated from results/summary.json)\n\n"
           f"At {sal['achieved']*100:.1f}% achieved sparsity on 5-arm noisy spirals ({sal['n']} seeds, "
           f"60 epochs, 64x64 MLP), saliency pruning reaches {sal['mean']*100:.1f}% +/- {sal['std']*100:.1f}% "
           f"test accuracy versus {mag['mean']*100:.1f}% +/- {mag['std']*100:.1f}% for magnitude pruning "
           f"(mean paired difference {p['mean_diff']*100:+.1f} pts, 95% bootstrap CI "
           f"[{lo*100:+.1f}, {hi*100:+.1f}], saliency better on {p['saliency_wins']}/{p['n']} seeds).\n\n"
           f"{verdict}\n\n"
           f"The 90% level was fixed as the primary comparison BEFORE running the sweep.\n\n"
           f"## All levels (paired saliency - magnitude, test accuracy)\n\n"
           f"| target sparsity | mean diff (pts) | 95% CI | saliency better |\n|---|---|---|---|\n")
    for k, v in summary["paired"].items():
        txt += (f"| {float(k)*100:.0f}% | {v['mean_diff']*100:+.2f} | "
                f"[{v['ci95'][0]*100:+.2f}, {v['ci95'][1]*100:+.2f}] | {v['saliency_wins']}/{v['n']} |\n")
    txt += ("\n## History and replication\n\n"
            "- The first run of this sweep (commit d95802d) showed a null at 90%. That run had a bug: "
            "scores were all zero before the first prune event, so the first cut was arbitrary for both "
            "criteria. The numbers above are from the fixed code; the 90% level was not re-chosen.\n"
            "- The 95% gap was first noticed post hoc, so 95% was then fixed as the single confirmatory "
            "test on MNIST with fresh seeds. See results/MNIST.md for whether it replicated, and note "
            "that on MNIST the other levels do NOT all favour saliency.\n")
    open("results/CLAIM.md", "w").write(txt)
    return txt


if __name__ == "__main__":
    raw = sweep()
    json.dump(raw, open("results/sweep_raw.json", "w"), indent=1)
    summary = summarize(raw)
    json.dump(summary, open("results/summary.json", "w"), indent=1)
    plot(summary)
    json.dump(cost(), open("results/cost.json", "w"), indent=1)
    print(write_claim(summary))
