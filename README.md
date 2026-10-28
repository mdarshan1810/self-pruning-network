# Self-Pruning Network (NumPy only)

Pure NumPy autodiff engine + MLP that prunes itself during training.
Scikit-learn is used **only to load MNIST** (data loading), never for training, models or gradients.

## Install
    python -m venv .venv && .venv/Scripts/activate
    pip install -r requirements.txt

## Commands (filled in as parts are completed)
- Gradient checks:        `pytest tests -q`
- Part 2 training:        `python -m train.train_dense`
- Part 3 self-pruning:    `python -m train.train_prune --sparsity 0.9 --criterion saliency`
- Part 4 Pareto sweep:    `python -m train.sweep` (~6 min; writes results/)
- Regrowth ablation:      `python -m train.ablate_regrow` (saliency, regrow_frac 0/0.05/0.1 at 90% and 95%, 20 seeds; ~4 min; writes results/regrow_ablation.json, results/REGROW.md)
- MNIST replication:      `python -m train.sweep_mnist` (10k/2k MNIST subset, 784-128-10, 20 seeds; ~10 min parallel; writes results/mnist_*, results/MNIST.md)

## Part 4 results (see results/)
- `pareto.png`, `sweep_raw.json`, `summary.json`: accuracy vs sparsity, 20 seeds, saliency vs magnitude.
- `CLAIM.md`: falsifiable claim, generated from the numbers (spirals, pre-specified 90%: saliency +1.5 pts, CI [+0.5, +2.4]).
- `MNIST.md`, `mnist_*`: pre-specified replication at 95% on MNIST: saliency +0.6 pts, CI [+0.3, +0.9].
  Not uniformly better: at 75-90% on MNIST magnitude wins by ~0.4 pts.
- `REGROW.md`, `regrow_ablation.json`: regrowth works mechanically but shows no reliable accuracy effect (null).
- `cost.json`: FLOPs fall ~1/(1-sparsity), but the NumPy sparse path is **slower in wall-clock** than dense BLAS
  (gather/reduceat overhead). FLOPs/active-params are the honest cost metric here; this is not a real speedup.
