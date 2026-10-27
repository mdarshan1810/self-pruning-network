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

## Part 4 results (see results/)
- `pareto.png`, `sweep_raw.json`, `summary.json`: accuracy vs sparsity, 20 seeds, saliency vs magnitude.
- `CLAIM.md`: falsifiable claim, generated from the numbers. Pre-specified 90% comparison is a **null result**.
- `cost.json`: FLOPs fall ~1/(1-sparsity), but the NumPy sparse path is **slower in wall-clock** than dense BLAS
  (gather/reduceat overhead). FLOPs/active-params are the honest cost metric here; this is not a real speedup.
