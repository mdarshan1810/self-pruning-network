# Self-Pruning Network (NumPy only)

A reverse-mode autodiff engine written from scratch in NumPy, and an MLP that prunes its own
connections while it trains.

- No PyTorch, TensorFlow, JAX, Keras or any autodiff/NN/pruning library is used.
- **scikit-learn is used only to load MNIST** (`train/data.py: load_mnist`). It is never used for
  models, training or gradients.
- Design reasoning and the four required answers: [`DESIGN.md`](DESIGN.md).

## Install

Python 3.10+.

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

The MNIST commands download MNIST from OpenML on first use (about 50 MB, cached by scikit-learn).

## Commands

| What | Command | Time | Writes |
|---|---|---|---|
| All tests (gradient checks, masked-weight tests, pruning, regrowth) | `pytest tests -q` | ~5 s | |
| Part 2: dense training + learning curve | `python -m train.train_dense` | ~5 s | `results/dense_curve.{json,png}` |
| Part 3: self-pruning run vs dense | `python -m train.train_prune --sparsity 0.9 --criterion saliency` | ~5 s | `results/prune_curve.{json,png}` |
| Part 4: Pareto sweep, baseline, cost, claim | `python -m train.sweep` | ~4 min | `results/pareto.png`, `sweep_raw.json`, `summary.json`, `cost.json`, `CLAIM.md` |
| Bonus: regrowth ablation | `python -m train.ablate_regrow` | ~2 min | `results/REGROW.md`, `regrow_ablation.json` |
| Bonus: MNIST replication | `python -m train.sweep_mnist` | ~18 min (7 processes) | `results/MNIST.md`, `mnist_*` |

All seeds are fixed. Re-running the commands from a clean clone reproduces the committed
numbers exactly (checked for sample spirals and MNIST runs on the same machine; other BLAS
builds may differ in the last digits).

## Layout

    engine/tensor.py     Tensor, ops, backward(), masked(), softmax_cross_entropy
    nn/layers.py         Parameter (with mask), Linear (He init), ReLU, Sequential
    train/optim.py       mask-aware SGD-momentum and Adam
    prune/criteria.py    magnitude and saliency |w*g| scores
    prune/schedule.py    cubic sparsity ramp
    prune/pruner.py      the self-pruning controller (+ optional regrowth)
    prune/sparse.py      sparse-aware inference path (stores only non-zero weights)
    train/               data, training loops, sweeps
    tests/               32 tests

## Results at a glance

**Part 2.** On 3-arm spirals a 2-64-64-3 MLP (He init, Adam lr 1e-2) reaches 100% train and test
accuracy with no NaNs (the loop asserts the loss stays finite). He init (std = sqrt(2/fan_in)) is
used because ReLU zeroes about half of each layer's inputs, which halves activation variance per
layer. The factor of 2 compensates, so activations neither vanish nor explode with depth.

**Parts 3-4.** Mean test accuracy over 20 seeds. Spirals is a harder 5-arm noisy variant, where the
dense model reaches only ~72%.

| sparsity | spirals magnitude | spirals saliency | MNIST magnitude | MNIST saliency |
|---|---|---|---|---|
| 0% (dense) | 72.2 | 72.2 | 95.0 | 95.0 |
| 50% | 72.5 | 72.7 | 94.5 | 94.6 |
| 75% | 72.9 | 73.2 | 94.1 | 93.7 |
| 90% | 70.9 | 72.4 | 92.7 | 92.3 |
| 95% | 53.8 | 61.7 | 90.5 | 91.1 |

- **Cost of pruning:** at 90% sparsity, about 0 pts on spirals (saliency) and about 2.7 pts on
  MNIST.
- **Falsifiable claim** (`results/CLAIM.md`, pre-specified): at 90% sparsity on spirals, saliency
  beats magnitude by +1.5 pts (paired 95% CI [+0.5, +2.4], 16/20 seeds).
- **Replication** (`results/MNIST.md`, pre-specified): at 95% sparsity on MNIST, saliency is ahead
  by +0.6 pts (CI [+0.3, +0.9]).
- **Not uniformly better:** on MNIST at 75-90%, magnitude wins by about 0.4 pts.
- **Regrowth** (`results/REGROW.md`): it works mechanically (revived weights train and some are
  kept), but it has no reliable accuracy effect. This is a null result.
- **Cost** (`results/cost.json`): at 90% sparsity, active weights fall from 4,544 to 455 and
  multiply-adds to 0.10x. These come from a sparse path that stores and touches only the non-zero
  weights, not dense times zero.
  - Wall-clock is **slower** than dense BLAS (about 1 ms vs 0.15 ms per batch of 256), because
    NumPy gather/reduceat overhead outweighs the saved FLOPs.
  - So this is a real reduction in work and parameters, but **not** a real latency speedup.
  - See DESIGN.md section 3.
