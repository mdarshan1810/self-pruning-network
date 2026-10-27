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
- Part 4 Pareto sweep:    TODO
