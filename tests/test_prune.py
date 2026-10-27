"""Part 3 tests: schedule, exact sparsity, sparse path equals dense path."""
import numpy as np

from engine.tensor import Tensor
from prune.schedule import cubic_sparsity
from prune.sparse import SparseMLP, dense_flops
from train.train_prune import run


def test_cubic_schedule_endpoints_and_monotone():
    vals = [cubic_sparsity(t, 0.9, 10, 110) for t in range(0, 130)]
    assert vals[0] == 0.0 and abs(vals[-1] - 0.9) < 1e-12
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_pruning_hits_target_and_sparse_forward_matches_dense():
    model, curve, (Xte, _) = run(seed=0, sparsity=0.8, epochs=15)
    assert abs(curve[-1]["sparsity"] - 0.8) < 0.01
    sp = SparseMLP(model)
    dense_out = model(Tensor(Xte)).data
    assert np.allclose(sp(Xte), dense_out, atol=1e-10)
    assert sp.flops() < 0.25 * dense_flops(model)  # ~80% fewer multiply-adds
