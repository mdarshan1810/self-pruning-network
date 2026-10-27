# DESIGN

Everything below refers to code in this repo and numbers in `results/`. Where a
claim is not backed by a committed experiment, it says so.

## 1. Importance criterion: why `|w * g|` approximates "loss increase if removed"

Let `L(w)` be the loss and `w_i` one weight with gradient `g_i = dL/dw_i`.
Removing the connection sets `w_i -> 0`, i.e. a step `delta = -w_i` along axis `i`.
Taylor-expand the loss around the current weights:

    L(w - w_i e_i) = L(w) - g_i * w_i + 0.5 * H_ii * w_i^2 + O(w_i^3)

so to first order the loss change from removal is `-g_i * w_i`, and the size of
that change is `|g_i * w_i|`. A weight is important to the extent that removing it
would move the loss, so we keep the weights with the largest `|g_i * w_i|`
(`prune/criteria.py: saliency_score`).

Why this differs from magnitude (`|w_i|`): magnitude assumes a big weight always
matters. A large weight feeding a ReLU that is always off has `g_i = 0`, so its
removal changes nothing, yet magnitude keeps it. A small weight on a sensitive
path can have a large `|g_i * w_i|`.

Practical details:
- A single mini-batch gradient is noisy, so the pruner keeps an exponential moving
  average of `|w * g|` (`Pruner.accumulate`, decay 0.9) and ranks by that.
- Pruning is per layer (every layer gets the same sparsity). Score scales differ
  between layers, so a global threshold could wipe out one layer entirely.

Limits of the estimate (honest caveats):
- It is first order. Near a minimum `g -> 0`, so the first-order term vanishes and
  the curvature term `0.5 * H_ii * w_i^2` dominates; `|w*g|` then under-ranks
  weights that matter through curvature. A second-order (OBS/OBD-style) score
  would address this but needs diagonal Hessian estimates, which we did not build.
- It scores weights one at a time. Removing many weights at once ignores
  interactions between them.

Empirical status (`results/CLAIM.md`, 20 paired seeds, 5-arm noisy spirals):
- At 90% sparsity saliency and magnitude are **tied** (paired diff -0.1 pts,
  95% CI [-1.6, +1.2]). The criterion we derived did not help there.
- At 95% saliency is ahead by +8.4 pts (CI [+4.1, +12.8], 15/20 seeds). This was
  noticed after seeing the data, so it is exploratory and needs fresh seeds or a
  second dataset before it is treated as established.
- At 50% magnitude is slightly better (-0.67 pts, CI [-1.22, -0.12]); with several
  comparisons this may be chance. We did not tune either criterion separately.

## 2. The gradient of a masked weight

Forward uses `w_eff = w ⊙ m` with a constant 0/1 mask `m` (`Tensor.masked`). A
pruned weight is a real zero in the forward pass, not a small number.

By the chain rule, with `m` constant:

    dL/dw = m ⊙ dL/dw_eff

So for a masked entry (`m_i = 0`) the exact gradient of the loss with respect to
the stored parameter is **zero**. The engine therefore gives masked weights a zero
**update gradient** (`w.grad = g * m`). This is the mathematically correct gradient
of the masked model, not a hack.

We also keep the *unmasked* quantity `dL/dw_eff` in `w.dense_grad`. It answers
"how much would the loss drop if this connection were switched on?", which is the
signal regrowth needs to rank pruned weights. It is recorded but is never applied
as an update while the weight is masked.

The optimizer is where the usual bug lives. Zeroing a masked weight's gradient is
not enough: Adam's moments `m, v` still hold old values, so `m / (sqrt(v) + eps)` is
non-zero and the "pruned" weight drifts away from zero. A revived weight would also
inherit stale moments and a stale bias-correction step count. `train/optim.py`
therefore:
1. multiplies the update by the mask and re-zeroes pruned weights after each step;
2. in `set_mask`, clears `m`, `v` and the step count for every element whose mask
   changed (pruned or revived);
3. keeps the step count per element, so a revived weight gets a fresh bias
   correction (its first step has magnitude about `lr`, not an inflated one).

`tests/test_masked.py` demonstrates all of this, including a test that reproduces
the naive bug (pruned weight drifts) and tests that the fixed optimizer holds
pruned weights at exactly 0 and revives with clean state.

## 3. Where the engine bottlenecks, and how to fix it

Measured (`results/cost.json`, batch of 256, 64x64 MLP): FLOPs fall to 0.10x at 90%
sparsity, but the hand-written sparse forward path is **slower** in wall-clock than
dense BLAS (about 270-12,000 us vs about 140 us). It is not a real speedup.

Bottlenecks, in rough order of importance:
- **Sparse path uses NumPy gather + `reduceat`.** Gathers (`x[:, rows]`) allocate a
  `B x nnz` temporary and are far slower per FLOP than a BLAS matmul. Fix: CSR/BSR
  kernels in compiled code (C/Rust/Numba), or structured sparsity (below).
- **Training does not get cheaper.** Training runs dense masked matmuls, so pruning
  saves no training compute; `Adam.step` also touches all elements via `np.where`.
  The savings are at inference only.
- **Python-level graph.** Every op allocates a `Tensor` and a closure, and
  `backward()` rebuilds the topological order each call using recursion (a deep
  graph would hit the recursion limit). Fix: cache the order, iterate instead of
  recurse, fuse ops (e.g. linear + bias + ReLU), reuse buffers.
- **float64 everywhere.** float32 halves memory traffic; needed for any real model.
- **Mask applied via an extra elementwise multiply** each forward pass for pruned
  layers. Fix: store weights pre-masked and only re-mask when the mask changes.

## 4. Serving a self-pruned model in a multi-tenant service at scale

This is a design sketch; none of it is implemented or benchmarked here.

- **Export, do not serve the training graph.** Freeze the pruned weights into an
  immutable, versioned artifact (CSR or block-sparse, plus the dense fallback),
  with a checksum. Serving never touches the autodiff engine.
- **Make the sparsity hardware-friendly.** Our pruning is unstructured. On GPUs and
  most CPUs, unstructured sparsity rarely gives real speedup (consistent with our
  own timing). To get latency wins, prune in structured patterns: whole neurons or
  channels (shrinking the dense matrices), N:M (e.g. 2:4) on hardware that
  supports it, or block sparsity matched to the kernel.
- **Batch requests.** Dynamic batching across requests amortizes per-call overhead
  and is where sparse kernels pay off; tenants share a model but never share
  request state.
- **Tenant isolation and fairness.** Per-tenant rate limits and queues so one tenant
  cannot starve others; pruned models are smaller, so more model versions stay
  resident per node (the real multi-tenant gain, via memory and cost per request).
- **Safe rollout.** A pruned model is a different model. Gate each version on a
  held-out accuracy check, run shadow traffic against the dense model and compare
  outputs, canary a small percentage, and keep an automatic rollback to dense.
- **Observability.** Track per-model latency percentiles (p50/p99), accuracy
  proxies, and input drift, since sparse models can degrade faster off-distribution.

## Limitations of this submission

- Single, small synthetic dataset (spirals, 2D input). Results may not transfer;
  MNIST was not run.
- Regrowth is implemented (`Pruner(regrow_frac=...)`) but neither tested nor
  measured, so no claim is made about it.
- Same hyperparameters for both criteria; no tuning. Reported accuracy is final-epoch
  test accuracy with no validation split used for selection.
- The exploratory 95% result has not been replicated on fresh seeds.
