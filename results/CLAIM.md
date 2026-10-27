# Falsifiable claim (generated from results/summary.json)

At 90.0% achieved sparsity on 5-arm noisy spirals (20 seeds, 60 epochs, 64x64 MLP), saliency pruning reaches 71.4% +/- 2.6% test accuracy versus 71.5% +/- 2.6% for magnitude pruning (mean paired difference -0.1 pts, 95% bootstrap CI [-1.6, +1.2], saliency better on 9/20 seeds).

The 95% CI includes 0, so this experiment does NOT show a reliable difference.

The 90% level was fixed as the primary comparison BEFORE running the sweep.

## All levels (paired saliency - magnitude, test accuracy)

| target sparsity | mean diff (pts) | 95% CI | saliency better |
|---|---|---|---|
| 0% | +0.00 | [+0.00, +0.00] | 0/20 |
| 50% | -0.67 | [-1.22, -0.12] | 8/20 |
| 75% | -0.38 | [-1.12, +0.38] | 8/20 |
| 90% | -0.10 | [-1.60, +1.23] | 9/20 |
| 95% | +8.35 | [+4.13, +12.78] | 15/20 |

## Exploratory (post-hoc, NOT the pre-specified claim)

The largest gap appears at the most extreme level. It was noticed after seeing the data and 4 non-trivial levels were compared, so treat it as a hypothesis for replication (e.g. on MNIST), not a confirmed result. A confirmed claim needs a fresh set of seeds.
