"""Part 3: cubic sparsity ramp (Zhu & Gupta style).

    s(t) = s_final * (1 - (1 - (t - t0) / (t_end - t0))^3)   for t0 <= t <= t_end

Why cubic: the network is most redundant early, so we prune fast first, then slow
down as the remaining weights become important and the loss becomes sensitive.
The ramp ends before training does, leaving a fine-tuning phase at fixed sparsity
so the surviving weights can recover. One-shot pruning at the end skips that
recovery and is the weaker choice.
"""


def cubic_sparsity(step, s_final, t0, t_end):
    if step <= t0:
        return 0.0
    if step >= t_end:
        return s_final
    frac = (step - t0) / (t_end - t0)
    return s_final * (1.0 - (1.0 - frac) ** 3)
