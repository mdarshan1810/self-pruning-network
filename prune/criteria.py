"""Part 3: importance criteria. Higher score = more important = keep.

Saliency (first-order Taylor). Let L(w) be the loss and w_i one weight. Removing
it sets w_i -> 0, so
    L(w with w_i=0) - L(w) ~ -g_i * w_i + O(w_i^2)      (g_i = dL/dw_i)
    |delta L_i| ~ |g_i * w_i|
Magnitude pruning keeps |w_i| and ignores g_i: a large weight feeding a dead
unit changes nothing, a small weight on a sensitive path changes a lot. A single
mini-batch g_i is noisy, so the pruner keeps an EMA of |g_i * w_i| over steps.
Caveat: at a minimum g -> 0 and the first-order term vanishes (the second-order
term, ~ 0.5*H_ii*w_i^2, would dominate); that is a known limit of this estimate.
"""
import numpy as np


def magnitude_score(w, **_):
    return np.abs(w)


def saliency_score(w, g, **_):
    return np.abs(w * g)


CRITERIA = {"magnitude": magnitude_score, "saliency": saliency_score}
