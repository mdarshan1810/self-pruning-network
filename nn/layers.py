"""Part 2: composable layers. Linear weights carry a 0/1 pruning mask."""
import numpy as np

from engine.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        self.mask = np.ones_like(self.data)

    def effective(self):
        """Tensor used in forward: real zeros where pruned."""
        return self.masked(self.mask) if not self.mask.all() else self


class Linear:
    """He init (std = sqrt(2/fan_in)) because ReLU zeroes half the activations,
    halving variance; this keeps activation variance ~constant across depth."""

    def __init__(self, n_in, n_out, rng):
        self.W = Parameter(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)))
        self.b = Parameter(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W.effective() + self.b

    def parameters(self):
        return [self.W, self.b]

    def prunable(self):
        return [self.W]  # biases are never pruned


class ReLU:
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []

    def prunable(self):
        return []


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def prunable(self):
        return [p for l in self.layers for p in l.prunable()]


def mlp(sizes, seed=0):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], rng))
        if i < len(sizes) - 2:
            layers.append(ReLU())
    return Sequential(*layers)
