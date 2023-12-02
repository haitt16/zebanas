import numpy as np
from .population import Population


class AverageSwapPart0Part2Crossover:
    def __init__(self, px, p_swap):
        self.px = px
        self.p_swap = p_swap

    def __call__(self, cfg, population):
        X = population.get_data()
        X_tmp = X[:, 1:-5]
        X = X[:, [0, -5, -4, -3, -2, -1]]
        dim = X.shape[-1]
        X = X.reshape(-1, dim, 2)
        off = X.copy()

        # X_avg = np.mean(X, axis=-1)
        n_pairs = X.shape[0]

        cross_idxs = np.random.random(n_pairs) < self.px

        X_cross = X[cross_idxs]
        X_cross_avg = np.mean(X_cross, axis=-1)
        off_cross = off[cross_idxs]
        swap_idxs = np.random.random(X_cross.shape) < self.p_swap
        swap_idxs = np.all(swap_idxs, axis=-1)
        avg_idxs = ~swap_idxs

        off_cross[swap_idxs, 0] = X_cross[swap_idxs, 1]
        off_cross[swap_idxs, 1] = X_cross[swap_idxs, 0]
        off_cross[avg_idxs, 0] = X_cross_avg[avg_idxs]
        off_cross[avg_idxs, 1] = X_cross_avg[avg_idxs]

        off[cross_idxs] = off_cross
        off = off.reshape(-1, dim)
        off = np.concatenate([off[:, [0]], X_tmp, off[:, 1:]], axis=-1)

        return Population.new(cfg, off.tolist())
