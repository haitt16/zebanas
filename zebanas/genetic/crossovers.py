import numpy as np
from .population import Population
# import random


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


class CellBased2PointCrossover:
    def __init__(self, p):
        self.p = p

    def __call__(self, population):
        size = len(population)
        ncells = len(population[0].data)
        assert size % 2 == 0

        parent1 = population[:size//2].get_data()
        parent2 = population[size//2:].get_data()
        off1 = parent1.copy()
        off2 = parent2.copy()

        x = np.random.random(size//2) < self.p
        x = np.where(x)[0]

        if len(x) == 0:
            return population

        perm = np.concatenate([
            np.random.permutation(ncells)[None, :] for _ in range(len(x))
        ])

        # print(perm.shape)
        y = perm[:, :2].T

        for i in range(2):
            _y = y[i]
            tmp = off1[x, _y].copy()
            off1[x, _y] = off2[x, _y]
            off2[x, _y] = tmp

        offspring = np.concatenate([off1, off2])

        return Population.new(offspring.tolist())
