import numpy as np
from .population import Population
import random


class SumDivSwapPart0Part2Mutation:
    def __init__(self, pm, p_sum, p_div):
        self.pm = pm
        self.p_sum = p_sum
        self.p_div = p_div
        self.p_swap = 1 - p_sum - p_div

    def _nlayer_mutate(self, data):
        pop_size = data.shape[0]
        mutate_idxs = np.random.random(pop_size) < self.pm

        X_mutate = data[mutate_idxs]
        div_idxs = np.random.random(X_mutate.shape) < 0.5
        sum_idxs = ~div_idxs

        X_mutate[div_idxs] = X_mutate[div_idxs] // 2
        X_mutate[sum_idxs] = X_mutate[sum_idxs] + 3

        data[mutate_idxs] = X_mutate
        return data

    def _ops_mutate(self, data):
        pop_size = data.shape[0]
        mutate_idxs = np.random.random(pop_size) < self.pm

        X_mutate = data[mutate_idxs]
        mchoices = np.random.choice(
            a=[0, 1, 2],
            size=X_mutate.shape,
            p=[self.p_sum, self.p_div, self.p_swap]
        )

        sum_idxs = np.where(mchoices == 0)
        div_idxs = np.where(mchoices == 1)
        swap_idxs = np.where(mchoices == 2)

        X_mutate[sum_idxs] = X_mutate[sum_idxs] + 3
        X_mutate[div_idxs] = X_mutate[div_idxs] // 2

        tmp = X_mutate[swap_idxs[0], 0]
        X_mutate[swap_idxs[0], 0] = X_mutate[swap_idxs[0], 1]
        X_mutate[swap_idxs[0], 1] = tmp

        data[mutate_idxs] = X_mutate

        return data

    def _connect_mutate(self, data):
        # pop_size = data.shape[0]
        mutate_idxs = np.random.random(data.shape) < self.pm

        X_mutate = data[mutate_idxs]
        X_mutate = 1 - X_mutate

        data[mutate_idxs] = X_mutate
        return data

    def __call__(self, cfg, population):
        X = population.get_data()
        X_tmp = X[:, 1:-5]

        off_nlayers = self._nlayer_mutate(X[:, [0]])
        off_ops = self._ops_mutate(X[:, [-5, -4]])
        off_connect = self._connect_mutate(X[:, [-3, -2, -1]])

        off = np.concatenate(
            [off_nlayers, X_tmp, off_ops, off_connect],
            axis=-1
        )
        return Population.new(cfg, off.tolist())


class Gecco2024Mutation:
    def __init__(self, max_layers, bound, expand, p):
        self.p = p
        self.bound = bound
        self.max_layers = max_layers
        self.expand = expand

    def mutate(self, samples):
        '''
        0: change operations
        1: add block
        2: delete block
        3: modify block
        '''

        if len(samples.shape) == 1:
            samples = samples[None, :]
        elif len(samples.shape) == 0:
            return samples

        size = len(samples)
        op = np.random.randint(4, size=size)

        # assert len(samples.shape) == 2, str(samples)

        for i in range(size):
            n = samples[i][1]
            if op[i] == 0:
                o = random.choice([
                    j for j in range(self.bound) if j != samples[i][0]
                ])
                samples[i][0] = o

            elif op[i] == 1 and n < self.max_layers:
                samples[i][1] += 1
                xr = random.choice(self.expand)
                samples[i][n+2] = xr
            elif op[i] == 2 and 1 < n:
                samples[i][1] -= 1
            else:
                ci = random.choice(list(range(n)))
                xr = random.choice(
                    [j for j in self.expand if j != samples[i][ci+2]]
                )
                samples[i][ci+2] = xr

        return samples

    def __call__(self, population):

        size = len(population)
        ncells = len(population[0].data)
        offspring = population.get_data().copy()

        x = np.random.random(size) < self.p
        x = np.where(x)[0]
        y = np.random.randint(ncells, size=len(x))

        samples = self.mutate(
            np.squeeze(offspring[x, y])
        )
        offspring[x, y] = samples

        return Population.new(offspring.tolist())
