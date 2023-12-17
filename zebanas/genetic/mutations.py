import numpy as np
from .population import Population


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
