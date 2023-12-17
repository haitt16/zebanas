import numpy as np


class Chromosome:
    def __init__(
        self,
        chromo,
        # bound,
        # unsearch_segments
    ):
        # self.max_layers = bound.upper[0]
        if chromo is None:
            chromo = self._make_default()

        if not isinstance(chromo, list):
            chromo = list(chromo)

        self.data = chromo
        # self.unsearch_segments = unsearch_segments
        # self.bound = list(zip(bound.lower, bound.upper))
        self.rank = None
        self.cd = None
        self.obj = [None, None]
        self._break_data()

    def _break_data(self):
        self.nlayers = self.data[0]
        self.expands = self.data[1:-3]
        self.operations = self.data[-3:]

    # def _back_to_bound(self):
    #     for i, (lb, ub) in enumerate(self.bound):
    #         if self.data[i] < lb:
    #             self.data[i] = lb
    #         elif self.data[i] > ub:
    #             self.data[i] = ub

    def _make_default(self):
        chromo = [1] + [1]*self.max_layers + [1]*3
        return chromo

    def numpy(self):
        return np.array(self.data)
        # self.obj = np.array(self.obj)
