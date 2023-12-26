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


class ChromosomeV2:
    def __init__(
        self,
        chromo,
    ):

        if not isinstance(chromo, list):
            if isinstance(chromo, np.ndarray):
                raise ValueError("Chromo data must be list")
            chromo = list(chromo)

        self.data = chromo
        self.rank = None
        self.cd = None
        self.obj = [None, None]

    @classmethod
    def same_as(cls, chromo1, chromo2):
        same = True
        for cell1, cell2 in zip(chromo1.data, chromo2.data):
            _s = cell1[0:2] == cell2[0:2]
            n = cell1[1]
            _s = _s and (cell1[2:n+2] == cell2[2:n+2])
            same = same and _s
        return same
