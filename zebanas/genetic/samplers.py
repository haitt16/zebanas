import random
# from .chromosome import Chromosome
import numpy as np
import copy
from .population import Population
# from .chromosome import ChromosomeV2


class RandomPart0Part2Sampler:
    def __init__(self, bound, pop_size):
        self.bound = bound
        self.max_layers = bound.upper[0]
        self.max_lens = 1 + self.max_layers + 3
        self.pop_size = pop_size

    def __call__(self, cfg):
        pop_data = []
        while len(pop_data) < self.pop_size:
            nlayers = random.randint(self.bound.lower[0], self.bound.upper[0])
            data = [nlayers] + [1]*self.max_layers
            for lb, ub in zip(self.bound.lower[1:], self.bound.upper[1:]):
                data.append(random.randint(lb, ub))
            if data not in pop_data:
                pop_data.append(data)

        return Population.new(cfg, pop_data)


class BruteForcePart0Part2Sampler:
    def __init__(self, bound):
        self.bound = bound
        self.max_layers = bound.upper[0]

    def _generate(self, seq, i):
        if i == len(seq):
            sample = copy.deepcopy(seq)
            sample = [sample[0]] + [1]*self.max_layers + sample[1:]
            self.samples.append(sample)
            return

        for val in range(self.bound.lower[i], self.bound.upper[i]+1):
            seq[i] = val
            self._generate(seq, i+1)

    def __call__(self):
        self.samples = []
        print("Sampling")
        self._generate(copy.deepcopy(self.bound.lower), 0)
        return self.samples


class FullCellSampler:
    def __init__(
        self,
        bound,
        expand_choice
    ):
        self.part02_sampler = BruteForcePart0Part2Sampler(bound)
        self.expand_choice = expand_choice

    def _generate_part1(self, seq, i, part1_samples):
        if i == len(seq):
            sample = copy.deepcopy(seq)
            part1_samples.append(sample)
            return

        for val in self.expand_choice:
            seq[i] = val
            self._generate_part1(seq, i+1, part1_samples)

    def __call__(self):
        part02_samples = self.part02_sampler()
        self.samples = []

        for sample02 in part02_samples:
            nlayers = sample02[0]
            part1_samples = []

            base_seq = [self.expand_choice[0]]*nlayers
            self._generate_part1(copy.deepcopy(base_seq), 0, part1_samples)
            sample = copy.deepcopy(sample02)
            for sample1 in part1_samples:
                sample[1:nlayers+1] = sample1

                self.samples.append(copy.deepcopy(sample))

        # print(len(self.samples))

        # return Population.new(cfg, self.samples)
        return self.samples


class FullNetworkSampling:
    def __init__(self, n_cells, bound, expand_choice):
        self.bound = bound
        self.expand_choice = expand_choice
        self.n_cells = n_cells

    def __call__(self, pop_size):
        population = []
        for _ in range(pop_size):
            chromo = []
            for _ in range(self.n_cells):
                op_idx = np.random.randint(
                    self.bound.lower[0],
                    self.bound.upper[0]+1
                )
                nlayers = np.random.randint(
                    self.bound.lower[1],
                    self.bound.upper[1]+1
                )
                expands = random.choices(
                    self.expand_choice,
                    weights=[1, 1, 1],
                    k=nlayers
                )
                expands = expands + [1] * (self.bound.upper[1] - nlayers)

                chromo.append(
                    [op_idx, nlayers] + expands
                )

            population.append(chromo)

        return Population.new(population)
