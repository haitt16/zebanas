import numpy as np

from tqdm import tqdm
from ..genetic.population import Population
from ..evaluators.params import ParamsCounter

# import os
# import torch


class PruningLatencyAware:
    def __init__(
        self,
        pop_size,
        latency_bound,
        latency_eps,
        params_bound,
        sampler,
        score_evaluator,
        latency_evaluator,
        survivor,
    ):
        self.pop_size = pop_size
        self.sampler = sampler
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.params_evaluator = ParamsCounter()
        self.latency_bound = latency_bound
        self.latency_eps = latency_eps
        self.params_bound = params_bound
        self.survivor = survivor
        self.table = {}

        print("Latency bound:", latency_bound)

    def chunk_population(self, population):
        chunks = []
        for i in range(0, len(population), self.pop_size):
            chunks.append(population[i:i+self.pop_size])

        return chunks

    def get_bound(self, index):
        upper = self.latency_bound[index] + self.latency_eps[index]
        lower = self.latency_bound[index] - self.latency_eps[index]
        return upper, lower

    def run(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index
    ):
        samples = self.sampler(cfg)
        upper_bound, lower_bound = self.get_bound(search_index)

        selected_samples = []
        latency_list = []

        for sample in tqdm(samples, "Latency pruning"):
            latency = self.latency_evaluator.get_single(sample, search_index)
            if lower_bound < latency < upper_bound:
                selected_samples.append(sample)
                latency_list.append(latency)

        print(
            "[Number of samples after latency pruning]",
            len(selected_samples)
        )
        population = Population.new(cfg, selected_samples)

        params_list = self.params_evaluator(
            cfg=cfg,
            samples=population,
            chromosomes=chromosomes,
            search_index=search_index
        )

        selected_samples = []
        for params, chromo in zip(params_list, population):
            if params < self.params_bound:
                selected_samples.append(chromo)
        print(
            "[Number of samples after params pruning]",
            len(selected_samples)
        )
        population = Population.create(selected_samples)

        objs = self.score_evaluator(
            cfg=cfg,
            dataloader=dataloader,
            samples=population,
            chromosomes=chromosomes,
            search_index=search_index
        )

        best_idx = np.argmin(np.array(objs))
        solution = population[best_idx]

        print(solution.data)
        print(-objs[best_idx])
        # solution = None
        return solution
