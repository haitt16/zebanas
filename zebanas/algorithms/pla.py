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
        params_bound,
        sampler,
        score_evaluator,
        latency_evaluator,
        survivor,
        eps=0.05
    ):
        self.pop_size = pop_size
        self.sampler = sampler
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.params_evaluator = ParamsCounter()
        self.latency_bound = latency_bound
        self.eps = eps
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

    def latency_pruning(self, samples, search_index):
        upper_bound, lower_bound = self.get_bound(search_index)

        latency_list = self.latency_evaluator(samples, search_index)
        latency_list = np.array(latency_list)
        indexs = []
        for i, latency in enumerate(latency_list):
            if lower_bound < latency < upper_bound:
                indexs.append(i)
        indexs = np.array(indexs)
        print(
            "[Number of samples after latency pruning]",
            len(samples)
        )

        samples = np.array(samples)[indexs]
        latency_list = latency_list[indexs]
        indexs = np.argsort(latency_list)[-self.pop_size:]

        samples = samples[indexs]

        return samples.tolist()

    def params_pruning(self, cfg, population, chromosomes, search_index):
        params_list = self.params_evaluator(
            cfg=cfg,
            samples=population,
            chromosomes=chromosomes,
            search_index=search_index
        )

        samples = []
        for params, chromo in zip(params_list, population):
            if params < self.params_bound:
                samples.append(chromo)

        print(
            "[Number of samples after params pruning]",
            len(samples)
        )
        return samples

    def run(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index
    ):
        samples = self.sampler(cfg)
        samples = self.latency_pruning(samples, search_index)

        population = Population.new(cfg, samples)
        samples = self.params_pruning(
            cfg,
            population,
            chromosomes,
            search_index
        )

        if len(samples == 0):
            return

        population = Population.create(samples)

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
