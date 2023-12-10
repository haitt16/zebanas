import numpy as np

from tqdm import tqdm
from ..genetic.population import Population

# import os
# import torch


class PruningLatencyAware:
    def __init__(
        self,
        pop_size,
        latency_bound,
        sampler,
        score_evaluator,
        latency_evaluator,
        complexity_evaluator,
        survivor,
    ):
        self.pop_size = pop_size
        self.sampler = sampler
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.complexity_evaluator = complexity_evaluator
        self.latency_bound = latency_bound
        self.survivor = survivor
        self.table = {}

        print("Latency bound:", latency_bound)

    def chunk_population(self, population):
        chunks = []
        for i in range(0, len(population), self.pop_size):
            chunks.append(population[i:i+self.pop_size])

        return chunks

    def get_bound(self, search_index):
        upper = self.latency_bound.upper[search_index]
        lower = self.latency_bound.lower[search_index]
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
        if search_index < 0:
            search_index = cfg.searcher.n_cells + search_index

        selected_samples = []
        latency_list = []

        for sample in tqdm(samples, "Pruning"):
            latency = self.latency_evaluator.get_latency(sample, search_index)
            if lower_bound < latency < upper_bound:
                selected_samples.append(sample)
                latency_list.append(latency)

        print("After selected samples", len(selected_samples))

        # sindex = np.argsort(latency_list)[-self.pop_size*5:]
        # selected_samples = np.array(selected_samples)[sindex]
        population = Population.new(cfg, selected_samples)

        # cscores = self.complexity_evaluator(
        #     cfg=cfg,
        #     samples=population,
        #     chromosomes=chromosomes,
        #     search_index=search_index
        # )

        # flops_indexes = np.argsort(cscores)[-self.pop_size:]
        # population = population[flops_indexes]

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
