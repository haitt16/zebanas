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
        survivor,
    ):
        self.pop_size = pop_size
        self.sampler = sampler
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.latency_bound = latency_bound
        self.survivor = survivor
        self.table = {}

    def chunk_population(self, population):
        chunks = []
        for i in range(0, len(population), self.pop_size):
            chunks.append(population[i:i+self.pop_size])

        return chunks

    def get_bound(self, search_index):
        upper = self.latency_bound.upper[search_index]
        lower = self.latency_bound.lower[search_index]
        return upper, lower

    def get_part02_samples(self, samples):
        part02_samples = set()
        max_layers = self.sampler.part02_sampler.max_layers
        for sample in samples:
            sample = [sample[0]] + [1]*max_layers + sample[-5:]
            part02_samples.add(tuple(sample))

        part02_samples = [list(s) for s in part02_samples]
        print("After select part02", len(part02_samples))

        return part02_samples

    def get_part1_samples(self, sample, selected_samples):
        sample_p02 = [sample[0]] + [sample[-5:]]
        part1_samples = []

        for data in selected_samples:
            data_p02 = [data[0]] + [data[-5:]]
            if data_p02 == sample_p02:
                part1_samples.append(data)

        print("After select part1", len(part1_samples))
        return part1_samples

    def run_part02(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index,
        selected_samples
    ):
        samples = self.get_part02_samples(selected_samples)
        population = Population.new(cfg, samples)

        objs = self.score_evaluator(
            cfg=cfg,
            dataloader=dataloader,
            samples=population,
            chromosomes=chromosomes,
            search_index=search_index
        )

        best_idx = np.argmin(np.array(objs))
        solution = population[best_idx]

        return solution.data

    def run_part1(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index,
        sample,
        selected_samples
    ):
        samples = self.get_part1_samples(sample, selected_samples)
        population = Population.new(cfg, samples)

        objs = self.score_evaluator(
            cfg=cfg,
            dataloader=dataloader,
            samples=population,
            chromosomes=chromosomes,
            search_index=search_index
        )

        best_idx = np.argmin(np.array(objs))
        solution = population[best_idx]

        return solution, objs[best_idx]

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

        for sample in tqdm(samples, "Pruning"):
            latency = self.latency_evaluator.get_latency(sample, search_index)
            if lower_bound < latency < upper_bound:
                selected_samples.append(sample)
        print("After selected samples", len(selected_samples))

        population = Population.new(cfg, selected_samples)

        objs = self.score_evaluator(
            cfg=cfg,
            dataloader=dataloader,
            samples=population,
            chromosomes=chromosomes,
            search_index=search_index
        )

        best_idx = np.argmin(np.array(objs))
        solution = population[best_idx]

        # solution02 = self.run_part02(
        #     cfg=cfg,
        #     dataloader=dataloader,
        #     chromosomes=chromosomes,
        #     search_index=search_index,
        #     selected_samples=selected_samples
        # )

        # solution, obj = self.run_part1(
        #     cfg=cfg,
        #     dataloader=dataloader,
        #     chromosomes=chromosomes,
        #     search_index=search_index,
        #     sample=solution02,
        #     selected_samples=selected_samples
        # )

        print(solution.data)
        print(-objs[best_idx])
        # solution = None
        return solution
