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
        params_bound,
        score_evaluator,
        latency_evaluator,
        sampler,
        eps=0.05
    ):
        self.pop_size = pop_size
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.params_evaluator = ParamsCounter()
        self.eps = eps
        self.params_bound = params_bound
        self.samples = sampler()

    def get_bound(self, latency, index):
        upper = latency[index] + self.eps
        lower = latency[index] - self.eps
        return upper, lower

    def latency_pruning(self, samples, latency, search_index):
        upper_bound, lower_bound = self.get_bound(latency, search_index)

        latency_list = self.latency_evaluator(samples, search_index)
        latency_list = np.array(latency_list)
        indexs = []
        for i, lat in enumerate(latency_list):
            if lower_bound < lat < upper_bound:
                indexs.append(i)
        if len(indexs) == 0:
            return []
        indexs = np.array(indexs)

        samples = np.array(samples)[indexs]
        latency_list = latency_list[indexs]

        print(
            "[Number of samples after latency pruning]",
            len(samples)
        )
        indexs = np.argsort(np.absolute(
            latency_list - latency[search_index]
        ))[:self.pop_size]

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
        search_index,
        latency
    ):
        # samples = self.sampler(cfg)
        samples = self.latency_pruning(self.samples, latency, search_index)
        if len(samples) == 0:
            return

        population = Population.new(cfg, samples)
        samples = self.params_pruning(
            cfg,
            population,
            chromosomes,
            search_index
        )

        if len(samples) == 0:
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

        # print(solution.data)
        # print(-objs[best_idx])
        # solution = None
        return solution


class DEforLatencyAwarePruningGlobal:
    def __init__(
        self,
        evaluator,
        hparams
    ):
        self.evaluator = evaluator
        self.F = hparams.F
        self.CR = hparams.CR
        self.pop_size = hparams.pop_size
        self.n_gens = hparams.n_gens
        self.max_latency = hparams.max_latency
        self.dimension = hparams.n_cells
        self.max_score = hparams.max_score

    def sample(self):
        samples = np.random.random((self.pop_size, self.dimension))
        samples = self.max_latency * samples

        return samples

    def mating(self, population):
        indexs = np.random.randint(
            low=0,
            high=self.pop_size,
            size=(self.pop_size, 3)
        )

        k = [population[indexs[:, i]] for i in range(3)]
        v = k[2] + self.F * (k[1] - k[0])

        off = np.zeros(shape=population.shape)
        indexs = np.random.random(population.shape) < self.CR

        off[indexs] = v[indexs]
        off[~indexs] = population[~indexs]

        indexs = np.random.randint(
            self.dimension,
            size=self.pop_size
        )
        indexs = np.eye(self.dimension)[indexs].astype(bool)
        off[indexs] = v[indexs]

        return off

    def evaluate(
        self,
        cfg,
        algorithm,
        dataloader,
        chromosomes,
        latencies
    ):
        print("[DE Evaluating]")
        score_list = []
        for c in tqdm(latencies):
            if np.sum(c) > self.max_latency:
                score = 0.
            else:
                score = self.evaluator(
                    cfg=cfg,
                    algorithm=algorithm,
                    dataloader=dataloader,
                    chromosomes=chromosomes,
                    latency=c
                )

            score_list.append(score)
        return np.array(score_list)

    def run(
        self,
        cfg,
        algorithm,
        dataloader,
        chromosomes,
    ):
        population = self.sample()

        print("[Evaluate initial fitness]", )
        parent_scores = self.evaluate(
            cfg,
            algorithm,
            dataloader,
            chromosomes,
            population
        )

        for gen_idx in range(self.n_gens):
            print(f"[Generation {gen_idx}]")

            off = self.mating(population)

            off_scores = self.evaluate(
                cfg,
                algorithm,
                dataloader,
                chromosomes,
                off
            )

            indexs = off_scores < parent_scores
            parent_scores[indexs] = off_scores[indexs]
            population[indexs] = off[indexs]

            best_score = np.min(parent_scores)

            print("[Minimal fitness: {}]".format(-best_score))
            print("[Minimal solution: {}]".format(
                population[np.argmin(parent_scores)]
            ))

            if best_score < self.max_score:
                print("[Early Stopping]")
                break
