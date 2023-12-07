from tqdm import tqdm
from ..genetic.population import Population

import os
import torch


class BruteForce:
    def __init__(
        self,
        pop_size,
        sampler,
        evaluator,
        survivor,
    ):
        self.pop_size = pop_size
        self.sampler = sampler
        self.evaluator = evaluator
        self.survivor = survivor
        self.table = {}

    def chunk_population(self, population):
        chunks = []
        for i in range(0, len(population), self.pop_size):
            chunks.append(population[i:i+self.pop_size])

        return chunks

    def run(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index
    ):
        population = self.sampler(cfg)
        population_chunks = self.chunk_population(population)
        full_population = []

        for i, population in enumerate(tqdm(population_chunks, "Search")):
            population = self.evaluator(
                cfg=cfg,
                samples=population,
                dataloader=dataloader,
                chromosomes=chromosomes,
                search_index=search_index
            )
            torch.save(
                {"pop": population},
                os.path.join(
                    cfg.execute.sols_dir,
                    f"pop_chunk_{i}_{search_index}.pth")
            )
            full_population.append(population)

        full_population = Population.merge(full_population)
        solution = self.survivor(full_population)
        return solution
