import os
import torch

from tqdm import tqdm

from ..genetic.population import Population
from ..genetic.utils import eliminate_duplicates


class NSGA2:
    def __init__(
        self,
        n_generations,
        pop_size,
        sampler,
        selection,
        evaluator,
        crossover,
        mutation,
        survivor
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.evaluator = evaluator
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.sampler = sampler
        self.survivor = survivor
        self.table = {}
        self.history = set()

    def mating(self, cfg, population):
        off = Population([])

        while len(off) < self.pop_size:
            n_remain = self.pop_size - len(off)
            parents = self.selection(population, n_remain)
            _off = self.crossover(cfg, parents)
            _off = eliminate_duplicates(_off, [_off])
            _off = eliminate_duplicates(_off, [population, off])

            if len(off) + len(_off) > self.pop_size:
                n_remain = self.pop_size - len(off)
                _off = _off[:n_remain]
            if len(_off) == 0:
                continue
            off = Population.merge([off, _off])
            print("Number of offsprings:", len(off))
            if len(off) >= self.pop_size:
                break

            n_remain = self.pop_size - len(off)
            _off = self.mutation(cfg, _off)
            _off = eliminate_duplicates(_off, [_off])
            _off = eliminate_duplicates(_off, [population, off])

            if len(off) + len(_off) > self.pop_size:
                n_remain = self.pop_size - len(off)
                _off = _off[:n_remain]

            off = Population.merge([off, _off])
            print("Number of offsprings:", len(off))

        for ch in off:
            if tuple(ch.data) not in self.history:
                self.history.add(tuple(ch.data))
        print("history length", len(self.history))
        return off

    def initialize(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index
    ):
        population = self.sampler(cfg)
        for ch in population:
            if tuple(ch.data) not in self.history:
                self.history.add(tuple(ch.data))

        population = self.evaluator(
            cfg=cfg,
            samples=population,
            dataloader=dataloader,
            chromosomes=chromosomes,
            search_index=search_index
        )
        population = self.survivor(
            population,
            n_survive=self.pop_size
        )
        return population

    def step(
        self,
        cfg,
        population,
        dataloader,
        chromosomes,
        search_index
    ):
        offspring = self.mating(cfg, population)
        offspring = self.evaluator(
            cfg=cfg,
            samples=population,
            dataloader=dataloader,
            chromosomes=chromosomes,
            search_index=search_index
        )

        population = Population.merge([
            population, offspring
        ])
        population = self.survivor(
            population,
            n_survive=self.pop_size
        )

        return population

    def run(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index
    ):
        population = self.initialize(
            cfg=cfg,
            dataloader=dataloader,
            chromosomes=chromosomes,
            search_index=search_index
        )
        torch.save(
            {"pop": population},
            os.path.join(
                cfg.execute.sols_dir,
                "population_init.pth"
            )
        )

        for gen in tqdm(range(self.n_generations)):
            population = self.step(
                cfg=cfg,
                population=population,
                dataloader=dataloader,
                chromosomes=chromosomes,
                search_index=search_index
            )

            if (gen+1) % cfg.execute.save_interval == 0:
                torch.save(
                    {"pop": population},
                    os.path.join(cfg.execute.sols_dir, f"population_{gen}.pth")
                )

        return population
