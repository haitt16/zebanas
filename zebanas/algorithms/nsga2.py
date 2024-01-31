import os
import torch
import numpy as np
from tqdm import tqdm
# from pprint import pprint

from ..genetic.population import Population
from ..genetic.chromosome import ChromosomeV2
from ..genetic.utils import eliminate_duplicates, unique_populations
# from ..genetic.utils import fast_non_dominated_sorting


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
        self.history = None

    def mating(self, cfg, population):
        off = None

        while len(off) < self.pop_size:
            n_remain = self.pop_size - len(off)
            parents = self.selection(population, n_remain)
            _off = self.crossover(cfg, parents)
            _off = self.mutation(cfg, _off)
            _off = unique_populations(_off)
            _off = eliminate_duplicates(_off, [population, off])

            if len(off) + len(_off) > self.pop_size:
                n_remain = self.pop_size - len(off)
                _off = _off[:n_remain]

            if off is None:
                off = _off
            else:
                off = np.concatenate([off, _off])
            print("Number of offsprings:", len(off))

        return off

    def initialize(
        self,
        cfg,
        dataloader,
        chromosomes,
        search_index
    ):
        population = self.sampler(cfg)

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


class NSGA2_Network:
    def __init__(
        self,
        n_generations,
        pop_size,
        sampler,
        selection,
        score_evaluator,
        latency_evaluator,
        params_evaluator,
        crossover,
        mutation,
        survivor,
        device
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.params_evaluator = params_evaluator
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.sampler = sampler
        self.survivor = survivor
        self.device = device
        self.history = None
        self.gen_step = 0

    def high_latency_eliminate(self, cfg, population, eps=7e-3):
        new_population = []
        for chromo in population:
            latency = self.latency_evaluator(cfg, [chromo])
            if latency[0] <= self.latency_evaluator.bound + eps:
                new_population.append(chromo)

        new_population = Population.create(new_population)
        return new_population

    def in_history(self, chromo):
        vect = np.vectorize(ChromosomeV2.same_as)
        print(vect(chromo, self.history))
        return vect(chromo, self.history)

    def evaluate(self, cfg, population, dataloader):

        obj_list = []

        for sample in population:
            _ = self.in_history(sample)
            if np.any(self.in_history(sample)):
                print("X")
                index = np.where(self.in_history(sample))[0]
                obj = self.history[index].obj
                obj_list.append(obj)
                continue

            score = self.score_evaluator.get(
                cfg,
                sample,
                dataloader,
                self.device
            )
            latency = self.latency_evaluator.get(
                cfg,
                sample
            )
            obj_list.append([score, latency])

        return population.set_obj(obj_list)

    def mating(self, cfg, population):
        off = []

        while len(off) < self.pop_size:
            n_remain = self.pop_size - len(off)
            parents = self.selection(population, n_remain)
            _off = self.crossover(parents)
            _off = self.mutation(_off)
            _off = unique_populations(_off)
            _off = eliminate_duplicates(_off, [population, off])
            _off = self.high_latency_eliminate(cfg, _off)
            _off = self.low_paramters_eliminate(cfg, _off)

            if len(_off) % 2 != 0:
                n = 2 * (len(_off) // 2)
                _off = _off[:n]

            if len(off) + len(_off) > self.pop_size:
                n_remain = self.pop_size - len(off)
                _off = _off[:n_remain]

            if len(off) == 0:
                off = _off.copy()
            else:
                off = Population.merge([off, _off])
            print("[Number of offsprings]", len(off))

        return off

    def initialize(
        self,
        cfg,
        dataloader,
    ):
        population = []
        while len(population) < self.pop_size:
            if len(population) == 0:
                population = self.sampler(1)
                continue

            pop = self.sampler(self.pop_size)
            pop = unique_populations(pop)
            pop = eliminate_duplicates(pop, [population])
            pop = self.high_latency_eliminate(cfg, pop)
            

            if len(pop) % 2 != 0:
                n = 2 * (len(pop) // 2)
                pop = pop[:n]

            if len(population) + len(pop) > self.pop_size:
                n_remain = self.pop_size - len(population)
                pop = pop[:n_remain]

            population = Population.merge([population, pop])
            print("[Population length in sampling]", len(population))

        self.history = population.copy()

        population = self.evaluate(
            cfg=cfg,
            population=population,
            dataloader=dataloader
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
        dataloader
    ):
        offspring = self.mating(cfg, population)
        offspring = self.evaluate(
            cfg=cfg,
            population=offspring,
            dataloader=dataloader
        )

        _off = eliminate_duplicates(offspring, [self.history])
        self.history = Population.merge([self.history, _off])

        population = Population.merge([
            population, offspring
        ])
        population = self.survivor(
            population,
            n_survive=self.pop_size
        )

        return population

    def save_state_dict(self, dir, step, population):
        path = os.path.join(dir, f"state_dict_{step+1}.pth")
        state_dict = {
            "population": population,
            "step": step,
            "history": self.history
        }
        print(path)
        torch.save(state_dict, path)

    def run(
        self,
        cfg,
        dataloader,
        out_dir
    ):
        population = self.initialize(
            cfg=cfg,
            dataloader=dataloader
        )

        for gen in range(self.n_generations):
            print(f"[Step {gen+1}]")
            population = self.step(
                cfg=cfg,
                population=population,
                dataloader=dataloader
            )
            # self.log_front(population)

            if (gen+1) % 10 == 0:
                self.save_state_dict(out_dir, gen, population)

        return population


class GA_Network:
    def __init__(
        self,
        n_generations,
        pop_size,
        sampler,
        selection,
        score_evaluator,
        latency_evaluator,
        params_evaluator,
        crossover,
        mutation,
        survivor,
        device
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator
        self.params_evaluator = params_evaluator
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.sampler = sampler
        self.survivor = survivor
        self.device = device
        self.table = {}
        self.history = None
        self.gen_step = 0

    def high_latency_eliminate(self, cfg, population, eps=7e-3):
        new_population = []

        for chromo in population:
            latency = self.latency_evaluator(cfg, [chromo])
            if latency[0] <= self.latency_evaluator.bound + eps:
                new_population.append(chromo)

        new_population = Population.create(new_population)
        return new_population

    def low_paramters_eliminate(self, cfg, population):
        new_population = []
        for chromo in population:
            params = self.params_evaluator(cfg, [chromo])
            if self.params_evaluator.bound + 1_000_000 > params[0] > self.params_evaluator.bound:
                new_population.append(chromo)

        new_population = Population.create(new_population)
        return new_population

    def in_history(self, chromo):
        if self.history is None:
            return [False]
        vect = np.vectorize(ChromosomeV2.same_as)
        return vect(chromo, self.history)

    def evaluate(self, cfg, population, dataloader):

        obj_list = []
        for sample in population:
            if np.any(self.in_history(sample)):
                # print("X")
                index = np.where(self.in_history(sample))[0][0]
                obj = self.history[index].obj
                obj_list.append(obj)
                continue

            score = self.score_evaluator.get(
                cfg,
                sample,
                dataloader,
                self.device
            )
            latency = self.latency_evaluator.get(
                cfg,
                sample
            )
            obj_list.append([score, latency])
        return population.set_obj(obj_list)

    def mating(self, cfg, population, gen):
        off = []

        while len(off) < self.pop_size:
            n_remain = self.pop_size - len(off)
            parents = self.selection(population, n_remain)
            _off = self.crossover(parents)
            _off = self.mutation(_off)
            _off = unique_populations(_off)
            _off = eliminate_duplicates(_off, [population, off])
            _off = self.high_latency_eliminate(cfg, _off)
            _off = self.low_paramters_eliminate(cfg, _off)

            if len(_off) % 2 != 0:
                n = 2 * (len(_off) // 2)
                _off = _off[:n]

            if len(off) + len(_off) > self.pop_size:
                n_remain = self.pop_size - len(off)
                _off = _off[:n_remain]

            if len(off) == 0:
                off = _off.copy()
            else:
                off = Population.merge([off, _off])
            print("[Number of offsprings]", len(off))

        for i in range(len(off)):
            off[i].age = gen

        return off

    def initialize(
        self,
        cfg,
        dataloader,
    ):
        population = []
        first = False
        while len(population) < self.pop_size:
            
            if len(population) == 0:
                population = self.sampler(1)
                first = True
                continue

            pop = self.sampler(self.pop_size)
            pop = unique_populations(pop)
            pop = eliminate_duplicates(pop, [population])
            pop = self.high_latency_eliminate(cfg, pop)
            pop = self.low_paramters_eliminate(cfg, pop)

            if len(population) + len(pop) > self.pop_size:
                n_remain = self.pop_size - len(population)
                pop = pop[:n_remain]

            population = Population.merge([population, pop])
            if first:
                population = population[1:]
                first = False
            print("[Population length in sampling]", len(population))

        for i in range(len(population)):
            population[i].age = 0

        print("Evaluate")
        population = self.evaluate(
            cfg=cfg,
            population=population,
            dataloader=dataloader
        )
        self.history = population.copy()
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
        gen
    ):
        offspring = self.mating(cfg, population, gen)
        offspring = self.evaluate(
            cfg=cfg,
            population=offspring,
            dataloader=dataloader
        )
        _off = eliminate_duplicates(offspring, [self.history])
        print("[New offspring compare to history]", len(_off))
        self.history = Population.merge([self.history, _off])

        population = Population.merge([
            population, offspring
        ])
        population = self.survivor(
            population,
            n_survive=self.pop_size
        )

        return population

    def save_state_dict(self, dir, step, population):
        path = os.path.join(dir, f"state_dict_{step+1}.pth")
        state_dict = {
            "population": population,
            "step": step,
            "history": self.history
        }
        print(path)
        torch.save(state_dict, path)

    def run(
        self,
        cfg,
        dataloader,
        out_dir
    ):
        population = self.initialize(
            cfg=cfg,
            dataloader=dataloader
        )
        best_score = 0
        best_solution = None
        for gen in range(self.n_generations):
            min_score = 0
            min_solution = None
            for c in population:
                if c.obj[0] < min_score:
                    min_score = c.obj[0]
                    min_solution = c
            if min_score < best_score:
                best_score = min_score
                best_solution = min_solution
            print("[Best score in history]", best_score)
            print("[Best solution]\n", np.array(best_solution.data))
            print(f"[Step {gen+1}]")
            population = self.step(
                cfg=cfg,
                population=population,
                dataloader=dataloader,
                gen=gen
            )
            # self.log_front(population)

            if (gen+1) % 10 == 0:
                self.save_state_dict(out_dir, gen, population)

        return population
