import numpy as np
from .utils import fast_non_dominated_sorting, calc_crowding_distance


class RankandCrowdingSurvivor:
    def __call__(self, population, n_survive):
        F = population.get_obj()
        survivor_idxs = []
        fronts = fast_non_dominated_sorting(
            F,
            n_stop_if_ranked=n_survive,
            return_rank0=False
        )

        for k, front in enumerate(fronts):
            i_arange = np.arange(len(front))
            if len(survivor_idxs) + len(i_arange) > n_survive:

                n_remove = len(survivor_idxs) + len(front) - n_survive
                crowding_of_front = calc_crowding_distance(F[front, :])

                P = np.random.permutation(len(crowding_of_front))
                i_arange = np.argsort(-crowding_of_front[P], kind="quicksort")
                i_arange = P[i_arange]
                i_arange = i_arange[:-n_remove]

            else:
                crowding_of_front = calc_crowding_distance(F[front, :])

            for j, i in enumerate(front):
                population[i].rank = k
                population[i].cd = crowding_of_front[j]

            survivor_idxs.extend(front[i_arange])

        return population[survivor_idxs]


class ParetoFrontSurvivor:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, population):
        F = population.get_obj()

        front = fast_non_dominated_sorting(
            F, n_stop_if_ranked=np.inf, return_rank0=True
        )

        front_objs = F[front]
        front_objs = front_objs[front_objs[:, 0].argsort()]
        mid_obj = front_objs[0]*self.alpha + front_objs[-1]*(1 - self.alpha)
        distance = np.sqrt(np.sum(np.power(front_objs - mid_obj, 2), axis=1))
        min_idx = np.argmin(distance)

        for i in front:
            if np.all(np.array(population[i].obj) == front_objs[min_idx]):
                solution = population[i]
        print(solution.data, solution.obj)
        return solution


class AgingSurvivor:
    def __call__(self, population, n_survive):
        ages = np.array([c.age for c in population])
        indexs = np.argsort(-ages)

        survivor_idxs = indexs[:n_survive]

        return population[survivor_idxs]
