import numpy as np
import math
import random


class RankandCrowdingSelection:

    def __call__(self, population, n_pairs):
        if n_pairs % 2 == 1:
            n_pairs += 1

        k = math.ceil(n_pairs * 2 / len(population))
        perm = np.concatenate(
            [np.random.permutation(len(population)) for _ in range(k)]
        )
        perm = perm[: n_pairs * 2]
        P = perm.reshape(n_pairs, 2)

        S = np.full(n_pairs, np.nan)

        for i in range(n_pairs):
            a, b = P[i, 0], P[i, 1]
            dom = np.sum(population[a].obj < population[b].obj)
            if dom == 0:
                S[i] = b
            elif dom == 2:
                S[i] = a

            if np.isnan(S[i]):
                cda = population[a].cd > population[b].cd
                cdb = population[a].cd < population[b].cd
                if cda:
                    S[i] = a
                elif cdb:
                    S[i] = b
                else:
                    S[i] = random.choice([a, b])

        S = S.astype(int, copy=False)
        parents = population[S]
        return parents


class ScoreSelection:
    def __call__(self, population, n_pairs):
        k = math.ceil(n_pairs * 2 / len(population))
        perm = np.concatenate(
            [np.random.permutation(len(population)) for _ in range(k)]
        )
        perm = perm[: n_pairs * 2]
        P = perm.reshape(n_pairs, 2)

        S = np.full(n_pairs, np.nan)

        for i in range(n_pairs):
            a, b = P[i, 0], P[i, 1]

            # if np.isnan(S[i]):
            oa = population[a].obj[0] < population[b].obj[0]
            ob = population[a].obj[0] > population[b].obj[0]
            if oa:
                S[i] = a
            elif ob:
                S[i] = b
            else:
                S[i] = random.choice([a, b])

        S = S.astype(int, copy=False)
        parents = population[S]

        return parents
