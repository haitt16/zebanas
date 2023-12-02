import numpy as np
# from scipy.spatial import distance

from .population import Population


def calc_domination_matrix(A, B):
    n = A.shape[0]
    m = B.shape[0]

    L = np.repeat(A, m, axis=0)
    R = np.tile(B, (n, 1))

    smaller = np.any(L < R, axis=1).reshape(n, m)
    larger = np.any(L > R, axis=1).reshape(n, m)

    M = (~larger & smaller) * 1 + (~smaller & larger) * -1
    return M


def unique_populations(pop):
    new_pop = []
    for chromo in pop:
        if chromo not in new_pop:
            new_pop.append(chromo)

    new_pop = Population.create(new_pop)
    len_pop = 2*(len(new_pop)//2)
    return new_pop[:int(len_pop)]


def eliminate_duplicates(pop, pop_list):
    new_pop = []
    prev_pop = Population.merge(pop_list)
    for chromo in pop:
        if chromo not in prev_pop:
            new_pop.append(chromo)

    new_pop = Population.create(new_pop)
    len_pop = 2*(len(new_pop)//2)
    return new_pop[:int(len_pop)]


# def eliminate_duplicates(pop, pop_list):
#     if len(pop) == 0:
#         return pop
#     prev_pop = Population.merge(pop_list)
#     D = distance.cdist(pop.get_data(), prev_pop.get_data())
#     if len(pop_list) == 1 and pop_list[0].shape == pop.shape:
#         if np.all(pop_list[0] == pop):
#             D[np.triu_indices(len(pop))] = np.inf

#     D[np.isnan(D)] = np.inf
#     is_duplicate = np.full(len(pop), False)
#     is_duplicate[np.any(D <= 1e-16, axis=1)] = True

#     pop = pop[~is_duplicate]
#     len_pop = 2*(len(pop)//2)
#     return pop[:int(len_pop)]


def fast_non_dominated_sorting(
    objs,
    n_stop_if_ranked,
    return_rank0=False
):
    M = calc_domination_matrix(objs, objs)
    n = M.shape[0]
    fronts = []

    if n == 0:
        return fronts

    n_ranked = 0
    ranked = np.zeros(n)

    is_dominating = [[] for _ in range(n)]
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):
        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    fronts.append(current_front)
    while n_ranked < n:
        next_front = []

        for i in current_front:
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    _fronts = []
    n_ranked = 0
    for front in fronts:
        _fronts.append(np.array(front, dtype=int))
        n_ranked += len(front)
        if n_ranked >= n_stop_if_ranked:
            break

    if return_rank0:
        return _fronts[0]
    return _fronts


def calc_crowding_distance(objs):
    n_obj = objs.shape[1]
    sorted_objs = np.argsort(objs, axis=0, kind="mergesort")
    objs = objs[sorted_objs, np.arange(n_obj)]

    dist0 = np.row_stack([objs, np.full(n_obj, np.inf)])
    dist1 = np.row_stack([np.full(n_obj, -np.inf), objs])
    dist = dist0 - dist1

    norm = np.max(objs, axis=0) - np.min(objs, axis=0)
    norm[norm == 0] = np.nan

    dist_to_last, dist_to_next = dist, dist.copy()
    dist_to_last = dist_to_last[:-1] / norm
    dist_to_next = dist_to_next[1:] / norm

    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    J = np.argsort(sorted_objs, axis=0)

    dist_to_last = dist_to_last[J, np.arange(n_obj)]
    dist_to_next = dist_to_next[J, np.arange(n_obj)]
    cd = np.sum(dist_to_last + dist_to_next, axis=1) / n_obj

    return cd
