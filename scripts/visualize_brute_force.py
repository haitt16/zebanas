import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zebanas.genetic.population import Population
from zebanas.genetic.utils import fast_non_dominated_sorting


path = "/home/haitt/workspaces/codes/nas/zebanas/logs/2023-12-04/checkpoints"

full_population = []
for i in range(7):
    pop = torch.load(os.path.join(path, f"pop_chunk_{i}.pth"))
    full_population.append(pop["pop"])

full_population = Population.merge(full_population)

objs = full_population.get_obj()

fronts = fast_non_dominated_sorting(
    objs,
    n_stop_if_ranked=len(full_population),
    return_rank0=True
)

front_population = []
for i in fronts:
    front_population.append(full_population[i])

front_population = Population.create(front_population)
front_population = front_population.reshape(-1)
front_objs = front_population.get_obj()

# mid_obj = np.mean(front_objs, axis=0)


# dis_matrix = np.sum(np.power(front_objs - mid_obj, 2), axis=1)
# dis_matrix = np.sqrt(dis_matrix)
# min_idx = np.argmin(dis_matrix)

# print(min_idx)
# print(front_population[min_idx].data)

front_scores = front_objs[:, 0]
front_scores = (front_scores - np.min(front_scores)) / (np.max(front_scores) - np.min(front_scores))

front_latency = front_objs[:, 1]
front_latency = (front_latency - np.min(front_latency)) / (np.max(front_latency) - np.min(front_latency))

value = 0.6*front_scores + 0.4*front_latency
min_idx = np.argmin(value)
print(front_population[min_idx].data)

plt.figure()
plt.scatter(objs[:, 0], objs[:, 1])
plt.scatter(objs[fronts, 0], objs[fronts, 1])
# plt.scatter(mid_obj[0], mid_obj[1])
plt.scatter(front_objs[min_idx, 0], front_objs[min_idx, 1])
plt.show()
