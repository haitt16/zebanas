import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zebanas.genetic.population import Population
from zebanas.genetic.utils import fast_non_dominated_sorting


path = "/home/haitt/workspaces/codes/nas/zebanas/logs/2023-12-05"

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
    # if full_population[i].data[7] not in [0, 1, 2]:
    #     print(full_population[i].data, full_population[i].obj)

for chromo in full_population:
    print(chromo.data, chromo.obj)

front_objs = objs[fronts]
front_objs = front_objs[front_objs[:, 0].argsort()]
mid_obj = front_objs[0]*0.3 + front_objs[-1]*0.7

distance = np.sqrt(np.sum(np.power(front_objs - mid_obj, 2), axis=1))
print(distance.shape)

min_idx = np.argmin(distance)
for i in fronts:
    if np.all(np.array(full_population[i].obj) == front_objs[min_idx]):
        print(full_population[i].data, full_population[i].obj)

plt.figure()
plt.scatter(objs[:, 0], objs[:, 1])
plt.scatter(objs[fronts, 0], objs[fronts, 1])
plt.scatter(mid_obj[0], mid_obj[1])
plt.scatter(front_objs[min_idx, 0], front_objs[min_idx, 1])
plt.show()
