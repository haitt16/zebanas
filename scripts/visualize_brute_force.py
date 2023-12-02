import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import torch
from zebanas.genetic.population import Population
from zebanas.genetic.utils import fast_non_dominated_sorting
import numpy as np
from kneed import KneeLocator
path = "/home/haitt/workspaces/codes/nas/zebanas/logs/2023-11-25-1/checkpoints"

populations = []
for i in range(7):
    f = os.path.join(path, f"pop_chunk_{i}.pth")
    pop = torch.load(f)
    # print(pop)
    populations.append(pop["pop"])

full_pop = Population.merge(populations)

table = {}
for chromo in full_pop:
    key = tuple(chromo.data)
    table[key] = chromo.obj

torch.save(table, "/home/haitt/workspaces/codes/nas/zebanas/logs/2023-11-22/checkpoints/table.pth")

objs = full_pop.get_obj()
# objs[:, 0] = -objs[:, 0]

nobjs = objs.copy()
nobjs = []
for s, l in objs:
    if l < 0.01 and s != 0:
        nobjs.append([s, l])

nobjs = np.array(nobjs)
front = fast_non_dominated_sorting(
    nobjs, n_stop_if_ranked=np.inf, return_rank0=True
)

all_front = fast_non_dominated_sorting(
    objs, n_stop_if_ranked=np.inf, return_rank0=True
)

obj_front = nobjs[front]
obj_front = sorted(obj_front, key=lambda x: x[0])
obj_front = np.array(obj_front)

mid = obj_front[0]/2 + obj_front[-1]/2

min_i = None
min_val = 1e10

for idx, p in enumerate(obj_front):
    dis = np.sqrt(np.sum((p - mid)*(p - mid)))
    if dis < min_val:
        min_val = dis
        min_i = idx

mid_obj = obj_front[min_i]
print(mid_obj)
optimal_idx = np.where(objs == mid_obj)[0][0]
print(optimal_idx)
print(full_pop[optimal_idx].data)
print(all_front)

for fid in all_front:
    print(full_pop[fid].data)
    print(full_pop[fid].obj)

plt.figure()
# plt.scatter(nobjs[:, 0], nobjs[:, 1])
plt.scatter(nobjs[:, 0], nobjs[:, 1])
plt.scatter(obj_front[:, 0], obj_front[:, 1])
plt.show()
