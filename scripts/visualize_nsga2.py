import os
import torch
import matplotlib.pyplot as plt

dir = "/home/haitt/workspaces/codes/nas/zebanas/logs/2023-12-26"
point_list = []

for i in range(1, 1000):
    path = os.path.join(dir, f"state_dict_{50*i+10}.pth")
    if not os.path.isfile(path):
        break

    state_dict = torch.load(path)
    population = state_dict["population"]

    point_list.append(population.get_obj())

plt.figure()
for p in point_list:
    # p = point_list[0]
    plt.scatter(p[:, 1],  p[:, 0])

plt.show()
