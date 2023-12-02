import os
import torch
import matplotlib.pyplot as plt

path = "logs/2023-11-21/checkpoints"

pops = []
for file in os.listdir(path):
    if "pop" in file:
        pops.append(torch.load(os.path.join(path, file)))

objs = [p.get_obj() for p in pops]
print(objs)

plt.figure()
plt.scatter(objs[0][:, 0], objs[0][:, 1])
plt.show()
