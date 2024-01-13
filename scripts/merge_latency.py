import torch

latency_list1 = torch.load("zebanas/checkpoints/latency/latency_c10_gpu.pth")
latency_list2 = torch.load("zebanas/checkpoints/latency/latency_c10_new.pth")

new_latency_list = {}
for k, v in latency_list1.items():
    k = list(k)
    if k[-1] not in [0, 1, 2]:
        k[-1] = k[-1] - 3
        new_latency_list[tuple(k)] = v

for k, v in latency_list2.items():
    new_latency_list[k] = v

torch.save(new_latency_list, "zebanas/checkpoints/latency/c10_gpu.pth")
