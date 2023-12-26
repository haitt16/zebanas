import torch
# from tqdm import tqdm


class CellLatencyEstimator:
    def __init__(self, path="zebanas/evaluators/latency.pth"):
        self.data = torch.load(path)

    def get_latency(self, data, search_idx):
        latency = 0.

        for i in range(data[0]):
            if i > 0:
                btype = 1
            else:
                btype = 0

            code_id = [search_idx, btype, data[i+1]]
            code_id += data[-3:]
            latency += self.data[tuple(code_id)]

        return latency

    def __call__(
        self,
        samples,
        search_idx
    ):
        latency_list = []

        for data in samples:
            latency = self.get_latency(data, search_idx)
            latency_list.append(latency)

        return latency_list


class NetworkLatencyEstimator:
    def __init__(self, path, bound):
        self.latency_table = torch.load(path)
        self.latency_table = {
            k: v["mean"] for k, v in self.latency_table.items()
        }
        self.bound = bound

    def get_latency(self, cfg, chromos):
        latency = 0.
        for i in range(len(cfg.network_channels[:-1])):
            cell_chromo = chromos.data[i]
            n = cell_chromo[1]

            for j in range(n):
                id = (i, min(1, j), cell_chromo[j+2], cell_chromo[0])
                latency += self.latency_table[id]

        return latency

    def __call__(self, cfg, population):
        latency_list = []

        for chromo in population:
            latency = self.get_latency(cfg, chromo)
            latency_list.append(latency)

        return latency_list
