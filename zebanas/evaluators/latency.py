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
