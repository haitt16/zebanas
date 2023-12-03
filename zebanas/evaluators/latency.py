import torch


class CellLatencyEstimator:
    def __init__(self, path="zebanas/evaluators/latency.pth"):
        self.data = torch.load(path)

    def __call__(self, samples, search_idx):
        objs = []
        for chromo in samples:
            latency = 0.

            for i in range(chromo.nlayers):
                if i > 0:
                    btype = 1
                else:
                    btype = 0

                code_id = [search_idx, btype, chromo.expands[i]]
                code_id += chromo.operations

                latency += self.data[tuple(code_id)]

            objs.append(latency)

        return objs
