import torch


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
            code_id += [
                data[-5],
                data[-4],
                data[-2]
            ]
            latency += self.data[tuple(code_id)]
        return latency

    def get_latency_from_chromosomes(self, chromosomes):
        latency = 0.
        for idx, cell in enumerate(chromosomes):
            lat = self.get_latency(cell.data, idx)
            latency += lat

        return latency

    def __call__(self, cfg, samples, search_idx):
        if search_idx < 0:
            search_idx = cfg.searcher.n_cells + search_idx
        objs = []
        for chromo in samples:
            latency = 0.

            for i in range(chromo.nlayers):
                if i > 0:
                    btype = 1
                else:
                    btype = 0

                code_id = [search_idx, btype, chromo.expands[i]]
                code_id += [
                    chromo.operations[0],
                    chromo.operations[1],
                    chromo.operations[3]
                ]

                latency += self.data[tuple(code_id)]

            objs.append(latency)

        return objs
