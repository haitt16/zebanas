import torch


class CellLatencyEstimator:
    def __init__(self, path="zebanas/evaluators/latency.pth"):
        self.data = torch.load(path)

    def get_single(self, data, search_idx):
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

    def __call__(
            self,
            chromosomes,
            # search_idx
    ):
        result = 0.

        for cell_idx, data in enumerate(chromosomes):
            # data = chromo.data
            latency = 0.

            for i in range(data[0]):
                if i > 0:
                    btype = 1
                else:
                    btype = 0

                code_id = [cell_idx, btype, data[i+1]]
                code_id += [
                    data[-5],
                    data[-4],
                    data[-2]
                ]
                latency += self.data[tuple(code_id)]
            result += latency

        return result

    # def get_latency_from_chromosomes(self, chromosomes):
    #     latency = 0.
    #     for idx, cell in enumerate(chromosomes):
    #         lat = self(cell.data, idx)
    #         print(lat)
    #         latency += lat

    #     return latency
