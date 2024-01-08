from tqdm import tqdm
from ..spaces.model import Network, Gecco2024Network


class ParamsCounter:
    def __init__(self, bound):
        self.bound = bound

    def get(self, cfg, sample):
        chromos = sample.data
        model = Gecco2024Network(
            chromos,
            cfg.network_channels,
            cfg.strides,
            cfg.dropout,
            cfg.num_classes,
            cfg.last_channels,
        )
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def __call__(self, cfg, population):
        params_list = []

        for chromo in population:
            params = self.get(cfg, chromo)
            params_list.append(params)
        return params_list
