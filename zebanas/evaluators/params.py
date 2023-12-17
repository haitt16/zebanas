from tqdm import tqdm
from ..spaces.model import Network


class ParamsCounter:
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def __call__(
        self,
        cfg,
        samples,
        chromosomes,
        search_index
    ):
        params_list = []

        model_hparams = cfg.model

        for chromo in samples:
            chromosomes[search_index] = chromo
            model = Network(
                chromosomes,
                model_hparams.network_channels,
                model_hparams.strides,
                model_hparams.dropout,
                model_hparams.num_classes,
                model_hparams.last_channels,
            )

            params = self.count_params(model)
            params_list.append(params)
        return params_list
