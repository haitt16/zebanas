from tqdm import tqdm
from hydra.utils import instantiate


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

        for chromo in tqdm(samples, "Params"):
            chromosomes[search_index] = chromo
            model = instantiate(cfg.model, chromos=chromosomes)

            params = self.count_params(model)
            params_list.append(params)
        return params_list
