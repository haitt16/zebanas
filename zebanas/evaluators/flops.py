import torch
from pthflops import count_ops
from tqdm import tqdm

from hydra.utils import instantiate


class FLOPSCounter:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(
        self,
        cfg,
        samples,
        chromosomes,
        search_index
    ):
        flops_list = []
        dummy_input = torch.rand(1, 3, self.image_size, self.image_size)

        dummy_input = dummy_input.to(cfg.execute.device)
        for chromo in tqdm(samples, "FLOPS"):
            chromosomes[search_index] = chromo
            model = instantiate(cfg.model, chromos=chromosomes)
            model = model.to(cfg.execute.device)

            flops = count_ops(
                model,
                dummy_input,
                print_readable=False,
                verbose=False
            )

            flops_list.append(flops[0])
        return flops_list
