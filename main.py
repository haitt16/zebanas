import os
import hydra
from hydra.utils import instantiate
import torch
# from fvcore.nn import FlopCountAnalysis

# import time

@hydra.main(version_base=None, config_path="./zebanas/configs/search", config_name="brute_force")
def main(cfg):
    if not os.path.exists(cfg.execute.sols_dir):
        os.makedirs(cfg.execute.sols_dir)

    print("Loading dataset")
    data_getter = instantiate(cfg.data)
    dataloader = data_getter.load()

    print("Initializing")
    searcher = instantiate(cfg.searcher)
    algorithm = instantiate(cfg.algorithm)

    default_chromo = instantiate(
        cfg.chromosome,
        chromo=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])
    chromosomes = [default_chromo] * (cfg.searcher.n_cells)

    print("search")
    chromosomes = searcher.search(
        cfg=cfg,
        algorithm=algorithm,
        dataloader=dataloader,
        chromosomes=chromosomes,
        start_step=cfg.execute.start_step,
        end_step=cfg.execute.end_step,
        sols_dir=cfg.execute.sols_dir
    )

    model = instantiate(cfg.model, chromos=chromosomes)
    print(
        "Solution model parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


if __name__ == "__main__":
    main()
