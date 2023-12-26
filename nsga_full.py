import os
import hydra
from hydra.utils import instantiate
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
# torch.set_num_threads(1)


@hydra.main(config_path="./zebanas/configs/search", config_name="nsga2_full")
def main(cfg):
    if not os.path.exists(cfg.execute.sols_dir):
        os.makedirs(cfg.execute.sols_dir)

    print("Loading dataset")
    data_getter = instantiate(cfg.data)
    dataloader = data_getter.load()

    print("Initializing")
    searcher = instantiate(cfg.searcher)

    print("Search")
    solution = searcher.search(
        cfg=cfg,
        dataloader=dataloader,
        out_dir=os.path.join(cfg.execute.sols_dir, "checkpoints")
    )

    print(solution)


if __name__ == "__main__":
    main()
