import os
import hydra
from hydra.utils import instantiate
import time

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

    default_chromo = instantiate(cfg.chromosome, chromo=None)
    chromosomes = [default_chromo] * cfg.searcher.n_cells
    # chromosomes = [default_chromo] * (cfg.searcher.n_cells-1)
    # chromo1 = [5, 1, 1, 1, 1, 1, 1, 8, 9, 1, 1, 1]
    # chromo1 = instantiate(cfg.chromosome, chromo=chromo1)
    # chromosomes.append(chromo1)
    print("search")
    searcher.search(
        cfg=cfg,
        algorithm=algorithm,
        dataloader=dataloader,
        chromosomes=chromosomes,
        start_step=cfg.execute.start_step,
        end_step=cfg.execute.end_step,
        sols_dir=cfg.execute.sols_dir
    )


if __name__ == "__main__":
    main()
