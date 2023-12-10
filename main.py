import os
import hydra
from hydra.utils import instantiate
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

    efficient_b0 = [
        [1, 1, 1, 1, 1, 1, 1, 6, 2, 0, 1, 0],
        [2, 6, 6, 1, 1, 1, 1, 6, 2, 0, 1, 0],
        [2, 6, 6, 1, 1, 1, 1, 7, 2, 0, 1, 0],
        [3, 6, 6, 6, 1, 1, 1, 6, 2, 0, 1, 0],
        [3, 6, 6, 6, 1, 1, 1, 7, 2, 0, 1, 0],
        [4, 6, 6, 6, 6, 1, 1, 7, 2, 0, 1, 0],
        [1, 6, 1, 1, 1, 1, 1, 6, 2, 0, 1, 0]
    ]

    chromosomes_effb0 = []
    for cell in efficient_b0:
        chromo = instantiate(
            cfg.chromosome,
            chromo=cell
        )
        chromosomes_effb0.append(chromo)

    zico = algorithm.score_evaluator.get_zico_from_chromosomes(
        cfg,
        chromosomes_effb0,
        dataloader,
        "cuda"
    )
    latency = algorithm.latency_evaluator.get_latency_from_chromosomes(
        chromosomes_effb0
    )
    print("effb0", -zico, latency)

    default_chromo = instantiate(
        cfg.chromosome,
        chromo=[1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])
    [2, 6, 6, 6, 6, 6, 6, 5, 2, 0, 1, 0]
    chromosomes = [default_chromo] * (cfg.searcher.n_cells)

    # chromosomes = chromosomes + [chromo2, chromo1]
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
