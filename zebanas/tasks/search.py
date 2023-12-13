import os
import torch


class CellbyCellSearcher:
    def __init__(self, n_cells, select_func, backward=True):
        self.n_cells = n_cells

        if backward:
            self.search_ord = map(lambda x: n_cells-x, range(1, n_cells+1))
        else:
            self.search_ord = range(n_cells)

        self.search_ord = list(self.search_ord)
        self.select_func = select_func

    def search(
        self,
        cfg,
        algorithm,
        dataloader,
        chromosomes,
        start_step=None,
        end_step=None,
        sols_dir=None
    ):
        self.search_ord = self.search_ord[start_step:end_step]
        print(self.search_ord)
        for index in self.search_ord:
            solution = algorithm.run(
                cfg=cfg,
                dataloader=dataloader,
                chromosomes=chromosomes,
                search_index=index
            )

            # solution = self.select_func(solutions)
            chromosomes[index] = solution
            solution_ckpt = {
                "chromosomes": chromosomes,
                "solution": solution
            }
            torch.save(
                solution_ckpt,
                os.path.join(sols_dir, f"ckpt_solutions_{index}.pth")
            )

        return chromosomes
