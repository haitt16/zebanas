import numpy as np

import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

from hydra.utils import instantiate


class ZicoProxy:
    def __init__(
        self,
        loss_fn
    ):
        self.loss_fn = loss_fn

    def get_grad(self, grad_dict, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.weight.grad is None:
                    continue
                grad_dict[name].append(
                    module.weight.grad.data.cpu().reshape(-1).numpy()
                )

    def calc_zico(self, grad_dict):
        for name in grad_dict.keys():
            grad_dict[name] = np.array(grad_dict[name])

        score = 0
        for value in grad_dict.values():
            nsr_std = np.std(value, axis=0)
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(value), axis=0)
            tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])

            if tmpsum == 0:
                pass
            else:
                score += np.log(tmpsum)

        return -score

    def get_zico(self, model, dataloader, device):
        grad_dict = defaultdict(list)
        model.train()

        model.to(device)
        for x, y in dataloader:
            model.zero_grad()
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = self.loss_fn(y_pred, y)

            loss.backward()

            self.get_grad(grad_dict, model)

        return self.calc_zico(grad_dict)

    def __call__(
        self,
        cfg,
        dataloader,
        samples,
        chromosomes,
        search_index
    ):
        objs = []

        for chromo in tqdm(samples, "Evaluating"):
            chromosomes[search_index] = chromo
            model = instantiate(cfg.model, chromos=chromosomes)

            score = self.get_zico(
                model,
                dataloader,
                cfg.execute.device
            )
            objs.append(score)

        return objs
