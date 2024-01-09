import torch


class Accuracy:
    def __call__(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        bs = targets.size(0)
        return torch.sum(preds[:bs] == targets) / preds.size(0)
