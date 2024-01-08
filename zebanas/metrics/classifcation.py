import torch


class Accuracy:
    def __call__(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        return torch.sum(preds == targets) / preds.size(0)
