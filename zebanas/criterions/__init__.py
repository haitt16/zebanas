
import torch.nn as nn
import torch

from .scl import SupConLoss
from .ce import CrossEntropyCriterion


class MultitaskLossFunction(nn.Module):
    def __init__(self, num_classes, alpha=0.9):
        super().__init__()
        self.ce_loss = CrossEntropyCriterion(num_classes)
        self.scl_loss = SupConLoss()
        self.alpha = alpha

    def forward(self, emb, pred, y):
        bsz = y.size(0)
        f1, f2 = torch.split(emb, [bsz, bsz], dim=0)
        emb = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        scl = self.scl_loss(emb, y)
        y = torch.cat((y, y))
        ce = self.ce_loss(pred, y)

        loss = self.alpha * ce + (1 - self.alpha) * scl
        return {
            "loss": loss,
            "scl": scl,
            "ce": ce
        }
