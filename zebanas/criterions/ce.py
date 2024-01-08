import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyCriterion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        targets = F.one_hot(targets, num_classes=self.num_classes)
        return self.ce(logits, targets.float())
