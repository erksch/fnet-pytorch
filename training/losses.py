import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss


class MLMWeightedCELoss(nn.Module):
    def __init__(self):
        super(MLMWeightedCELoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets, weights):
        input = self.log_softmax(logits)
        loss = -torch.sum(targets * input, dim=-1) * weights
        loss = loss.sum() / weights.sum()
        return loss



