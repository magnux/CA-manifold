import torch
import torch.nn as nn


class ExpScale(nn.Module):
    def __init__(self, fin):
        super(ExpScale, self).__init__()
        self.exp_weight = nn.Parameter(torch.rand(1, fin))

    def forward(self, x):
        return x * self.exp_weight.exp()
