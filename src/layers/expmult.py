import torch
import torch.nn as nn


class ExpMult(nn.Module):
    def __init__(self, fin):
        super(ExpMult, self).__init__()
        self.exp_weight = nn.Parameter(torch.rand(1, fin) * 4)

    def forward(self, x):
        return x * self.exp_weight.exp()
