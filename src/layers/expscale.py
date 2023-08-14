import torch
import torch.nn as nn


class ExpScale(nn.Module):
    def __init__(self, fin):
        super(ExpScale, self).__init__()
        self.fin = fin
        self.exp_weight = nn.Parameter(torch.rand(1, fin))

    def forward(self, x):
        exp_weight = self.exp_weight
        if x.dim() > 2:
            exp_weight = exp_weight.view([1, self.fin] + [1 for _ in range(x.dim() - 2)])
        return x * exp_weight.exp()
