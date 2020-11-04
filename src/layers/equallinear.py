import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    def __init__(self, fin, fout, lr_mul=0.1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin))
        if bias:
            self.bias = nn.Parameter(torch.zeros(fout))

        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
