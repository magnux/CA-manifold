import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    def __init__(self, fin, fout, lr_mul=0.1, bias=True):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin))
        if bias:
            self.bias = nn.Parameter(torch.zeros(fout))
        else:
            self.bias = None

        self.lr_mul = lr_mul

    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        else:
            return F.linear(x, self.weight * self.lr_mul)


class EqualLinearBlock(nn.Module):
    def __init__(self, fin, fout, n_layers, lr_mul=0.1, bias=True):
        super(EqualLinearBlock, self).__init__()
        l_block = []
        for _ in range(n_layers):
            l_block.extend([EqualLinear(fin, fin, lr_mul, bias), nn.LeakyReLU(0.2, True)])
        l_block.append(EqualLinear(fin, fout, lr_mul, bias))
        self.l_block = nn.Sequential(*l_block)

    def forward(self, x):
        return self.l_block(x)
