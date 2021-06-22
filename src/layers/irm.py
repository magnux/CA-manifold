import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.expscale import ExpScale


class IRMLinear(nn.Module):
    def __init__(self, fin, n_layers=4, exp_scale=False):
        super(IRMLinear, self).__init__()
        self.fin = fin
        self.exp_scale = ExpScale(fin) if exp_scale else None
        self.block = nn.Sequential(*[nn.Linear(fin, fin, bias=False) for _ in range(n_layers)])
        # for l in self.block:
        #     nn.init.normal_(l.weight, 0, 0.5 / self.fin ** 0.5)
        self.compressed_block = None

    def forward(self, x):
        if self.training:
            self.compressed_block = None
            res = self.block(x)
        else:
            if self.compressed_block is None:
                with torch.no_grad():
                    compressed_block = self.block[0].weight.t()
                    for l in self.block[1:]:
                        compressed_block = compressed_block @ l.weight.t()
                    self.compressed_block = compressed_block.t()
            res = F.linear(x, self.compressed_block)

        if self.exp_scale is not None:
            res = self.exp_scale(res)
        return res


class IRMConv(nn.Module):
    def __init__(self, fin, n_layers=4, dim=2):
        super(IRMConv, self).__init__()
        self.fin = fin
        self.dim = dim
        if self.dim == 1:
            self.conv_fn = F.conv1d
            conv_mod = nn.Conv1d
        elif self.dim == 2:
            self.conv_fn = F.conv2d
            conv_mod = nn.Conv2d
        elif self.dim == 3:
            self.conv_fn = F.conv3d
            conv_mod = nn.Conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.block = nn.Sequential(*[conv_mod(fin, fin, 1, 1, 0, bias=False) for _ in range(n_layers)])
        # for l in self.block:
        #     nn.init.normal_(l.weight, 0, 0.5 / self.fin ** 0.5)
        self.compressed_block = None

    def forward(self, x):
        if self.training:
            self.compressed_block = None
            return self.block(x)
        else:
            if self.compressed_block is None:
                with torch.no_grad():
                    compressed_block = self.block[0].weight.view(self.fin, self.fin).t()
                    for l in self.block[1:]:
                        compressed_block = compressed_block @ l.weight.view(self.fin, self.fin).t()
                    self.compressed_block = compressed_block.t().view(self.fin, self.fin, 1, 1)
            return self.conv_fn(x, self.compressed_block)
