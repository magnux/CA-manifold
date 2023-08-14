import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IRMLinear(nn.Module):
    def __init__(self, fin, n_layers=4):
        super(IRMLinear, self).__init__()
        self.fin = fin
        self.block = nn.Sequential(*[nn.Linear(fin, fin, bias=False) for _ in range(n_layers)])
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
                    compressed_block = self.block[0].weight.t()
                    for l in self.block[1:]:
                        compressed_block = compressed_block @ l.weight.t()
                    self.compressed_block = compressed_block.t()
            return F.linear(x, self.compressed_block)


class IRMConv(nn.Module):
    def __init__(self, fin, fout, n_layers=4, dim=2):
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

        self.f_size = np.linspace(fin, fout, n_layers + 1).astype(int).tolist()

        self.block = nn.Sequential(*[conv_mod(self.f_size[l], self.f_size[l + 1], 3, 1, 1, bias=False) for l in range(n_layers)])
        # for l in self.block:
        #     nn.init.normal_(l.weight, 0, 0.5 / self.fin ** 0.5)
        self.compressed_block = None

    def forward(self, x):
        return self.block(x)
