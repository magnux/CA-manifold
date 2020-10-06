import torch
import torch.nn as nn
import torch.nn.functional as F


class ILRLinear(nn.Module):
    def __init__(self, fin, n_layers=16):
        super(ILRLinear, self).__init__()
        self.fin = fin
        self.block = nn.Sequential(*[nn.Linear(fin, fin, bias=False) for _ in range(n_layers)])
        self.compressed_block = None

    def forward(self, x):
        if self.training:
            self.compressed_block = None
            return self.block(x)
        else:
            if self.compressed_block is None:
                compressed_block = self.block[0].weight.t()
                for l in self.block[1:]:
                    compressed_block = compressed_block @ l.weight.t()
                self.compressed_block = compressed_block.t()
            return F.linear(x, self.compressed_block)


class ILRConv(nn.Module):
    def __init__(self, fin, n_layers=16, dim=2):
        super(ILRConv, self).__init__()
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
        self.compressed_block = None

    def forward(self, x):
        if self.training:
            self.compressed_block = None
            return self.block(x)
        else:
            if self.compressed_block is None:
                compressed_block = self.block[0].weight.view(self.fin, self.fin).t()
                for l in self.block[1:]:
                    compressed_block = compressed_block @ l.weight.view(self.fin, self.fin).t()
                self.compressed_block = compressed_block.t().view(self.fin, self.fin, 1, 1)
            return self.conv_fn(x, self.compressed_block)