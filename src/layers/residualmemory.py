import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.posencoding import PosEncoding
import numpy as np


class ResidualMemory(nn.Module):
    def __init__(self, size, fin, n_mem=64, dim=1, dropout=0.1):
        super(ResidualMemory, self).__init__()

        self.fin = fin
        self.sqrt_fin = int(self.fin ** 0.5)
        self.n_mem = n_mem
        self.dim = dim

        if self.dim == 1:
            conv_fn = nn.Conv1d
        elif self.dim == 2:
            conv_fn = nn.Conv2d
        elif self.dim == 3:
            conv_fn = nn.Conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.pos_encoding = PosEncoding(size, dim)

        self.q = conv_fn(self.fin + self.pos_encoding.size(), self.n_mem * self.sqrt_fin, 1, 1, 0)
        self.k = conv_fn(self.fin + self.pos_encoding.size(), self.n_mem * self.sqrt_fin, 1, 1, 0)
        self.v = nn.Parameter(nn.init.orthogonal_(torch.empty(self.n_mem, self.fin)).unsqueeze(0))

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.conv_out = conv_fn(self.fin, self.fin, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)

        x_pos = self.pos_encoding(x)

        x_q = self.q(x_pos).view(batch_size,  self.n_mem, self.sqrt_fin, -1).permute(0, 3, 1, 2).contiguous().view(-1, self.n_mem, self.sqrt_fin)
        x_k = self.k(x_pos).view(batch_size,  self.n_mem, self.sqrt_fin, -1).permute(0, 3, 2, 1).contiguous().view(-1, self.sqrt_fin, self.n_mem)
        x_v = torch.cat([self.v] * x_q.size(0), 0)

        mem_x = torch.bmm(F.normalize(x_q, dim=2), F.normalize(x_k, dim=1))
        if self.dropout is not None:
            mem_x = self.dropout(mem_x)
        mem_x = torch.bmm(mem_x, x_v)

        mem_x = mem_x.view(batch_size, np.prod(x.size()[2:]), self.n_mem, self.fin).permute(0, 2, 3, 1).contiguous()
        if self.dim == 2:
            mem_x = mem_x.view(batch_size, self.n_mem, self.fin, x.size(2), x.size(3))
        elif self.dim == 3:
            mem_x = mem_x.view(batch_size, self.n_mem, self.fin, x.size(2), x.size(3), x.size(4))

        mem_x = mem_x.mean(1)
        mem_x = self.conv_out(mem_x)

        return x + mem_x
