import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.pos_encoding import PosEncoding


class ResidualMemory(nn.Module):
    def __init__(self, size, fin, n_mem=8, dim=1, dropout=0.1):
        super(ResidualMemory, self).__init__()

        self.fin = fin
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

        self.q = conv_fn(self.fin + self.pos_encoding.size(), self.n_mem * self.fin, 1, 1, 0)
        self.k = conv_fn(self.fin + self.pos_encoding.size(), self.n_mem * self.fin, 1, 1, 0)
        self.v = conv_fn(self.fin + self.pos_encoding.size(), self.n_mem * (self.fin + 1), 1, 1, 0)
        self.temp = self.fin ** 0.5
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.conv_out = conv_fn(self.fin + 1, self.fin, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)

        x_pos = self.pos_encoding(x)

        x_q = self.q(x_pos).view(batch_size,  self.n_mem, self.fin, -1).view(batch_size * self.n_mem, self.fin, -1).contiguous()
        x_k = self.k(x_pos).view(batch_size,  self.n_mem, self.fin, -1).view(batch_size * self.n_mem, self.fin, -1).contiguous().permute(0, 2, 1)
        x_v = self.v(x_pos).view(batch_size,  self.n_mem, self.fin, -1).view(batch_size * self.n_mem, self.fin, -1).contiguous()

        mem_x = torch.bmm(x_q, x_k)
        if self.dropout is not None:
            mem_x = self.dropout(mem_x)
        mem_x = torch.bmm(mem_x, x_v)

        mem_x = mem_x.permute(0, 2, 1).view(batch_size, self.n_mem, self.fin + 1, -1).contiguous()
        if self.dim == 2:
            mem_x = mem_x.view(batch_size, self.n_mem, (self.fin + 1), x.size(2), x.size(3))
        elif self.dim == 3:
            mem_x = mem_x.view(batch_size, self.n_mem, (self.fin + 1), x.size(2), x.size(3), x.size(4))

        mem_x = mem_x.mean(1)
        mem_x = self.conv_out(mem_x)

        return x + mem_x
