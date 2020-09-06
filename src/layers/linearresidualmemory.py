import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearResidualMemory(nn.Module):
    def __init__(self, fin, n_mem=32, dropout=0.1):
        super(LinearResidualMemory, self).__init__()

        self.fin = fin
        self.n_mem = n_mem

        self.q = nn.Linear(self.fin, self.fin * self.n_mem)
        self.k = nn.Linear(self.fin, self.fin * self.n_mem)
        self.v = nn.Linear(self.fin, (self.fin + 1) * self.n_mem)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        self.l_out = nn.Linear((self.fin + 1), self.fin)

    def forward(self, x):
        batch_size = x.size(0)

        x_q = self.q(x).view(batch_size,  self.n_mem, self.fin)
        x_k = self.k(x).view(batch_size,  self.n_mem, self.fin).permute(0, 2, 1)
        x_v = self.v(x).view(batch_size,  self.n_mem, self.fin + 1)

        mem_x = torch.bmm(x_q, x_k)
        if self.dropout is not None:
            mem_x = self.dropout(mem_x)
        mem_x = torch.bmm(mem_x, x_v)
        mem_x = mem_x.sum(1)
        mem_x = F.normalize(mem_x)
        mem_x = self.l_out(mem_x)

        return x + mem_x
