import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearResidualMemory(nn.Module):
    def __init__(self, fin, n_mem=128, dropout=0.1):
        super(LinearResidualMemory, self).__init__()

        self.fin = fin
        self.sqrt_fin = int(self.fin ** 0.5)
        self.n_mem = n_mem

        self.q = nn.Linear(self.fin, self.sqrt_fin * self.n_mem)
        self.k = nn.Linear(self.fin, self.sqrt_fin * self.n_mem)
        self.v = nn.Parameter(nn.init.orthogonal_(torch.empty(self.n_mem, self.fin)).unsqueeze(0))

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        self.l_out = nn.Linear((self.fin), self.fin)

    def forward(self, x):
        batch_size = x.size(0)

        x_q = self.q(x).view(batch_size,  self.n_mem, self.sqrt_fin)
        x_k = self.k(x).view(batch_size,  self.n_mem, self.sqrt_fin).permute(0, 2, 1)
        x_v = torch.cat([self.v] * batch_size, 0)

        mem_x = torch.bmm(F.normalize(x_q, dim=2), F.normalize(x_k, dim=1))
        if self.dropout is not None:
            mem_x = self.dropout(mem_x)
        mem_x = torch.bmm(mem_x, x_v)
        mem_x = mem_x.mean(1)
        mem_x = self.l_out(mem_x)

        return x + mem_x
