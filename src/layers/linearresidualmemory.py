import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearResidualMemory(nn.Module):
    def __init__(self, fin, n_mem=8):
        super(LinearResidualMemory, self).__init__()

        self.fin = fin
        self.n_mem = n_mem

        self.q = nn.Linear(fin, fin * n_mem)
        self.k = nn.Linear(fin, fin * n_mem)
        self.v = nn.Linear(fin, (fin + 1) * n_mem)
        self.l_out = nn.Linear((fin + 1), fin)

    def forward(self, x):
        batch_size = x.size(0)

        x_q = self.q(x).view(batch_size,  self.n_mem, self.fin)
        x_k = self.k(x).view(batch_size,  self.n_mem, self.fin).permute(0, 2, 1)
        x_v = self.v(x).view(batch_size,  self.n_mem, self.fin + 1)

        mem_x = torch.bmm(x_q, x_k)
        mem_x = torch.bmm(mem_x, x_v)
        mem_x = mem_x.sum(1)
        mem_x = F.normalize(mem_x)
        mem_x = self.l_out(mem_x)

        return x + mem_x
