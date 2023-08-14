import torch.nn as nn
import torch.nn.functional as F


class LinearResidualBlockS(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bias=True, memory=False, mem_factor=1):
        super(LinearResidualBlockS, self).__init__()

        self.fin = fin
        self.fout = fout
        self.fhidden = max(fin + fout, 1) if fhidden is None else fhidden
        self.memory = memory
        self.mem_factor = mem_factor

        if self.memory:
            self.x_to_mem = nn.Linear(self.fin, int(self.fin * self.mem_factor))
            self.mem_to_x = nn.Linear(int(self.fin * self.mem_factor), self.fin)
            self.temp = int(self.fin * self.mem_factor) ** -0.5

        self.block = nn.Sequential(
            nn.Linear(self.fin, self.fhidden, bias=bias),
            nn.SiLU(True),
            nn.Linear(self.fhidden, self.fout, bias=bias)
        )
        self.shortcut = nn.Linear(self.fin, self.fout, bias=bias)

    def forward(self, x):

        new_x = x
        if self.memory:
            mem = self.x_to_mem(new_x) * self.temp
            mem = F.softmax(mem, dim=1)
            mem = self.mem_to_x(mem)
            new_x = new_x + mem

        new_x = self.shortcut(new_x) + self.block(new_x)

        return new_x
