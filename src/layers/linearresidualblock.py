import torch
import torch.nn as nn


class LinearResidualBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bias=True):
        super(LinearResidualBlock, self).__init__()

        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden

        self.block = nn.Sequential(
            nn.Linear(self.fin, self.fhidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(self.fhidden, self.fhidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(self.fhidden, self.fout, bias=bias)
        )
        self.shortcut = nn.Linear(self.fin, self.fout, bias=bias)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
