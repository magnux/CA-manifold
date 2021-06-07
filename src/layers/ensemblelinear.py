import torch
import torch.nn as nn
import random

class EnsembleLinear(nn.Module):
    def __init__(self, fin, fout, bias=True, n_ensemble=8):
        super(EnsembleLinear, self).__init__()

        self.fin = fin
        self.fout = fout
        self.bias = bias
        self.n_ensemble = n_ensemble

        self.linear = nn.ModuleList([
            nn.Linear(self.fin, self.fout, self.bias) for _ in range(self.n_ensemble)
        ])

    def forward(self, x):
        x_new = self.linear[random.randrange(0, self.n_ensemble)](x)

        return x_new
