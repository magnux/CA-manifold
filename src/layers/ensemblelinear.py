import torch
import torch.nn as nn


class EnsembleLinear(nn.Module):
    def __init__(self, fin, fout, bias=True, n_ensemble=8):
        super(EnsembleLinear, self).__init__()

        self.fin = fin
        self.fout = fout
        self.bias = bias
        self.n_ensemble = n_ensemble

        self.linear = nn.Linear(self.fin, self.fout * self.n_ensemble, self.bias)

    def forward(self, x):
        x_new = self.linear(x)
        x_new_l = torch.split(x_new, self.fout, dim=1)
        x_new = torch.stack(x_new_l, dim=2).mean(dim=2)

        return x_new
