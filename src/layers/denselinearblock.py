import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLinearBlock(nn.Module):
    def __init__(self, fin, n_layers=4, bias=True):
        super(DenseLinearBlock, self).__init__()

        self.fin = fin

        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for n in range(self.n_layers):
            self.layers.append(nn.Linear(self.fin * (n + 1), self.fin, bias=bias))

    def forward(self, x):
        xs = [x]
        for n in range(self.n_layers-1):
            new_x = self.layers[n](torch.cat(xs, dim=1))
            new_x = F.relu(new_x, True)
            xs.append(new_x)
        x = self.layers[-1](torch.cat(xs, dim=1))
        return x
