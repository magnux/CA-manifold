import torch
import torch.nn as nn


class ProjScale(nn.Module):
    def __init__(self, fin):
        super(ProjScale, self).__init__()
        self.proj_weight = nn.Parameter(torch.randn(1, fin, fin))

    def forward(self, x):
        return torch.bmm(x.view(x.shape[0], 1, x.shape[1]), self.proj_weight.repeat(x.shape[0], 1, 1)).squeeze_(1)
