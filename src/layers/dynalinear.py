import torch
import torch.nn as nn
from src.layers.linearresidualblock import LinearResidualBlock


class DynaLinear(nn.Module):
    def __init__(self, lat_size, fin, fout, bias=True):
        super(DynaLinear, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.bias = bias

        self.w_size = self.fout * self.fin
        self.b_size = self.fout if bias else 0

        self.dyna_w = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.w_size + self.b_size, self.lat_size * 2),
        )

    def forward(self, x, lat):
        batch_size = x.size(0)

        w = self.dyna_w(lat)
        if self.bias:
            w, b = torch.split(w, [self.w_size, self.b_size], dim=1)

        w = w.view(batch_size, self.fin, self.fout)
        x_new = x.view(batch_size, 1, self.fin)
        x_new = torch.bmm(x_new, w).squeeze(1)
        if self.bias:
            x_new = x_new + b

        return x_new
