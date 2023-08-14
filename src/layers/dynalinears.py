import torch
import torch.nn as nn
from src.layers.linearresidualblocks import LinearResidualBlockS


class DynaLinearS(nn.Module):
    def __init__(self, lat_size, fin, fout, bias=True, lat_factor=1):
        super(DynaLinearS, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.bias = bias

        self.w_size = self.fout * self.fin
        self.b_size = self.fout if bias else 0

        if lat_factor == 0:
            self.dyna_w = nn.Linear(self.lat_size, self.w_size + self.b_size)
        else:
            self.dyna_w = nn.Sequential(
                LinearResidualBlockS(self.lat_size, int(fin * lat_factor)),
                LinearResidualBlockS(int(fin * lat_factor), self.w_size + self.b_size, int(fout * lat_factor) * 2),
            )

        self.prev_lat = None
        self.w, self.b = None, 0

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():

            w = self.dyna_w(lat)
            if self.bias:
                w, b = torch.split(w, [self.w_size, self.b_size], dim=1)
                self.b = b.view(batch_size, 1, self.fout)
            self.w = w.view(batch_size, self.fin, self.fout)

            self.prev_lat = lat

        x_new = x.view(batch_size, -1, self.fin)
        x_new = torch.bmm(x_new, self.w) + self.b
        x_new = x_new.view(x.shape[:-1] + (self.fout,))

        return x_new
