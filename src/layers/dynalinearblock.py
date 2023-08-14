import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class DynaLinearBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, bias=True):
        super(DynaLinearBlock, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.bias = bias

        self.wf_size = self.fin * self.fin
        self.bf_size = self.fin if bias else 0
        self.w_size = self.fout * self.fin
        self.b_size = self.fout if bias else 0

        self.dyna_w = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.wf_size + self.bf_size + self.w_size + self.b_size, self.lat_size * 2),
        )

        self.prev_lat = None
        self.w, self.b = None, 0

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():

            w = self.dyna_w(lat)
            if self.bias:
                wf, bf, w, b = torch.split(w, [self.wf_size, self.bf_size, self.w_size, self.b_size], dim=1)
                self.bf = bf.view(batch_size, 1, self.fin)
                self.b = b.view(batch_size, 1, self.fout)
            else:
                wf, w = torch.split(w, [self.wf_size, self.w_size], dim=1)
            self.wf = wf.view(batch_size, self.fin, self.fin)
            self.w = w.view(batch_size, self.fin, self.fout)

            self.prev_lat = lat

        x_new = x.view(batch_size, -1, self.fin)
        x_new = torch.bmm(x_new, self.wf) + self.bf
        x_new = F.relu(x_new)
        x_new = torch.bmm(x_new, self.w) + self.b
        x_new = x_new.view(x.shape[:-1] + (self.fout,))

        return x_new
