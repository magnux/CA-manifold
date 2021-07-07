import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class DynaLinearResidualBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, bias=True):
        super(DynaLinearResidualBlock, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden
        self.bias = bias

        self.w_in_size = self.fhidden * self.fin
        self.w_mid_size = self.fhidden * self.fhidden
        self.w_out_size = self.fout * self.fhidden
        self.w_short_size = self.fout * self.fin
        self.b_in_size = self.fhidden if bias else 0
        self.b_mid_size = self.fhidden if bias else 0
        self.b_out_size = self.fout if bias else 0
        self.b_short_size = self.fout if bias else 0

        self.dyna_w = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.w_in_size + self.w_mid_size + self.w_out_size + self.w_short_size +
                                               self.b_in_size + self.b_mid_size + self.b_out_size + self.b_short_size, self.lat_size * 2),
        )

        self.prev_lat = None
        self.w_in, self.w_mid, self.w_out, self.w_short = None, None, None, None
        self.b_in, self.b_mid, self.b_out, self.b_short = 0, 0, 0, 0

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():

            w_in, w_mid, w_out, w_short, b_in, b_mid, b_out, b_short = torch.split(self.dyna_w(lat), [self.w_in_size, self.w_mid_size,
                                                                                                      self.w_out_size, self.w_short_size,
                                                                                                      self.b_in_size, self.b_mid_size,
                                                                                                      self.b_out_size, self.b_short_size], dim=1)
            self.w_in = w_in.view(batch_size, self.fin, self.fhidden) / self.fin ** 0.5
            self.w_mid = w_mid.view(batch_size, self.fhidden, self.fhidden) / self.fhidden ** 0.5
            self.w_out = w_out.view(batch_size, self.fhidden, self.fout) / self.fhidden ** 0.5
            self.w_short = w_short.view(batch_size, self.fin, self.fout) / self.fin ** 0.5

            if self.bias:
                self.b_in = b_in.view(batch_size, 1, self.fhidden) / self.fhidden
                self.b_mid = b_mid.view(batch_size, 1, self.fhidden) / self.fhidden
                self.b_out = b_out.view(batch_size, 1, self.fout) / self.fout
                self.b_short = b_short.view(batch_size, 1, self.fout) / self.fout

            self.prev_lat = lat

        x_new = x.view(batch_size, 1, self.fin)
        x_new_s = torch.bmm(x_new, self.w_short) + self.b_short
        x_new = torch.bmm(x_new, self.w_in) + self.b_in
        x_new = F.relu(x_new, True)
        x_new = torch.bmm(x_new, self.w_mid) + self.b_mid
        x_new = F.relu(x_new, True)
        x_new = torch.bmm(x_new, self.w_out) + self.b_out
        x_new = x_new + x_new_s
        x_new = x_new.squeeze(1)

        return x_new
