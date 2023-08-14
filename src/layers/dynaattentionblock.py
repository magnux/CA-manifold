import torch
import torch.nn as nn
from src.layers.dynaconvs import DynaConvS
import torch.nn.functional as F


class DynaAttentionBlock(nn.Module):
    def __init__(self, lat_size, fin, fhidden=None, nheads=8, dim=1, dropout=0, residual=True):
        super(DynaAttentionBlock, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fhidden = fin if fhidden is None else fhidden
        self.nheads = nheads
        self.dim = dim
        self.residual = residual

        self.q = DynaConvS(self.lat_size, self.fin, self.nheads * self.fhidden, 1, 1, 0, dim=dim)
        self.k = DynaConvS(self.lat_size, self.fin, self.nheads * self.fhidden, 1, 1, 0, dim=dim)
        self.v = DynaConvS(self.lat_size, self.fin, self.nheads * self.fhidden, 1, 1, 0, dim=dim)
        self.temp = (self.fhidden // nheads) ** -0.5
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.conv_out = DynaConvS(self.lat_size, self.nheads * self.fhidden, self.fin, 1, 1, 0, dim=dim)

    def forward(self, x, lat):
        batch_size = x.size(0)

        x_q = self.q(x, lat).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous().permute(0, 2, 1)
        x_k = self.k(x, lat).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous()
        x_v = self.v(x, lat).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous().permute(0, 2, 1)

        attn = torch.bmm(x_q, x_k) * self.temp
        attn = torch.softmax(attn, dim=2)
        if self.dropout is not None:
            attn = self.dropout(attn)
        x_out = torch.bmm(attn, x_v)

        x_out = x_out.permute(0, 2, 1).view(batch_size, self.nheads, self.fhidden, -1).contiguous().view(batch_size, self.nheads * self.fhidden, -1).contiguous()
        if self.dim == 2:
            x_out = x_out.view(batch_size, self.nheads * self.fhidden, x.size(2), x.size(3))
        elif self.dim == 3:
            x_out = x_out.view(batch_size, self.nheads * self.fhidden, x.size(2), x.size(3), x.size(4))

        x_out = self.conv_out(x_out, lat)

        if self.residual:
            return x + x_out
        else:
            return x_out
