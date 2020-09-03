import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.pos_encoding import PosEncoding


class ResidualAttentionBlock(nn.Module):
    def __init__(self, size, fin, fhidden=None, nheads=8, dim=1, dropout=0.1):
        super(ResidualAttentionBlock, self).__init__()

        self.fin = fin
        self.fhidden = fin if fhidden is None else fhidden
        self.nheads = nheads
        self.dim = dim

        if self.dim == 1:
            conv_fn = nn.Conv1d
        elif self.dim == 2:
            conv_fn = nn.Conv2d
        elif self.dim == 3:
            conv_fn = nn.Conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.pos_encoding = PosEncoding(size, dim)

        self.q = conv_fn(self.fin + self.pos_encoding.size(), self.nheads * self.fhidden, 1, 1, 0)
        self.k = conv_fn(self.fin + self.pos_encoding.size(), self.nheads * self.fhidden, 1, 1, 0)
        self.v = conv_fn(self.fin + self.pos_encoding.size(), self.nheads * self.fhidden, 1, 1, 0)
        self.temp = self.fhidden ** 0.5
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.conv_out = conv_fn(self.nheads * self.fhidden, self.fin, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)

        x_pos = self.pos_encoding(x)

        x_q = self.q(x_pos).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous().permute(0, 2, 1)
        x_k = self.k(x_pos).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous()
        x_v = self.v(x_pos).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous().permute(0, 2, 1)

        attn = torch.bmm(x_q, x_k)
        attn = attn / self.temp
        attn = torch.softmax(attn, dim=2)
        if self.dropout is not None:
            attn = self.dropout(attn)
        x_out = torch.bmm(attn, x_v)

        x_out = x_out.permute(0, 2, 1).view(batch_size, self.nheads, self.fhidden, -1).contiguous().view(batch_size, self.nheads * self.fhidden, -1).contiguous()
        if self.dim == 2:
            x_out = x_out.view(batch_size, self.nheads * self.fhidden, x.size(2), x.size(3))
        elif self.dim == 3:
            x_out = x_out.view(batch_size, self.nheads * self.fhidden, x.size(2), x.size(3), x.size(4))

        x_out = self.conv_out(x_out)

        return x + x_out
