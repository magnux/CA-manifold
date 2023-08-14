import torch
import torch.nn as nn


class ResidualAttentionBlock(nn.Module):
    def __init__(self, fin, fhidden=None, nheads=8, dim=1, dropout=0, residual=True, condition=False, lat_size=0):
        super(ResidualAttentionBlock, self).__init__()

        self.fin = fin
        self.fhidden = fin if fhidden is None else fhidden
        self.nheads = nheads
        self.dim = dim
        self.residual = residual
        self.condition = condition

        if self.dim == 1:
            conv_fn = nn.Conv1d
        elif self.dim == 2:
            conv_fn = nn.Conv2d
        elif self.dim == 3:
            conv_fn = nn.Conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.q = conv_fn(self.fin, self.nheads * self.fhidden, 1, 1, 0)
        self.k = conv_fn(self.fin, self.nheads * self.fhidden, 1, 1, 0)
        self.v = conv_fn(self.fin, self.nheads * self.fhidden, 1, 1, 0)
        self.conv_out = conv_fn(self.nheads * self.fhidden, self.fin, 1, 1, 0)

        if self.condition:
            # self.q_mod = nn.Linear(lat_size, self.nheads * self.fhidden * 2)
            # self.k_mod = nn.Linear(lat_size, self.nheads * self.fhidden * 2)
            self.v_mod = nn.Linear(lat_size, self.nheads * self.fhidden * 2)
            # self.conv_out_mod = nn.Linear(lat_size, self.fin * 2)

        self.temp = (self.fhidden // nheads) ** -0.5
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, lat=None):
        assert (lat is not None) == self.condition
        batch_size = x.shape[0]

        x_q = self.q(x).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous()
        x_k = self.k(x).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous()
        x_v = self.v(x).view(batch_size,  self.nheads, self.fhidden, -1).view(batch_size * self.nheads, self.fhidden, -1).contiguous()

        if self.condition:
            # x_q_mod_m, x_q_mod_a = torch.chunk(self.q_mod(lat).view(batch_size,  self.nheads, self.fhidden * 2).reshape(batch_size * self.nheads, self.fhidden * 2, 1), 2, 1)
            # x_q = x_q + x_q_mod_m * x_q + x_q_mod_a
            # x_k_mod_m, x_k_mod_a = torch.chunk(self.k_mod(lat).view(batch_size,  self.nheads, self.fhidden * 2).reshape(batch_size * self.nheads, self.fhidden * 2, 1), 2, 1)
            # x_k = x_k + x_k_mod_m * x_k + x_k_mod_a
            x_v_mod_m, x_v_mod_a = torch.chunk(self.v_mod(lat).view(batch_size,  self.nheads, self.fhidden * 2).reshape(batch_size * self.nheads, self.fhidden * 2, 1), 2, 1)
            x_v = x_v + x_v_mod_m * x_v + x_v_mod_a

        attn = torch.bmm(x_q.permute(0, 2, 1), x_k) * self.temp
        attn = torch.softmax(attn, dim=2)
        if self.dropout is not None:
            attn = self.dropout(attn)
        x_out = torch.bmm(attn, x_v.permute(0, 2, 1))

        x_out = x_out.permute(0, 2, 1).view(batch_size, self.nheads, self.fhidden, -1).contiguous().view(batch_size, self.nheads * self.fhidden, -1).contiguous()
        x_out = x_out.view(*([batch_size, self.nheads * self.fhidden] + [s for s in x.shape[2:]]))

        x_out = self.conv_out(x_out)

        # if self.condition:
        #     x_out_mod_m, x_out_mod_a = torch.chunk(self.conv_out_mod(lat).view(*([batch_size, self.fin * 2] + [1 for _ in range(self.dim)])), 2, 1)
        #     x_out = x_out + x_out_mod_m * x_out + x_out_mod_a

        if self.residual:
            return x + x_out
        else:
            return x_out
