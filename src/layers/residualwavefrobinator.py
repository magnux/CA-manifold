import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.complexwave import complex_wave


class ResidualWaveFrobinator(nn.Module):
    def __init__(self, fin, fout, fhidden=None, dim=2):
        super(ResidualWaveFrobinator, self).__init__()

        self.fin = fin
        self.fout = fout
        self.fhidden = int(self.fin ** 0.5) if fhidden is None else fhidden

        self.dim = dim
        if dim == 1:
            self.f_conv = F.conv1d
            self.l_conv = nn.Conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
            self.l_conv = nn.Conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
            self.l_conv = nn.Conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.to_frob_a = self.l_conv(self.fin, self.fhidden, 1, 1, 0)
        self.to_frob_b = self.l_conv(self.fin, self.fhidden, 1, 1, 0)
        self.to_frob_c = self.l_conv(((2 * self.fhidden) ** 2), self.fhidden, 1, 1, 0)
        self.to_w = nn.Linear(self.fhidden, self.fin * self.fout)
        if self.fin != self.fout:
            self.res = self.l_conv(self.fin, self.fout, 1, 1, 0)
        else:
            self.res = None

        self.kernel_size = [1 for _ in range(self.dim)]

    def forward(self, x):
        batch_size = x.size(0)

        frob_a = complex_wave(self.to_frob_a(x)).view([batch_size, 2 * self.fhidden, 1] + [x.size(d + 2) for d in range(self.dim)])
        frob_b = complex_wave(self.to_frob_b(x)).view([batch_size, 1, 2 * self.fhidden] + [x.size(d + 2) for d in range(self.dim)])
        frob_ab = (frob_a * frob_b).view([batch_size, (2 * self.fhidden) ** 2] + [x.size(d + 2) for d in range(self.dim)]) / self.fhidden

        frob_c = F.instance_norm(self.to_frob_c(frob_ab)).permute(0, 2, 3, 1).unsqueeze(3)
        w = self.to_w(frob_c).view(-1, self.fin, self.fout) / self.fout ** 0.5
        if self.res is not None:
            x_res = self.res(x)
        else:
            x_res = x
        x_new = x_res - torch.bmm(x.permute(0, 2, 3, 1).contiguous().view(-1, 1, self.fin), w).squeeze(1).view([batch_size] + [x.size(d + 2) for d in range(self.dim)] + [self.fout]).permute(0, 3, 1, 2)

        return x_new

