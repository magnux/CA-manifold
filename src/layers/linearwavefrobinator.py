import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.complexwave import complex_wave


class LinearWaveFrobinator(nn.Module):
    def __init__(self, fin, fout, fhidden=None):
        super(LinearWaveFrobinator, self).__init__()

        self.fin = fin
        self.fout = fout
        self.fhidden = int(self.fin ** 0.5) if fhidden is None else fhidden

        self.to_frob_a = nn.Linear(self.fin, self.fhidden)
        self.to_frob_b = nn.Linear(self.fin, self.fhidden)
        self.to_frob_c = nn.Linear((2 * self.fhidden) ** 2, self.fhidden)
        self.to_w = nn.Linear(self.fhidden, self.fin * self.fout)
        if self.fin != self.fout:
            self.res = nn.Linear(self.fin, self.fout)
        else:
            self.res = None

    def forward(self, x):
        batch_size = x.size(0)

        frob_a = complex_wave(self.to_frob_a(x)).view(batch_size, 2 * self.fhidden, 1)
        frob_b = complex_wave(self.to_frob_b(x)).view(batch_size, 1, 2 * self.fhidden)
        frob_ab = (frob_a * frob_b).view(batch_size, -1) / self.fhidden

        frob_c = F.normalize(self.to_frob_c(frob_ab))
        w = self.to_w(frob_c).view(batch_size, self.fin, self.fout) / self.fout ** 0.5
        if self.res is not None:
            x_res = self.res(x)
        else:
            x_res = x
        x_new = x_res - torch.bmm(x[:, None, :], w).squeeze(1)

        return x_new

