import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


def lin_pos_encoding_1d(size):
    pos_encoding_start = ((1. / torch.arange(1, 1 + size, dtype=torch.float32)).view(1, 1, size) * 2) - 1
    pos_encoding_end = ((1. - (1. / torch.arange(1, 1 + size, dtype=torch.float32))).view(1, 1, size) * 2) - 1
    pos_encoding_center = (torch.min(pos_encoding_start, pos_encoding_end) * 2) + 1

    return torch.cat([pos_encoding_start, pos_encoding_end, pos_encoding_center], 1)


def cos_pos_encoding_1d(size, freq_div=2., sub_freq_max=4.):
    ## Note: freqs can be computed up to 'size', but only 'size / 2' are taken to reduce computation
    freqs = list(range(1, int(size / freq_div)))
    spaces = [np.linspace(0, freq * 2 * np.pi, size) for freq in freqs]

    ## Note: more sub freqs (low freqs, waves longer than size) can be computed, but only 2 are taken to reduce computation
    sub_freqs = [1./ float(i) for i in range(2, int(sub_freq_max))]
    sub_spaces = [np.linspace(i * freq * 2 * np.pi, (i + 1) * freq * 2 * np.pi, size) for freq in sub_freqs for i in range(int(1 / freq))]
    spaces = spaces + sub_spaces

    cos = [np.cos(space) for space in spaces]
    return torch.tensor(np.stack(cos).reshape(1, len(spaces), size), dtype=torch.float32)


def cos_pos_encoding_nd(size, dim):
    if isinstance(size, int):
        size = (size,)
        if dim > 0:
            size = size * dim
    # else :
    # TODO: check size tuples are correct
    if dim == 0:
        pos_encoding = cos_pos_encoding_1d(size[0]).view(1, -1)
    elif dim == 1:
        pos_encoding = cos_pos_encoding_1d(size[0])
    elif dim > 1:
        pos_enc_l = []
        for d in range(2, dim + 2):
            pos_enc = cos_pos_encoding_1d(size[d - 2])
            pos_enc = pos_enc.view(*[size[dd - 2] if dd == d else (pos_enc.size(1) if dd == 1 else 1) for dd in range(dim + 2)])
            pos_enc = pos_enc.repeat(*[size[dd - 2] if dd != d and dd > 1 else 1 for dd in range(dim + 2)])
            pos_enc_l.append(pos_enc)
        pos_enc_l_comb = []
        for c in range(2, dim + 1):
            combs = list(combinations(pos_enc_l, c))
            for comb in combs:
                comb_l = list(comb)
                while len(comb_l) > 1:
                    to_comb_a_l = torch.split(comb_l.pop(), 1, dim=1)
                    to_comb_b = comb_l.pop()
                    comb_pos_l = []
                    for to_comb_a in to_comb_a_l:
                        to_comb_a = torch.cat([to_comb_a] * to_comb_b.size(1), dim=1)
                        comb_pos_l.append(torch.stack([to_comb_a, to_comb_b], dim=-1).prod(dim=-1))
                    comb_l.append(torch.cat(comb_pos_l, 1))
                pos_enc_l_comb.append(torch.cat(comb_l, 1))
        pos_encoding = torch.cat(pos_enc_l + pos_enc_l_comb, 1)

    return pos_encoding


class PosEncoding(nn.Module):
    def __init__(self, size, dim=1):
        super(PosEncoding, self).__init__()
        self.register_buffer('pos_encoding', cos_pos_encoding_nd(size, dim))

    def forward(self, x):
        pos_encoding = torch.cat([self.pos_encoding] * x.size(0), 0)
        return torch.cat([x, pos_encoding], 1)

    def size(self):
        return int(self.pos_encoding.size(1))


class CosFreqEncoding(nn.Module):
    def __init__(self, lat_size, norm=True):
        super(CosFreqEncoding, self).__init__()
        self.lat_size = lat_size
        self.norm = norm
        cos_frec_encoding = cos_pos_encoding_1d(lat_size, 1, 8).unsqueeze_(0)
        self.register_buffer('cos_frec_weight', cos_frec_encoding)
        self.to_freq_size = nn.Linear(self.lat_size, cos_frec_encoding.size(1), bias=None)

    def forward(self, x):
        x_freqs = self.to_freq_size(x)
        x_freqs = (x_freqs.unsqueeze(2) * self.cos_freq_weight).sum(dim=1)
        if self.norm:
            return x_freqs / x_freqs.max()
        else:
            return x_freqs
