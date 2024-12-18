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


def surf_encoding_nd(size, dim):
    if isinstance(size, int):
        size = (size,)
        if dim > 0:
            size = size * dim

    surf_encodings = []
    for d in range(2, dim + 2):
        surf_encoding = (-1. + (2. / torch.arange(1 + size[d - 2], 1, -1, dtype=torch.float32)))
        surf_encoding = surf_encoding.view(*[size[dd - 2] if dd == d else 1 for dd in range(dim + 2)])
        surf_encoding = surf_encoding.repeat(*[size[dd - 2] if dd != d and dd > 1 else 1 for dd in range(dim + 2)])
        surf_encodings.append(surf_encoding)

    return torch.cat(surf_encodings, dim=1)


def cos_pos_encoding_1d(size, phase=0., log_freq=True, freq_div=2., sub_freq_max=2):
    if log_freq:
        ## Note: freqs can be computed up to 'size', but taking intervals in 2^i intervals
        freqs = [2 ** i for i in range(int(np.log2(size)))]
    else:
        ## Note: freqs can be computed up to 'size / 2', but only 'size / freq_div' are taken to reduce computation
        freqs = list(range(1, int(size / freq_div)))
    spaces = [np.linspace(phase, phase + (freq * 2 * np.pi), size) for freq in freqs]

    ## Note: more sub freqs (low freqs, waves longer than size) can be computed, but only 'sub_freq_max - 2' are taken to reduce computation
    sub_freqs = [1. / float(i) for i in range(2, int(sub_freq_max))]
    sub_spaces = [np.linspace(i * freq * 2 * np.pi, (i + 1) * freq * 2 * np.pi, size) for freq in sub_freqs for i in range(int(1 / freq))]
    spaces = spaces + sub_spaces

    cos = [np.cos(space) for space in spaces]
    return torch.tensor(np.stack(cos).reshape(1, len(spaces), size), dtype=torch.float32)


def cos_pos_encoding_nd(size, dim, phase=0.):
    if isinstance(size, int):
        size = (size,)
        if dim > 0:
            size = size * dim
    # else :
    # TODO: check size tuples are correct
    if dim == 0:
        pos_encoding = cos_pos_encoding_1d(size[0], phase).view(1, -1)
    elif dim == 1:
        pos_encoding = cos_pos_encoding_1d(size[0], phase)
    elif dim > 1:
        pos_enc_l = []
        for d in range(2, dim + 2):
            pos_enc = cos_pos_encoding_1d(size[d - 2], phase)
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


def cos_pos_encoding_dyn(size, dim, n_calls):
    pos_enc_l = []
    phase_step = 2 * np.pi / n_calls
    for c in range(n_calls):
        pos_enc_l.append(cos_pos_encoding_nd(size, dim, c * phase_step))
    return torch.stack(pos_enc_l)


def sin_cos_pos_encoding_1d(size, phase=0., log_freq=True, freq_div=2.):
    if log_freq:
        freqs = [2 ** i for i in range(int(np.log2(size)))]
    else:
        freqs = list(range(1, int(size / freq_div)))
    spaces = [np.linspace(phase, phase + (freq * 2 * np.pi), size) for freq in freqs]

    sin = [np.sin(space) * (len(spaces) - s) / len(spaces) for s, space in enumerate(spaces)]
    cos = [np.cos(space) * (len(spaces) - s) / len(spaces) for s, space in enumerate(spaces)]
    return torch.tensor(np.stack(sin + cos)[None, :, :], dtype=torch.float32)


def sin_cos_pos_encoding_1d_2(size, phase=0., pos_scale=32):
    scales = [(i + 1) / pos_scale for i in range(pos_scale)]
    spaces = [(scale / np.sum(scales)) * np.linspace(phase, phase + (2 * np.pi), size) for scale in scales]

    sin = [np.sin(space) for space in spaces]
    cos = [np.cos(space) for space in spaces]
    return torch.tensor(np.stack(sin + cos)[None, :, :], dtype=torch.float32)


def sin_cos_pos_encoding_nd(size, dim, version=1, phase=0., log_freq=True, pos_scale=32):
    if isinstance(size, int):
        size = (size,)
        if dim > 0:
            size = size * dim
    if version == 1 or version == 4:
        encoding_fun = sin_cos_pos_encoding_1d
    elif version == 2:
        encoding_fun = sin_cos_pos_encoding_1d_2
    else:
        raise RuntimeError('version {} not implemented.'.format(version))

    # else :
    # TODO: check size tuples are correct
    if dim == 0:
        pos_encoding = encoding_fun(size[0], phase, pos_scale if version == 2 else log_freq).view(1, -1)
    elif dim == 1:
        pos_encoding = encoding_fun(size[0], phase, pos_scale if version == 2 else log_freq)
    elif dim > 1:
        sin_enc_l = []
        cos_enc_l = []
        for d in range(2, dim + 2):
            pos_enc = encoding_fun(size[d - 2], phase, pos_scale if version == 2 else log_freq)
            pos_enc = pos_enc.view(*[size[dd - 2] if dd == d else (pos_enc.size(1) if dd == 1 else 1) for dd in range(dim + 2)])
            pos_enc = pos_enc.repeat(*[size[dd - 2] if dd != d and dd > 1 else 1 for dd in range(dim + 2)])
            sin_enc, cos_enc = torch.split(pos_enc, pos_enc.shape[1] // 2, dim=1)
            sin_enc_l.append(sin_enc)
            cos_enc_l.append(cos_enc)
        pos_enc_l_comb = []
        for pos_enc_l in (sin_enc_l, cos_enc_l):
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
        pos_encoding = torch.cat(sin_enc_l + cos_enc_l + pos_enc_l_comb, 1)

    return pos_encoding


def sin_cos_pos_encoding_dyn(size, dim, n_calls):
    pos_enc_l = []
    phase_step = 2 * np.pi / n_calls
    for c in range(n_calls):
        pos_enc_l.append(sin_cos_pos_encoding_nd(size, dim, version=4, phase=c * phase_step))
    return torch.stack(pos_enc_l)


class PosEncoding(nn.Module):
    def __init__(self, size, dim=1, version=1, n_calls=1, sum_out=False):
        super(PosEncoding, self).__init__()
        self.version = version
        self.sum_out = sum_out
        if self.version == 3:
            self.register_buffer('pos_encoding', cos_pos_encoding_dyn(size, dim, n_calls))
        elif self.version == 4:
            self.register_buffer('pos_encoding', sin_cos_pos_encoding_dyn(size, dim, n_calls))
        else:
            self.register_buffer('pos_encoding', sin_cos_pos_encoding_nd(size, dim, version=version))

    def forward(self, x, call_n=0):
        if self.version == 3 or self.version == 4:
            if isinstance(call_n, int):
                pos_encoding = torch.cat([self.pos_encoding[call_n]] * x.size(0), 0)
            elif isinstance(call_n, torch.Tensor):
                pos_encoding = self.pos_encoding[call_n].squeeze(1)
            else:
                raise RuntimeError('call_n has to be int or Tensor')
        else:
            pos_encoding = torch.cat([self.pos_encoding] * x.size(0), 0)

        if self.sum_out:
            return x + pos_encoding
        else:
            return torch.cat([x, pos_encoding], 1)

    def size(self):
        if self.version == 3 or self.version == 4:
            return int(self.pos_encoding.size(2))
        else:
            return int(self.pos_encoding.size(1))


class LatFreqEncoding(nn.Module):
    def __init__(self, lat_size, norm=False, version=1):
        super(LatFreqEncoding, self).__init__()
        self.lat_size = lat_size
        self.norm = norm
        sin_cos_freq_encoding = sin_cos_pos_encoding_nd(self.lat_size, 1, version=version)
        self.register_buffer('sin_cos_freq_encoding', sin_cos_freq_encoding)
        self.to_freq_size = nn.Linear(self.lat_size, sin_cos_freq_encoding.size(1), bias=False)

    def forward(self, x):
        x_freqs = self.to_freq_size(x)
        x_freqs = (x_freqs.unsqueeze(2) * self.sin_cos_freq_encoding).mean(dim=1)

        if self.norm:
            x_freqs = F.normalize(x_freqs)

        return x_freqs


class ConvFreqEncoding(nn.Module):
    def __init__(self, n_filter, size, dim=2, version=1):
        super(ConvFreqEncoding, self).__init__()
        sin_cos_freq_encoding = sin_cos_pos_encoding_nd(size, dim, version=version)
        self.register_buffer('sin_cos_freq_encoding', sin_cos_freq_encoding)

        if dim == 1:
            self.l_conv = nn.Conv1d
        elif dim == 2:
            self.l_conv = nn.Conv2d
        elif dim == 3:
            self.l_conv = nn.Conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.out_conv = self.l_conv(n_filter, self.sin_cos_freq_encoding.shape[1], 1, 1, 0, bias=False)
        nn.init.normal_(self.out_conv.weight)

    def forward(self, x):
        return self.out_conv(x) * self.sin_cos_freq_encoding

    def size(self):
        return int(self.sin_cos_freq_encoding.size(1))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_size = 16
    n_calls = 8
    cos_pos = sin_cos_pos_encoding_dyn(img_size, 2, n_calls)

    # for c in range(n_calls):
    for i in range(cos_pos.shape[2]):
        plt.imshow(cos_pos[0, :, i, ...].reshape(img_size, img_size).detach().numpy())
        plt.show()

    # lat_size = 512
    # cos_enc = LatFreqEncoding(lat_size, True)
    # print(cos_enc.cos_freq_encoding.shape)
    #
    # n_samples = 16
    # sample = cos_enc(torch.randn((n_samples, lat_size))).unsqueeze(2).repeat(1, 1, lat_size // n_samples)
    # plt.imshow(sample.view(n_samples, lat_size, lat_size // n_samples).permute(1, 0, 2).reshape(lat_size, n_samples * (lat_size // n_samples)).detach().numpy())
    # plt.show()
