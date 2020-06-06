import torch
import torch.nn as nn
import numpy as np


def lin_pos_encoding_1d(size):
    pos_encoding_start = ((1. / torch.arange(1, 1 + size, dtype=torch.float32)).view(1, 1, size) * 2) - 1
    pos_encoding_end = ((1. - (1. / torch.arange(1, 1 + size, dtype=torch.float32))).view(1, 1, size) * 2) - 1
    pos_encoding_center = (torch.min(pos_encoding_start, pos_encoding_end) * 2) + 1

    return torch.cat([pos_encoding_start, pos_encoding_end, pos_encoding_center], 1)


def cos_pos_encoding_1d(size):
    sub_freqs = [1./ float(i) for i in range(2, 9)]
    freqs = list(range(1, size))
    spaces = [np.linspace(i * freq * 2 * np.pi, (i + 1) * freq * 2 * np.pi, size) for freq in sub_freqs for i in range(int(1 / freq))]
    spaces = spaces + [np.linspace(0, freq * 2 * np.pi, size) for freq in freqs]
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
            pos_enc = pos_enc.view(
                *[size[dd - 2] if dd == d else (pos_enc.size(1) if dd == 1 else 1) for dd in range(dim + 2)])
            pos_enc = pos_enc.repeat(*[size[dd - 2] if dd != d and dd > 1 else 1 for dd in range(dim + 2)])
            pos_enc_l.append(pos_enc)
        pos_encoding = torch.cat(pos_enc_l, 1)

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
