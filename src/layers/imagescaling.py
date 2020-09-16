import torch.nn as nn
from src.layers.residualblock import ResidualBlock
from src.layers.lambd import LambdaLayer


class DownScale(nn.Module):
    def __init__(self, in_f, out_f, in_size, out_size, n_blocks=None, dim=2):
        super(DownScale, self).__init__()
        assert in_size > out_size,  'It should only be used for downscaling'

        self.n_blocks = n_blocks if n_blocks is not None else (in_size // (out_size * 2) + 1)
        self.sizes = [int(in_size * ((self.n_blocks - b) / self.n_blocks) +
                          out_size * (b / self.n_blocks)) for b in range(self.n_blocks + 1)]
        self.fs = [int(in_f * ((self.n_blocks - b) / self.n_blocks) +
                       out_f * (b / self.n_blocks)) for b in range(self.n_blocks + 1)]

        self.dim = dim
        self.conv_fn = nn.Conv1d if dim == 1 else nn.Conv2d

        scale_blocks = []
        for b in range(self.n_blocks):
            in_f = self.fs[b]
            out_f = self.fs[b + 1]
            out_size = self.sizes[b + 1]
            scale_blocks.append(LambdaLayer(lambda x: nn.functional.interpolate(x, size=out_size)))
            scale_blocks.append(ResidualBlock(in_f, out_f, conv_fn=self.conv_fn))
        self.scale_blocks = nn.Sequential(*scale_blocks)

    def forward(self, x):
        return self.scale_blocks(x)


class UpScale(nn.Module):
    def __init__(self, in_f, out_f, in_size, out_size, n_blocks=None, dim=2):
        super(UpScale, self).__init__()
        assert in_size < out_size,  'It should only be used for upscaling'

        self.n_blocks = n_blocks if n_blocks is not None else (out_size // (in_size * 2) + 1)
        self.sizes = [int(in_size * ((self.n_blocks - b) / self.n_blocks) +
                          out_size * (b / self.n_blocks)) for b in range(self.n_blocks + 1)]
        self.fs = [int(in_f * ((self.n_blocks - b) / self.n_blocks) +
                       out_f * (b / self.n_blocks)) for b in range(self.n_blocks + 1)]

        self.dim = dim
        self.conv_fn = nn.Conv1d if dim == 1 else nn.Conv2d

        scale_blocks = []
        for b in range(self.n_blocks):
            in_f = self.fs[b]
            out_f = self.fs[b + 1]
            out_size = self.sizes[b + 1]
            scale_blocks.append(LambdaLayer(lambda x: nn.functional.interpolate(x, size=out_size)))
            scale_blocks.append(ResidualBlock(in_f, out_f, conv_fn=self.conv_fn))
        self.scale_blocks = nn.Sequential(*scale_blocks)

    def forward(self, x):
        return self.scale_blocks(x)
