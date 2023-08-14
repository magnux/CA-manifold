import torch
import torch.nn as nn


def complex_to_real(x, concat_dim=1, float_type=torch.float32):
    return torch.cat([x.real.to(float_type), x.imag.to(float_type)], dim=concat_dim)


def complex_wave(x, return_real=True, concat_dim=1, float_type=torch.float32):
    wave = torch.zeros_like(x, dtype=torch.complex64)
    wave.imag = x
    wave = wave.exp()
    if return_real:
        return complex_to_real(wave, concat_dim, float_type)
    else:
        return wave


class ComplexWave(nn.Module):
    def __init__(self, concat_dim=1):
        super(ComplexWave, self).__init__()
        self.concat_dim = concat_dim

    def forward(self, x):
        return complex_wave(x, concat_dim=self.concat_dim)

