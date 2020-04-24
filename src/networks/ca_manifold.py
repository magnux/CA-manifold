import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.scale import DownScale, UpScale
from src.layers.lambd import LambdaLayer
from src.layers.sobel import Sobel, SinSobel
from src.layers.dynaconvblock import DynaConvBlock

from src.networks.conv_ae import Encoder, InjectedEncoder


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, train_noise, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls * 16
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.train_noise = train_noise

        self.frac_sobel = Sobel(self.n_filter)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * 3)
        self.frac_dyna_conv = DynaConvBlock(self.lat_size, self.n_filter * 3, self.n_filter)

    def forward(self, lat):
        batch_size = lat.size(0)

        # The canvas
        out = torch.zeros(batch_size, self.n_filter, self.image_size, self.image_size, device=lat.device)
        # The starting point of the wave
        out[:, 0, self.image_size // 2, self.image_size // 2] = 1.0

        if self.train_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = out
            if self.train_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)
            out = out + (leak_factor * out_new)
            out_embs.append(out)

        out = out[:, :self.out_chan, :, :]
        out_raw = out
        out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

