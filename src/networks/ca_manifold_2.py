import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.scale import DownScale, UpScale
from src.layers.lambd import LambdaLayer
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
from src.utils.model_utils import ca_seed

from src.networks.conv_ae import Encoder, InjectedEncoder


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls * 8
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate

        self.frac_sobel = SinSobel(self.n_filter, 5, 2)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * 3)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter * 3, self.n_filter)

        self.conv_img = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.image_size))] if self.ds_size < self.image_size else []),
            *([UpScale(self.n_filter, self.n_filter, self.ds_size, self.image_size)] if self.ds_size < self.image_size else []),
            nn.Conv2d(self.n_filter, self.out_chan, 3, 1, 1),
        )

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)

        if ca_init is None:
            out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device)
        else:
            out = ca_init

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, self.image_size, self.image_size], device=lat.device) <= self.fire_rate).to(torch.float32)
            out = out + (leak_factor * out_new)
            out_embs.append(out)

        out = self.conv_img(out)
        out_raw = out
        out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

