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
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np

from src.networks.conv_ae import Encoder, InjectedEncoder


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, perception_noise, fire_rate, skip_fire=False, injected=False, log_mix_out=False, ext_canvas=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls * 8
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.injected = injected
        self.log_mix_out = log_mix_out
        self.ext_canvas = ext_canvas

        self.frac_sobel = SinSobel(self.n_filter, 5, 2)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * 3)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size * (2 if self.injected else 1), self.n_filter * 3, self.n_filter, self.n_filter)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.ds_size, self.ds_size + (self.n_calls if self.ext_canvas else 0))).sum(axis=0) % 2, dtype=torch.float32, requires_grad=False)

        self.conv_img = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.image_size))] if self.ds_size < self.image_size else []),
            *([UpScale(self.n_filter, self.n_filter, self.ds_size, self.image_size)] if self.ds_size < self.image_size else []),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1),
        )

    def forward(self, lat, ca_init=None, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected decoders'
        batch_size = lat.size(0)

        if ca_init is None:
            out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device)
        else:
            out = ca_init

        if inj_lat is not None:
            lat = torch.cat([lat, inj_lat], dim=1)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        if self.ext_canvas:
            out = F.pad(out, [0, self.n_calls, 0, 0])
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=lat.device) <= self.fire_rate).to(torch.float32)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=lat.device)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=lat.device))
            out = out + (leak_factor * out_new)
            out_embs.append(out)

        if self.ext_canvas:
            out = out[:, :, :, self.n_calls:]

        out = self.conv_img(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw


class InjectedDecoder(Decoder):
    def __init__(self, **kwargs):
        kwargs['injected'] = True
        super().__init__(**kwargs)
