import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.sobel import Sobel
from src.layers.dynaconvblock import DynaConvBlock
from src.utils.model_utils import ca_seed

from src.networks.conv_ae import Encoder, InjectedEncoder


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 fixed_conv=False, alive_masking=False, deactivate_norm=False, leak_factor=None, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls * 16
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.fixed_conv = fixed_conv
        self.alive_masking = alive_masking
        self.deactivate_norm = deactivate_norm

        if leak_factor is not None:
            self.leak_factor = leak_factor
        else:
            self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        self.frac_sobel = Sobel(self.n_filter)

        if not self.deactivate_norm:
            self.frac_norm = nn.InstanceNorm2d(self.n_filter * 3)

        if self.fixed_conv:
            self.frac_conv = nn.Sequential(
                nn.Conv2d(self.n_filter, self.n_filter * 8, 1, 1, 0),
                nn.ReLU(True),
                nn.Conv2d(self.n_filter * 8, self.n_filter, 1, 1, 0),
            )
        else:
            self.frac_dyna_conv = DynaConvBlock(self.lat_size, self.n_filter * 3, self.n_filter)

    def forward(self, lat, ca_init=None):
        assert self.fixed_conv == lat is None, 'when using fixed convs, the model is fixed to produce a single output, thus the latent should be None'
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device).to(float_type)
        else:
            out = ca_init

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = out
            if self.alive_masking:
                pre_life_mask = F.max_pool2d(out_new[:, 3:4, :, :], 3, 1, 1) > 0.1
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            if not self.dectivate_norm:
                out_new = self.frac_norm(out_new)
            if self.fixed_conv:
                out_new = self.frac_conv(out_new)
            else:
                out_new = self.frac_dyna_conv(out_new, lat)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, self.image_size, self.image_size], device=lat.device) <= self.fire_rate).to(float_type)
            if self.alive_masking:
                post_life_mask = F.max_pool2d(out_new[:, 3:4, :, :], 3, 1, 1) > 0.1
                life_mask = (pre_life_mask & post_life_mask).to(float_type)
                out = out_new * life_mask
            out = out + (leak_factor * out_new)
            out_embs.append(out)

        out = out[:, :self.out_chan, :, :]
        out_raw = out
        out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

