import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.networks.base import LabsEncoder
from src.utils.model_utils import checkerboard_seed
from itertools import chain
import numpy as np


class Encoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, injected=False, **kwargs):
        super().__init__()
        self.injected = injected
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size

        self.conv_img = nn.Sequential(
            nn.Conv2d(self.in_chan, self.n_filter, 3, 1, 1),
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
        )

        self.conv_block = nn.Sequential(
            *chain(*[(nn.InstanceNorm2d(self.n_filter),
                      ResidualBlock(self.n_filter, self.n_filter, self.n_filter, 3, 2, 1),
                      ResidualBlock(self.n_filter, self.n_filter, self.n_filter * 2 ** i, 1, 1, 0)) for i in range(int(np.log2(image_size)))])
        )

        if self.injected:
            self.lat_to_in = nn.Sequential(
                LinearResidualBlock(self.lat_size, self.lat_size),
                LinearResidualBlock(self.lat_size, self.n_filter),
            )
            self.inj_cond = ResidualBlock(self.n_filter * 2, self.n_filter, None, 1, 1, 0)

        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(self.n_filter, self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            nn.Linear(self.lat_size, self.lat_size)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = self.conv_img(x)

        if self.injected:
            conv_state = self.lat_to_in(inj_lat)
            cs_f_m = conv_state.view(batch_size, self.n_filter, 1, 1).repeat(1, 1, self.image_size, self.image_size)
            out = torch.cat([out, cs_f_m], dim=1)
            out = self.inj_cond(out)

        out = self.conv_block(out)
        lat = self.out_to_lat(out.mean(dim=(2, 3)))

        return lat, [out], None


class InjectedEncoder(Encoder):
    def __init__(self, **kwargs):
        kwargs['injected'] = True
        super().__init__(**kwargs)


class LabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = LabsEncoder(**kwargs)

    def forward(self, x, labels):
        inj_lat = self.labs_encoder(labels)
        return super().forward(x, inj_lat)


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, **kwargs):
        super().__init__()
        self.n_labels = n_labels
        self.image_size = image_size
        self.out_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_blocks = int(np.ceil(np.log2(image_size)))

        self.lat_to_cond = nn.ModuleList(
            [LinearResidualBlock(self.lat_size, self.n_filter) for _ in range(self.n_blocks)]
        )

        self.seed = nn.Parameter(checkerboard_seed(1, self.n_filter, self.ds_size, 'cpu').to(torch.float32))

        self.conv_block = nn.ModuleList(
            [nn.Sequential(nn.InstanceNorm2d(self.n_filter),
                           ResidualBlock(self.n_filter * 3, self.n_filter, self.n_filter, 3, 1, 1),
                           ResidualBlock(self.n_filter, self.n_filter, self.n_filter * 2 ** (int(np.log2(image_size)) - i), 1, 1, 0)) for i in range(self.n_blocks)]
        )

        self.conv_img = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            nn.Conv2d(self.n_filter, self.out_chan, 3, 1, 1),
        )

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            ca_init = torch.cat([self.seed.to(float_type)] * batch_size, 0)

        out_embs = []
        out = torch.zeros((batch_size, self.n_filter, 1, 1), device=lat.device)
        for i in range(self.n_blocks):
            out = F.interpolate(out, size=2 ** (i + 1))
            out_init = F.interpolate(ca_init, size=2 ** (i + 1))
            out_cond = self.lat_to_cond(lat).view(batch_size, self.n_filter, 1, 1).repeat(1, 1, 2 ** (i + 1), 2 ** (i + 1))
            out = torch.cat([out, out_init, out_cond], dim=1)
            out = self.conv_block[i](out)
            out_embs.append(out)

        if out.size(2) != self.image_size:
            out = F.interpolate(out, size=self.image_size)
        out = self.conv_img(out)
        out_raw = out
        out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

