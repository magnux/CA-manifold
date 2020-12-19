import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.base import LabsEncoder
from src.layers.modconv import ModConv
from src.layers.noiseinjection import NoiseInjection
from src.layers.gaussiansmoothing import GaussianSmoothing
from src.layers.lambd import LambdaLayer
from src.utils.model_utils import checkerboard_seed
import numpy as np


class EncoderBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, downsample=True, lat_mask_prob=0.9):
        super().__init__()

        if lat_mask_prob < 1.:
            self.lat_mask = nn.Parameter((torch.rand(1, lat_size) < lat_mask_prob).to(torch.float32), requires_grad=False)

        self.conv_res = nn.Conv2d(fin, fout, 1, stride=(2 if downsample else 1))

        self.conv0 = ModConv(lat_size, fin, fout, 3)
        self.conv1 = ModConv(lat_size, fout, fout, 3)
        self.activation = nn.LeakyReLU(0.2, True)

        self.downsample = nn.Sequential(
            GaussianSmoothing(fout, 3, 1, 1),
            nn.Conv2d(fout, fout, 3, padding=1, stride=2)
        ) if downsample else None

    def forward(self, x, lat):
        if self.lat_mask is not None:
            lat = lat * self.lat_mask

        res = self.conv_res(x)

        x = self.conv0(x, lat)
        x = self.activation(x)
        x = self.conv1(x, lat)
        x = self.activation(x)

        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / np.sqrt(2))
        return x


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, z_out=False, z_dim=0, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size

        filters = [self.in_chan] + [self.n_filter * (2 ** (i + 1)) for i in range(int(np.log2(self.image_size)))]

        self.blocks = nn.ModuleList()
        for i in range(len(filters)-1):
            block = EncoderBlock(self.lat_size, filters[i], filters[i+1], downsample=i < len(filters) - 2)
            self.blocks.append(block)

        self.conv_out = nn.Conv2d(filters[-1], filters[-1], 3, 1, 1)

        self.out_to_lat = nn.Linear(2 * 2 * filters[-1], self.lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = x
        out_embs = []
        for block in self.blocks:
            out = block(out, inj_lat)
            out_embs.append(out)

        out = self.conv_out(out)
        out_embs.append(out)

        lat = self.out_to_lat(out.view(batch_size, -1))

        return lat, out_embs, None


class LabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = LabsEncoder(**kwargs)

    def forward(self, x, labels):
        inj_lat = self.labs_encoder(labels)
        return super().forward(x, inj_lat)


class ZInjectedEncoder(LabsInjectedEncoder):
    def __init__(self, **kwargs):
        kwargs['z_out'] = True
        super().__init__(**kwargs)


class DecoderBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, out_chan, upsample_in=True, upsample_out=True, lat_mask_prob=0.9):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample_in else None

        if lat_mask_prob < 1.:
            self.lat_mask = nn.Parameter((torch.rand(1, lat_size) < lat_mask_prob).to(torch.float32))

        self.conv0 = ModConv(lat_size, fin, fout, 3)
        self.noise0 = NoiseInjection(fout)

        self.conv1 = ModConv(lat_size, fout, fout, 3)
        self.noise1 = NoiseInjection(fout)

        self.activation = nn.LeakyReLU(0.2, True)

        self.conv_out = ModConv(lat_size, fout, out_chan, 1, demod=False)

        self.upsample_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GaussianSmoothing(out_chan, 3, 1, 1)
        ) if upsample_out else None

    def forward(self, x, lat, prev_out):
        if self.lat_mask is not None:
            lat = lat * self.lat_mask

        if self.upsample is not None:
            x = self.upsample(x)

        x = self.conv0(x, lat)
        x = self.noise0(x)
        x = self.activation(x)

        x = self.conv1(x, lat)
        x = self.noise1(x)
        x = self.activation(x)

        out = self.conv_out(x, lat)

        if prev_out is not None:
            out = out + prev_out

        if self.upsample_out is not None:
            out = self.upsample_out(out)

        return x, out


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, log_mix_out=False, n_seed=1, **kwargs):
        super().__init__()
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.out_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.log_mix_out = log_mix_out

        filters = [self.n_filter * (2 ** (i + 1)) for i in range(int(np.log2(self.image_size)))][::-1]

        self.blocks = nn.ModuleList()
        for i in range(len(filters) - 1):
            block = DecoderBlock(self.lat_size, filters[i], filters[i + 1], self.out_chan,
                                 upsample_in=i > 0, upsample_out=i < len(filters) - 2)
            self.blocks.append(block)

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, filters[0])).unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4))

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.out_chan, filters[0], 3, 1, 1),
            LambdaLayer(lambda x: F.interpolate(x, size=4, mode='bilinear', align_corners=False)),
        )

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, 1, 1)

        self.conv_img = nn.Conv2d(filters[-1], 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1)

    def forward(self, lat, ca_init=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.ds_size, lat.device).to(float_type)
            if isinstance(seed_n, tuple):
                mean_seed = self.seed[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
                out_emb = torch.cat([mean_seed.to(float_type)] * batch_size, 0)
            else:
                out_emb = torch.cat([self.seed[seed_n:seed_n+1, ...].to(float_type)] * batch_size, 0)
        else:
            out_emb = self.in_conv(ca_init)

        out_embs = [out_emb]
        out = None
        for block in self.blocks:
            out_emb, out = block(out_emb, lat, out)
            out_embs.append(out_emb)

        return out, out_embs, None
