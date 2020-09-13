import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.residualmemory import ResidualMemory
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.scale import UpScale
from src.utils.loss_utils import sample_from_discretized_mix_logistic

from src.networks.conv_ae import Encoder, InjectedEncoder


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, shared_params, log_mix_out=False, **kwargs):
        super().__init__()
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.out_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.log_mix_out = log_mix_out
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.merge_sizes = [self.n_filter, self.n_filter, self.n_filter, 1]
        self.conv_state_size = [self.n_filter, self.n_filter * self.ds_size, self.n_filter * self.ds_size, self.ds_size ** 2]

        self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
        self.frac_mem = nn.ModuleList([ResidualMemory(self.ds_size, self.n_filter, 16, 2) for _ in range(1 if self.shared_params else self.n_calls)])

        self.lat_to_out = nn.Sequential(
            *([] if lat_size > 3 else [nn.Linear(lat_size, self.lat_size, bias=False)]),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, sum(self.conv_state_size), self.lat_size * 2),
        )

        self.in_conv = ResidualBlock(sum(self.merge_sizes), self.n_filter, None, 1, 1, 0)

        self.conv_img = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.image_size))] if self.ds_size < self.image_size else []),
            *([UpScale(self.n_filter, self.n_filter, self.ds_size, self.image_size)] if self.ds_size < self.image_size else []),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1),
        )

    def forward(self, lat):
        batch_size = lat.size(0)

        conv_state = self.lat_to_out(lat)
        cs_f_m, cs_fh, cs_fw, cs_hw = torch.split(conv_state, self.conv_state_size, dim=1)
        cs_f_m = cs_f_m.view(batch_size, self.n_filter, 1, 1).repeat(1, 1, self.ds_size, self.ds_size)
        cs_fh = cs_fh.view(batch_size, self.n_filter, self.ds_size, 1).repeat(1, 1, 1, self.ds_size)
        cs_fw = cs_fw.view(batch_size, self.n_filter, 1, self.ds_size).repeat(1, 1, self.ds_size, 1)
        cs_hw = cs_hw.view(batch_size, 1, self.ds_size, self.ds_size)
        out = torch.cat([cs_f_m, cs_fh, cs_fw, cs_hw], dim=1)
        out = self.in_conv(out)

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = self.frac_norm[0 if self.shared_params else c](out)
            out_new = self.frac_mem[0 if self.shared_params else c](out_new)
            out = out + (leak_factor * out_new)
            out_embs.append(out)

        out = self.conv_img(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

