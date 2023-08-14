import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualattentionblock import ResidualAttentionBlock
from src.layers.dynaconvs import DynaConvS


class ResidualBlockS(nn.Module):
    def __init__(self, fin, fout, fhidden=None, dim=2, pos_enc=False, image_size=0, canvas_size=0, condition=False, lat_size=0, memory=False, mem_factor=1, attention=False, attn_patches=16):
        super(ResidualBlockS, self).__init__()

        self.fin = fin
        self.fout = fout

        self.dim = dim
        if dim == 1:
            conv_fn = nn.Conv1d
            convtr_fn = nn.ConvTranspose1d
        elif dim == 2:
            conv_fn = nn.Conv2d
            convtr_fn = nn.ConvTranspose2d
        elif dim == 3:
            conv_fn = nn.Conv3d
            convtr_fn = nn.ConvTranspose3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.fhidden = max(fin + fout, 1) if fhidden is None else fhidden

        self.pos_enc = pos_enc
        self.image_size = image_size
        self.canvas_size = canvas_size if canvas_size > 0 else image_size
        self.condition = condition
        self.lat_size = lat_size
        self.memory = memory
        self.mem_factor = mem_factor
        self.attention = attention
        self.attn_patches = min(attn_patches, image_size)

        self.conv_wave = conv_fn(self.fin, self.fhidden, kernel_size=3, stride=1, padding=1)
        self.norm_wave = nn.GroupNorm(1, self.fhidden)

        if self.condition:
            self.wave_cond = nn.Linear(self.lat_size, self.fhidden * 2)
            # self.wave_cond = DynaConvS(self.lat_size, self.fhidden, self.fhidden, lat_factor=0)
            self.norm_cond = nn.GroupNorm(1, self.fhidden)

        if pos_enc:
            self.wave_freqs = nn.Parameter(torch.randn([1, self.fhidden] + [self.image_size for _ in range(self.dim)]))
            # if self.condition:
            #     self.wave_freqs_cond = nn.Linear(self.lat_size, self.fhidden * 2)
            self.norm_pos = nn.GroupNorm(1, self.fhidden)

        if self.attention:
            assert self.image_size % self.attn_patches == 0

            self.patch_size = self.image_size // self.attn_patches
            self.fattn = self.fhidden * self.patch_size

            self.wave_attn = ResidualAttentionBlock(self.fattn, dim=self.dim, residual=False, lat_size=lat_size)  # condition=condition

            if self.patch_size > 1:
                self.wave_to_patch = conv_fn(self.fhidden, self.fattn, self.patch_size, self.patch_size, 0)
                self.patch_to_wave = convtr_fn(self.fattn, self.fhidden, self.patch_size, self.patch_size, 0)

            self.norm_attn = nn.GroupNorm(1, self.fhidden)

        if self.memory:
            self.wave_to_mem = nn.Conv2d(self.fhidden, self.fhidden * self.mem_factor, 3, 1, 1)
            self.mem_to_wave = nn.Conv2d(self.fhidden * self.mem_factor, self.fhidden, 3, 1, 1)
            self.temp = (self.fhidden * self.mem_factor) ** -0.5
            self.norm_mem = nn.GroupNorm(1, self.fhidden)

        self.nl_conv = nn.Sequential(
            conv_fn(self.fhidden, self.fhidden, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            conv_fn(self.fhidden, self.fout, kernel_size=1, stride=1, padding=0)
        )

        self.l_conv = conv_fn(self.fhidden, self.fout, kernel_size=1, stride=1, padding=0)

    def forward(self, x, lat=None, pos_x=0, pos_y=0):
        assert (lat is None) == (not self.condition)

        x_wave = self.conv_wave(x)
        x_wave = self.norm_wave(x_wave)

        if self.condition:
            x_wave_mod_m, x_wave_mod_a = torch.chunk(self.wave_cond(lat).view(*([x.shape[0], self.fhidden * 2] + [1 for _ in range(self.dim)])), 2, 1)
            x_wave = x_wave + x_wave_mod_m * x_wave + x_wave_mod_a
            # x_wave = self.wave_cond(x_wave, lat)
            x_wave = self.norm_cond(x_wave)

        if self.pos_enc:
            x_wave_freqs = self.wave_freqs.repeat([x.shape[0], 1] + [1 for _ in range(self.dim)])

            # if self.image_size != self.canvas_size:
            #     x_wave_freqs_canvas = torch.zeros([x.shape[0], self.fhidden] + [self.canvas_size for _ in range(self.dim)])
            #     init_mid = (self.canvas_size // 2) - (self.image_size // 2)
            #     end_mid = (self.canvas_size // 2) + (self.image_size // 2)
            #     x_wave_freqs_canvas[:, :, init_mid:end_mid, init_mid:end_mid] = x_wave_freqs
            #     x_wave_freqs = x_wave_freqs_canvas

            x_wave_freqs = torch.fft.irfftn(x_wave_freqs, x_wave_freqs.shape[-self.dim:], dim=tuple(-d for d in range(self.dim, 0, -1)), norm="ortho")
            # if self.condition:
            #     x_wave_freqs_mod_m, x_wave_freqs_mod_a = torch.chunk(self.wave_freqs_cond(lat).view(*([x.shape[0], self.fhidden * 2] + [1 for _ in range(self.dim)])), 2, 1)
            #     x_wave_freqs = x_wave_freqs + x_wave_freqs_mod_m * x_wave_freqs + x_wave_freqs_mod_a

            # if self.image_size != self.canvas_size:
            #     x_wave_freqs = x_wave_freqs[:, :, pos_x:pos_x+self.image_size, pos_y:pos_y+self.image_size]

            x_wave = x_wave + x_wave_freqs
            x_wave = self.norm_pos(x_wave)

        if self.attention:
            x_wave_patch = x_wave
            if self.patch_size > 1:
                x_wave_patch = self.wave_to_patch(x_wave_patch)
            # if self.condition:
            #     x_wave_patch = self.wave_attn(x_wave_patch, lat)
            # else:
            x_wave_patch = self.wave_attn(x_wave_patch)
            if self.patch_size > 1:
                x_wave_patch = self.patch_to_wave(x_wave_patch)
            x_wave = x_wave + x_wave_patch
            x_wave = self.norm_attn(x_wave)

        if self.memory:
            mem = self.wave_to_mem(x_wave) * self.temp
            mem = F.softmax(mem, dim=1)
            mem = self.mem_to_wave(mem)
            x_wave = x_wave + mem
            x_wave = self.norm_mem(x_wave)

        x_wave = self.l_conv(x_wave) + self.nl_conv(x_wave)

        return x_wave
