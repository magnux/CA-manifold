import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.irm import IRMLinear
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
from src.utils.model_utils import ca_seed
from itertools import chain
import numpy as np


class Classifier(nn.Module):
    def __init__(self, n_labels, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, n_labels)

    def forward(self, lat):
        return self.labs(lat)


class Discriminator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_weight = nn.Sequential(
            nn.Linear(n_labels, embed_size),
            LinearResidualBlock(embed_size, embed_size),
            LinearResidualBlock(embed_size, lat_size, int(embed_size ** 0.5)),
        )

    def forward(self, lat, y):
        assert(lat.size(0) == y.size(0))
        batch_size = lat.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        lat_weight = self.labs_to_weight(yembed)
        lat_weight = lat_weight.view(batch_size, self.lat_size, 1)
        lat = lat.view(batch_size, 1, self.lat_size)
        score = torch.bmm(lat, lat_weight).squeeze(1)

        return score


class Generator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_weight = nn.Sequential(
            nn.Linear(n_labels, embed_size),
            LinearResidualBlock(embed_size, embed_size),
            LinearResidualBlock(embed_size, z_dim * lat_size, int(embed_size ** 0.5)),
        )

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        lat_weight = self.labs_to_weight(yembed)
        lat_weight = lat_weight.view(batch_size, self.z_dim, self.lat_size)
        z = z.clamp(-3, 3).view(batch_size, 1, self.z_dim)
        lat = torch.bmm(z, lat_weight).squeeze(1)

        return lat


class LabsEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, lat_size, bias=False)

    def forward(self, y):
        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)

        return yembed


class UnconditionalDiscriminator(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, 1, bias=False)

    def forward(self, lat):
        labs = self.labs(lat)

        return labs


class UnconditionalGenerator(nn.Module):
    def __init__(self, lat_size, z_dim, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.embed_to_lat = nn.Linear(z_dim, self.lat_size, bias=False)

    def forward(self, z):
        z = z.clamp(-3, 3)
        lat = self.embed_to_lat(z)

        return lat


class VarEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)
        self.lat_to_z = nn.Sequential(
            LinearResidualBlock(self.lat_size + embed_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, z_dim * 2),
        )

    def forward(self, lat, y):
        assert (lat.size(0) == y.size(0))
        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        z = self.lat_to_z(torch.cat([lat, yembed], dim=1))
        z_mu, z_log_var = torch.split(z, self.z_dim, dim=1)

        return z_mu, z_log_var


class VarDecoder(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)
        self.z_to_lat = nn.Sequential(
            LinearResidualBlock(z_dim + embed_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
        )

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        lat = self.z_to_lat(torch.cat([z, yembed], dim=1))

        return lat


class LetterEncoder(nn.Module):
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.lat_to_letters = nn.Sequential(
            ResidualBlock(1, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
        )

    def forward(self, lat):
        lat = lat.view(lat.size(0), 1, self.lat_size)
        letters = self.lat_to_letters(lat)
        letters = letters.view(letters.size(0), self.letter_channels, self.letter_bits * self.lat_size)
        letters = F.softmax(letters, dim=1)

        return letters


class LetterDecoder(nn.Module):
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.letters_to_lat = nn.Sequential(
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, 1, None, 1, 1, 0, nn.Conv1d),
        )

    def forward(self, letters):
        letters = letters.view(letters.size(0), self.letter_channels * self.letter_bits, self.lat_size)
        lat = self.letters_to_lat(letters)
        lat = lat.squeeze(dim=1)

        return lat


class IRMTranslator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_weight = nn.Sequential(
            nn.Linear(n_labels, embed_size),
            LinearResidualBlock(embed_size, embed_size),
            LinearResidualBlock(embed_size, lat_size * lat_size, int(embed_size ** 0.5)),
        )
        self.irm_layer = IRMLinear(lat_size)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        lat_weight = self.labs_to_weight(yembed)
        lat_weight = lat_weight.view(batch_size, self.lat_size, self.lat_size)
        z = self.irm_layer(z).view(batch_size, 1, self.lat_size)
        lat = torch.bmm(z, lat_weight).squeeze(1)

        return lat


class IRMGenerator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, n_filter, n_calls, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.n_filter = n_filter
        self.n_calls = n_calls * 4

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        self.frac_sobel = SinSobel(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(self.lat_size)), 1)],
                                                  [2 ** (i - 1) for i in range(1, int(np.log2(self.lat_size)), 1)], dim=1)
        self.frac_norm = nn.InstanceNorm1d(self.n_filter * self.frac_sobel.c_factor)
        self.frac_dyna_conv = DynaResidualBlock(self.z_dim + self.embed_size, self.n_filter * self.frac_sobel.c_factor, self.n_filter, self.n_filter, dim=1)

        self.out_conv = nn.Sequential(
            nn.InstanceNorm1d(self.n_filter),
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0, nn.Conv1d),
            nn.Conv1d(self.n_filter, 1, 1, 1, 0),
        )

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        z = z.clamp(-3, 3)
        lat = torch.cat([z, yembed], dim=1)

        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        out = ca_seed(batch_size, self.n_filter, self.lat_size, lat.device).to(float_type)[:, :, :, 0]

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = out

            out_new = self.frac_sobel(out_new)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)

            out = out + (leak_factor * out_new)

            out_embs.append(out)

        out = self.out_conv(out)
        lat = out.squeeze(1)

        return lat


class LatEncoder(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.lat_to_lat = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
        )

    def forward(self, lat):
        lat = self.lat_to_lat(lat)

        return lat
