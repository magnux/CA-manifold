import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.centroids import Centroids
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
from itertools import chain
import math


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
        self.labs = nn.Linear(self.lat_size, n_labels)

    def forward(self, lat, y):
        assert(lat.size(0) == y.size(0))
        batch_size = lat.size(0)

        labs = self.labs(lat)
        index = torch.arange(0, batch_size, device=lat.device)
        labs = labs[index, y]

        return labs


class Generator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size, bias=False)
        self.embed_to_lat = nn.Linear(z_dim + embed_size, self.lat_size, bias=False)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        z = z.clamp(-3, 3)
        lat = self.embed_to_lat(torch.cat([z, yembed], dim=1))

        return lat


class LabsEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size, bias=False)
        self.embed_to_lat = nn.Linear(embed_size, self.lat_size, bias=False)

    def forward(self, y):
        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        lat = self.embed_to_lat(yembed)

        return lat


class UnconditionalDiscriminator(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, 1)

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


class CodeBookEncoder(nn.Module):
    def __init__(self, lat_size, letter_channels=16, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.lat_to_codes = nn.Sequential(
            ResidualBlock(1, letter_channels, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels, letter_channels, None, 1, 1, 0, nn.Conv1d),
        )

    def forward(self, lat):
        lat = lat.view(lat.size(0), 1, self.lat_size)
        codes = self.lat_to_codes(lat)
        return codes


class CodeBookDecoder(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, letter_channels=16, n_cents=1024, fire_rate=1.0, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.n_labels = n_labels
        self.fire_rate = fire_rate

        self.lat_size_cr = int(math.ceil(self.lat_size ** (1 / 3)))
        self.lat_pad = int(self.lat_size_cr ** 3) - self.lat_size

        self.n_calls = 8
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.codes_sobel = SinSobel(self.letter_channels, 5, 2, 3, True)
        self.codes_norm = nn.InstanceNorm3d(self.letter_channels * 4)
        self.codes_conv = DynaResidualBlock(embed_size, self.letter_channels * 4, self.letter_channels, dim=3)
        self.centroids = Centroids(letter_channels, n_cents)

        self.codes_to_lat = nn.Sequential(
            ResidualBlock(letter_channels, letter_channels, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels, 1, None, 1, 1, 0, nn.Conv1d),
        )

        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)

    def forward(self, codes, y):
        batch_size = codes.size(0)
        float_type = torch.float16 if isinstance(codes, torch.cuda.HalfTensor) else torch.float32

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)

        pred_codes, loss_cent = self.centroids(codes)
        if self.lat_pad > 0:
            pred_codes = F.pad(pred_codes, [0, self.lat_pad])
        pred_codes = pred_codes.reshape(batch_size, self.letter_channels, self.lat_size_cr, self.lat_size_cr, self.lat_size_cr)

        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for m in range(self.n_calls):
            pred_codes = F.pad(pred_codes, [0, 2, 0, 2, 0, 2])

            pred_codes_new = self.codes_sobel(pred_codes)
            pred_codes_new = self.codes_norm(pred_codes_new)
            pred_codes_new = self.codes_conv(pred_codes_new, yembed)

            if self.fire_rate < 1.0:
                pred_codes_new = pred_codes_new * (torch.rand([batch_size, 1, self.lat_size_cr + 2, self.lat_size_cr + 2, self.lat_size_cr + 2], device=codes.device) <= self.fire_rate).to(float_type)

            pred_codes = pred_codes + (leak_factor * pred_codes_new)

            pred_codes = pred_codes[:, :, 2:, 2:, 2:]

        pred_codes = pred_codes.reshape(batch_size, self.letter_channels, self.lat_size + self.lat_pad)
        if self.lat_pad > 0:
            pred_codes = pred_codes[:, :, :self.lat_size]

        lat = self.codes_to_lat(pred_codes)
        lat = lat.squeeze(dim=1)

        return lat, loss_cent


class LetterGenerator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, letter_channels=4, letter_bits=16, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.letter_size = letter_channels * letter_bits
        self.n_labels = n_labels

        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)

        self.n_calls = 8
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.letter_sobel = SinSobel(self.letter_size, 5, 2, 3, left_sided=True)
        self.letter_norm = nn.InstanceNorm3d(self.letter_size * 4)
        self.letter_conv = DynaResidualBlock(embed_size, self.letter_size * 4, self.letter_size * 2, self.letter_size, dim=3)

    def forward(self, letters, y):
        batch_size = letters.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)

        lat_size_cr = int(round(self.lat_size ** (1/3)))
        lat_pad = int(lat_size_cr ** 3) - self.lat_size
        pred_letters = letters.reshape(batch_size, self.letter_size, self.lat_size)
        if lat_pad > 0:
            pred_letters = F.pad(pred_letters, [0, lat_pad])
        pred_letters = pred_letters.reshape(batch_size, self.letter_size, lat_size_cr, lat_size_cr, lat_size_cr)

        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for _ in range(self.n_calls):
            pred_letters = F.pad(pred_letters, [0, 2, 0, 2, 0, 2])

            pred_letters_new = self.letter_sobel(pred_letters)
            pred_letters_new = self.letter_norm(pred_letters_new)
            pred_letters_new = self.letter_conv(pred_letters_new, yembed)

            pred_letters_new, pred_letters_new_gate = torch.split(pred_letters_new, self.letter_channels, dim=1)
            pred_letters_new = pred_letters_new * torch.sigmoid(pred_letters_new_gate)

            pred_letters = pred_letters + (leak_factor * pred_letters_new)

            pred_letters = pred_letters[:, :, 2:, 2:, 2:]

        pred_letters = pred_letters.reshape(batch_size, self.letter_size, self.lat_size + lat_pad)
        if lat_pad > 0:
            pred_letters = pred_letters[:, :, :self.lat_size]
        pred_letters = pred_letters.reshape(batch_size, self.letter_channels, self.letter_bits * self.lat_size)

        return pred_letters


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
