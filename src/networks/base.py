import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.centroids import Centroids
from src.layers.pos_encoding import PosEncoding
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock


class Classifier(nn.Module):
    def __init__(self, n_labels, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, n_labels)

    def forward(self, lat):
        return self.labs(lat)


class Discriminator(nn.Module):
    def __init__(self, n_labels, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
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
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)
        self.embed_to_lat = nn.Linear(z_dim + embed_size, self.lat_size)
        nn.init.xavier_normal_(self.embed_to_lat.weight, 0.1)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        lat = self.embed_to_lat(torch.cat([z, yembed], dim=1))

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
        self.embed_to_lat = nn.Linear(z_dim, self.lat_size)
        nn.init.xavier_normal_(self.embed_to_lat.weight, 0.1)

    def forward(self, z):
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
    def __init__(self, lat_size, letter_channels=64, **kwargs):
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
        codes = F.softmax(codes, dim=1)
        return codes


class CodeBookDecoder(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, letter_channels=64, n_cents=1024, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels

        self.pos_enc = PosEncoding(self.lat_size)
        self.codes_in = ResidualBlock(letter_channels + self.pos_enc.size(), letter_channels, None, 3, 1, 1, nn.Conv1d)

        self.n_calls = 8
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.codes_norm = nn.InstanceNorm3d(self.letter_channels)
        self.codes_sobel = SinSobel(letter_channels, 3, 1, 3)
        self.codes_conv = DynaResidualBlock(embed_size, self.letter_channels * 4, self.letter_channels, self.letter_channels, dim=3)

        self.codes_out = ResidualBlock(letter_channels, letter_channels, None, 3, 1, 1, nn.Conv1d)

        # self.centroids = Centroids(letter_channels, n_cents)
        self.codes_to_lat = nn.Sequential(
            ResidualBlock(letter_channels, letter_channels, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels, 1, None, 1, 1, 0, nn.Conv1d),
        )

        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)

    def forward(self, codes, y):
        batch_size = codes.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding_mat[y]
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)

        if self.training:
            perm_codes = codes[:, :, torch.randperm(self.lat_size)]
            rand_mask = torch.rand((batch_size, 1, self.lat_size), device=codes.device) > 0.5
            rand_mask[:batch_size//2, ...] = 0
            pred_codes = torch.where(rand_mask, perm_codes, codes)
        else:
            pred_codes = codes
        pred_codes = self.pos_enc(pred_codes)
        pred_codes = self.codes_in(pred_codes)
        pred_codes = pred_codes.reshape(batch_size, self.letter_channels, 8, 8, 8)
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for _ in range(self.n_calls):
            # pred_codes = F.pad(pred_codes, [0, 1, 0, 1, 0, 1])
            pred_codes_new = self.codes_norm(pred_codes)
            pred_codes_new = self.codes_sobel(pred_codes_new)
            pred_codes_new = self.codes_conv(pred_codes_new, yembed)
            pred_codes = pred_codes + (leak_factor * pred_codes_new)
            # pred_codes = pred_codes[:, :, 1:, 1:, 1:]

        pred_codes = pred_codes.reshape(batch_size, self.letter_channels, self.lat_size)
        pred_codes = self.codes_out(pred_codes)

        code_idx = torch.argmax(codes, dim=1)
        code_idx = code_idx.reshape(batch_size * self.lat_size)
        pred_codes_idx = pred_codes.permute(0, 2, 1).reshape(batch_size * self.lat_size, self.letter_channels)
        loss_cent = F.cross_entropy(pred_codes_idx, code_idx)

        # pred_codes, loss_cent = self.centroids(pred_codes)
        lat = self.codes_to_lat(pred_codes)
        lat = lat.squeeze(dim=1)

        return lat, loss_cent


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
