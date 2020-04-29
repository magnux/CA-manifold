import numpy as np
import torch
import torchvision
from src.utils.randompolygon import randomPolygons


def sample_codes(codes):
    rand_idxs = np.random.randint(0, codes.size(1), 1)
    codes = codes[:, rand_idxs, ...]
    codes = codes.squeeze(1)
    return codes


def save_images(imgs, outfile, nrow=8):
    imgs = (imgs * 0.5) + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def random_polygons(batch_size, im_size, device):
    polygons = [randomPolygons(im_size, np.random.randint(1, 6), 'np') for _ in range(batch_size)]
    polygons = np.stack(polygons, 0).transpose(0, 3, 1, 2)
    polygons = (torch.tensor(polygons, dtype=torch.float32) / 127.) - 1.
    polygons = polygons.to(device)
    return polygons


rerase = torchvision.transforms.RandomErasing()


def rand_erase_images(images):
    return torch.stack([rerase(images[i, ...]) for i in range(images.size(0))]).to(images.device)


def rand_erase_text(text):
    re_mask = torch.round_(torch.rand([text.size(0), 1, 1], device=text.device))
    re_mask = re_mask * torch.round_(torch.rand([text.size(0), 1, text.size(2)], device=text.device))
    re_mask = (1 - re_mask)
    return text * re_mask


def rand_change_letters(letters):
    noise_mask = torch.round_(torch.rand([letters.size(0), 1, 1], device=letters.device))
    noise_mask = noise_mask * torch.round_(torch.rand([letters.size(0), 1, letters.size(2)], device=letters.device))
    rand_letters = torch.softmax(torch.randn_like(letters) * 10., dim=1)
    return torch.where(noise_mask == 1, rand_letters, letters)


def rand_circle_masks(images, n_mask):
    b, c, h, w = images.size()
    x = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
    y = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
    center = torch.rand([2, n_mask, 1, 1]) - 0.5
    r = (torch.rand([n_mask, 1, 1]) * 0.3) + 0.1
    x, y = (x-center[0, ...])/r, (y-center[1, ...])/r
    mask = (x*x+y*y < 1.0).to(dtype=torch.float32).to(device=images.device)
    damage = torch.ones_like(images)
    damage[-n_mask:, ...] = damage[-n_mask:, ...] - mask.unsqueeze(1)
    return images * damage
