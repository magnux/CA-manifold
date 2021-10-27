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


def rand_circle_masks(images):
    b, c, h, w = images.size()
    x = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
    y = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
    center = torch.rand([2, b, 1, 1]) - 0.5
    r = (torch.rand([b, 1, 1]) * 0.3) + 0.1
    x, y = (x-center[0, ...])/r, (y-center[1, ...])/r
    mask = (x*x+y*y < 1.0).to(dtype=torch.float32).to(device=images.device).unsqueeze(1)
    damage = torch.ones_like(images) - mask
    cropped_images = images * damage
    mushed_images = cropped_images + mask * (images * mask).mean()
    shuffled_images = images.reshape(b, c, h * w)[:, :, np.random.choice(np.arange(0, h * w), h * w, replace=False)].reshape(b, c, h, w)
    shuffled_images = cropped_images + mask * (shuffled_images * mask).mean()
    return cropped_images, mushed_images, shuffled_images


EMB_MATRIX = torch.eye(256, device='cpu')


def pix_to_onehot(images):
    images_oh = ((images + 1) * 127.5).long()
    images_oh = images_oh.permute(0, 2, 3, 1).contiguous().view(images.size(0) * images.size(2) * images.size(3), images.size(1))
    images_oh = EMB_MATRIX[images_oh]
    images_oh = images_oh.view(images.size(0), images.size(2), images.size(3), images.size(1) * 256).permute(0, 3, 1, 2).contiguous()
    return images_oh
