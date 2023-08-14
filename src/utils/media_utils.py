import numpy as np
import torch
import torch.nn.functional as F
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


def rand_circle_masks(images, mask=None):
    b, c, h, w = images.size()
    if mask is None:
        x = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
        y = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
        center = torch.rand([2, b, 1, 1]) - 0.5
        r = (torch.rand([b, 1, 1]) * 0.3) + 0.1
        x, y = (x-center[0, ...])/r, (y-center[1, ...])/r
        mask = (x*x+y*y < 1.0).to(dtype=torch.float32).to(device=images.device).unsqueeze(1)
    damage = torch.ones_like(images) - mask
    cropped_images = images * damage
    mushed_images = cropped_images + mask * (images * mask).mean(dim=(2, 3), keepdim=True)
    shuffled_images = images.reshape(b, c, h * w)[:, :, np.random.choice(np.arange(0, h * w), h * w, replace=False)].reshape(b, c, h, w)
    shuffled_images = cropped_images + mask * shuffled_images
    return cropped_images, mushed_images, shuffled_images, mask


EMB_MATRIX = torch.eye(256, device='cpu')


def pix_to_onehot(images):
    images_oh = ((images + 1) * 127.5).long()
    images_oh = images_oh.permute(0, 2, 3, 1).contiguous().view(images.size(0) * images.size(2) * images.size(3), images.size(1))
    images_oh = EMB_MATRIX[images_oh]
    images_oh = images_oh.view(images.size(0), images.size(2), images.size(3), images.size(1) * 256).permute(0, 3, 1, 2).contiguous()
    return images_oh


def mask_templates(image_size, n_scales=None, anti_sym=True, only_last=False):
    if n_scales is None:
        n_scales = image_size
    mask_temps = []
    max_size = min(2 ** (n_scales + 1), image_size)

    sizes = [2 ** s for s in range(int(np.log2(max_size)) if only_last else 1, int(np.log2(max_size)) + 1)]
    for s in sizes:
        mask = np.indices((1, 1, s, 1)).sum(axis=0) % 2
        mask = np.tile(mask, (1, 1, 1, s))
        mask = F.interpolate(torch.tensor(mask, dtype=torch.uint8), size=image_size, mode='nearest')
        mask_temps.append(mask)
        mask = mask.permute(0, 1, 3, 2)
        mask_temps.append(mask)
        if anti_sym:
            mask = np.indices((1, 1, s, s)).sum(axis=0) % 2
            mask = F.interpolate(torch.tensor(mask, dtype=torch.uint8), size=image_size, mode='nearest')
            mask_temps.append(mask)
    anti_temps = []
    for mask in mask_temps:
        anti_temps.append(1 - mask)
    mask_temps.extend(anti_temps)
    return mask_temps


def mask_jamm(images_a, images_b, mask_templates, multi_mask=True, idx=None, return_idx=False, jamm_ratio=1., scale_factor=1., mask_weight=None):
    if scale_factor != 1.:
        images_a = F.interpolate(images_a, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        images_b = F.interpolate(images_b, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    batch_size = images_a.size(0)
    if multi_mask:
        # Multi mask per batch
        if idx is None:
            idx = np.random.choice(range(len(mask_templates)), replace=True, size=batch_size).astype(int)
        rand_masks = [mask_templates[i] for i in idx]
        rand_masks = torch.cat(rand_masks).to(images_a.device)
    else:
        # Single mask per batch
        if idx is None:
            idx = np.random.randint(len(mask_templates))
        rand_masks = mask_templates[idx].to(images_a.device)

    if jamm_ratio < 1:
        ratio_mask = (torch.rand((batch_size, images_a.size(1), 1, 1), device=images_a.device) < jamm_ratio).float()
        rand_masks = ratio_mask * rand_masks + (1 - ratio_mask)

    if mask_weight is None:
        images_out = rand_masks * images_a + (1 - rand_masks) * images_b
    else:
        images_out = rand_masks * images_a + (1 - rand_masks) * (mask_weight * images_b + (1 - mask_weight) * images_a)

    if scale_factor != 1.:
        images_out = F.interpolate(images_out, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
        # rand_masks = F.interpolate(rand_masks, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)

    if return_idx:
        return images_out, idx
    else:
        return images_out


def tensor_jamm(tensor_a, tensor_b, fussion_progress=0.5, collapse_dims=()):
    jamm_mask_shape = [1 if i in collapse_dims else d for i, d in enumerate(tensor_a.shape)]
    jamm_mask = torch.rand(jamm_mask_shape, device=tensor_a.device)
    jamm_mask = (jamm_mask < fussion_progress).float()
    return (1 - jamm_mask) * tensor_a + jamm_mask * tensor_b


def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1 / (2*torch.pi*sx*sy) * \
      torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))


def gaussian_blobs(h, w, origins):
    x = torch.linspace(0, h, h)
    y = torch.linspace(0, w, w)
    x, y = torch.meshgrid(x, y)

    z = torch.zeros(h, w)
    for x0, y0 in origins:
        z += gaussian_2d(x, y, mx=x0, my=y0, sx=h / 10, sy=w / 10)

    return z


def gaussian_blob_masks(batch_size, h, w, blob_n):
    masks = []
    for _ in range(batch_size):
        orig_h = torch.randint(0, h, (blob_n, 1))
        orig_w = torch.randint(0, w, (blob_n, 1))
        origins = torch.cat([orig_h, orig_w], dim=1)
        new_mask = gaussian_blobs(h, w, origins)
        new_mask = (new_mask - new_mask.min()) / (new_mask.max() - new_mask.min())
        if torch.rand([]) > 0.5:
            new_mask = 1 - new_mask
        masks.append(new_mask)
    return torch.stack(masks)[:, None, :, :]


def gaussian_blob_jamm(images_a, images_b, blob_n=8):
    masks = gaussian_blob_masks(images_a.shape[0], images_a.shape[2], images_a.shape[3], blob_n).to(images_a.device)
    return masks * images_a + (1 - masks) * images_b


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_size = 16
    masks = mask_templates(img_size, 2)

    for i in range(len(masks)):
        plt.imshow(masks[i][0, 0].detach().numpy())
        plt.show()
