import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from src.utils.captiondataset import CaptionDataset


def get_dataset(name, type, data_dir, size=64, lsun_categories=None):
    transform = transforms.Compose([
        transforms.Resize(size),
        # transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(15, (0.1, 0.1), (0.9, 1.1), 5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.randn_like(x)),
        transforms.Lambda(lambda x: x.clamp_(-1.0, 1.0)),
    ])
    if type == 'image':
        if name == 'image':
            dataset = datasets.ImageFolder(data_dir, transform)
            # n_labels = len(dataset.classes)
        elif name == 'npy':
            # Only support normalization for now
            dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
            # n_labels = len(dataset.classes)
        elif name == 'cifar10':
            dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                       transform=transform)
            # n_labels = 10
        elif name == 'lsun':
            if lsun_categories is None:
                lsun_categories = 'train'
            dataset = datasets.LSUN(data_dir, lsun_categories, transform)
            # n_labels = len(dataset.classes)
        elif name == 'lsun_class':
            dataset = datasets.LSUNClass(data_dir, transform,
                                         target_transform=(lambda t: 0))
            # n_labels = 1
        else:
            raise NotImplemented
    elif type == 'caption':
        if name in ('flowers', 'birds'):
            text_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                transforms.Lambda(lambda x: x + 1e-2 * torch.randn_like(x)),
                # transforms.Lambda(lambda x: x.clamp_(-10.0, 10.0)),
                transforms.Lambda(lambda x: x.clamp_(0.0, 1.0)),
            ])
            dataset = CaptionDataset(data_dir, preload=False, image_transform=transform, text_transform=text_transform)
        else:
            raise NotImplemented
    else:
        raise NotImplemented

    return dataset


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
