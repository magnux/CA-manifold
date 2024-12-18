import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from src.utils.mnist_dataset import MNIST
from src.utils.celeba_dataset import CelebA
from src.utils.cifar_dataset import CIFAR10
from src.utils.pnga_dataset import PNGADataset
from src.utils.movingmnist_dataset import MovingMnistDataset
# from src.utils.captiondataset import CaptionDataset


def get_dataset(name, type, data_dir, size=32):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        # transforms.Lambda(lambda x: x + 1./128 * torch.randn_like(x)),
        # transforms.Lambda(lambda x: x.clamp_(-1.0, 1.0)),
    ])
    if type == 'image':
        if name == 'image':
            dataset = datasets.ImageFolder(data_dir, transform)
            # n_labels = len(dataset.classes)
        elif name == 'npy':
            # Only support normalization for now
            dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
            # n_labels = len(dataset.classes)
        elif name == 'mnist':
            dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
        elif name == 'celeba':
            dataset = CelebA(root=data_dir, split="train", target_type="attr", download=True, transform=transform)
        elif name == 'cifar10':
            dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        elif name == 'emoji':
            dataset = PNGADataset(data_dir, transform, True)
        else:
            raise NotImplemented
    elif type == 'video':
        if name == 'movingmnist':
            dataset = MovingMnistDataset(root=data_dir, seq_len=4, nums_per_image=2, fake_dataset_size=1024,
                                         speed=1, speed_type='fixed', reflect_end=True)
    # elif type == 'caption':
    #     if name in ('flowers', 'birds'):
    #         text_transform = transforms.Compose([
    #             transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    #             transforms.Lambda(lambda x: x + 1e-2 * torch.randn_like(x)),
    #             # transforms.Lambda(lambda x: x.clamp_(-10.0, 10.0)),
    #             transforms.Lambda(lambda x: x.clamp_(0.0, 1.0)),
    #         ])
    #         dataset = CaptionDataset(data_dir, preload=False, image_transform=transform, text_transform=text_transform)
    #     else:
    #         raise NotImplemented
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
