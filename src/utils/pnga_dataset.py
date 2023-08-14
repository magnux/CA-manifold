from torch.utils.data import Dataset
import torch
from PIL import Image
from glob import glob


class PNGADataset(Dataset):
    def __init__(self, data_dir, transform, preload=False):
        self.data_dir = data_dir
        self.transform = transform
        self.preload = preload
        self.classes = sorted([file for file in glob(data_dir + "/*")])
        self.num_classes = len(self.classes)
        self.images = []
        self.labels = []
        self.glob_to_class_idx = []
        for class_idx, class_dir in enumerate(self.classes):
            class_images = sorted([file for file in glob(class_dir + "/*.png")])
            self.images += [class_images]
            self.labels += [[class_idx for _ in range(len(class_images))]]
            for img_idx in range(len(class_images)):
                self.glob_to_class_idx.append((class_idx, img_idx))

        if self.preload:
            self.images_buffer = []
            for idx in range(len(self.glob_to_class_idx)):
                self.images_buffer.append(self._load_image(idx))

    def get_equalized_class_sampler(self):
        samples_weight = torch.tensor([1 / len(self.images[class_idx]) for class_idx in range(self.num_classes) for _ in range(len(self.images[class_idx]))])
        return torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    def _load_image(self, idx):
        class_idx, img_idx = self.glob_to_class_idx[idx]
        img_loc = self.images[class_idx][img_idx]
        image = Image.open(img_loc).convert("RGBA")
        return self.transform(image)

    def __len__(self):
        return len(self.glob_to_class_idx)

    def __getitem__(self, idx):
        if self.preload:
            transformed_image = self.images_buffer[idx]
        else:
            transformed_image = self._load_image(idx)
        class_idx, img_idx = self.glob_to_class_idx[idx]
        label = self.labels[class_idx][img_idx]
        return transformed_image, label, idx
