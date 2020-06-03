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
        self.images = []
        self.labels = []
        for class_idx, class_dir in enumerate(self.classes):
            class_images = sorted([file for file in glob(class_dir + "/*.png")])
            self.images += class_images
            self.labels += [class_idx for _ in range(len(class_images))]
        if self.preload:
            self.images_buffer = []
            for idx in range(len(self.images)):
                self.images_buffer.append(self._load_image(idx))

    def _load_image(self, idx):
        img_loc = self.images[idx]
        image = Image.open(img_loc).convert("RGBA")
        return self.transform(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.preload:
            transformed_image = self.images_buffer[idx]
        else:
            transformed_image = self._load_image(idx)
        label = self.labels[idx]
        return transformed_image, label
