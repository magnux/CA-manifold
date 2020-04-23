from torch.utils.data import Dataset
import torch
from PIL import Image
from glob import glob


class PNGADataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([file for file in glob(data_dir + "/*")])
        self.images = []
        self.labels = []
        for class_idx, class_dir in enumerate(self.classes):
            class_images = sorted([file for file in glob(class_dir + "/*.png")])
            self.images += class_images
            self.labels += [class_idx for _ in range(len(class_images))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = self.images[idx]
        image = Image.open(img_loc).convert("RGBA")
        tensor_image = self.transform(image)
        label = self.labels[idx]
        return tensor_image, label
