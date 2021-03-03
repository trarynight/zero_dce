import numpy as np
from PIL import Image
from random import shuffle

import torch
from torch.utils import data

import cv2
from albumentations import (
    Compose,
    Resize,
    )
from albumentations.pytorch import ToTensor, ToTensorV2


class LowLightDataset(data.Dataset):
    """Low-light image dataset

    Pytorch dataset for low-light images

    Args:
        image_files: List of image file paths
        image_size: size of each image
    """

    def __init__(self, image_files=None, image_size=256):
        self.image_files = image_files
        self.image_size = image_size
        shuffle(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        image = Image.open(image_path)
        image = image.resize(
            (self.image_size, self.image_size), Image.ANTIALIAS)
        image = (np.asarray(image) / 255.0)
        image = torch.from_numpy(image).float()
        image_data = image.permute(2, 0, 1)
        return image_data


class LowLightDatasetAlbu(data.Dataset):
    """Low-light image dataset

    Pytorch dataset for low-light images

    Args:
        image_files: List of image file paths
        image_size: size of each image
    """

    def __init__(self, image_files=None, image_size=256):
        self.image_files = image_files
        self.image_size = image_size
        self.transform = Compose([
            Resize(height=image_size, width=image_size),
            ToTensor(),
        ])
        shuffle(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_files[idx]), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image