import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from enum import Enum


class PairedDataset(Dataset):
    def __init__(self, source_files, target_files, image_size=512):
        super().__init__()
        self.sources = source_files
        self.targets = target_files
        self.image_size = image_size
        self.resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Lambda(lambda x: x[:3, :, :])  # 去除 alpha 通道
        self.scale = transforms.Lambda(lambda x: x * 2 - 1.0)  # 归一化到 [-1, 1]

    def __len__(self):
        return len(self.sources)

    def random_crop(self, img1, img2, crop_size):
        """对 img1 和 img2 执行相同位置的随机裁剪"""
        assert img1.size == img2.size
        w, h = img1.size
        th, tw = crop_size, crop_size
        if w == tw and h == th:
            return img1, img2
        x1 = torch.randint(0, w - tw + 1, (1,)).item()
        y1 = torch.randint(0, h - th + 1, (1,)).item()
        img1 = img1.crop((x1, y1, x1 + tw, y1 + th))
        img2 = img2.crop((x1, y1, x1 + tw, y1 + th))
        return img1, img2

    def __getitem__(self, index):
        source = Image.open(self.sources[index]).convert("RGB")
        target = Image.open(self.targets[index]).convert("RGB")

        source = self.resize(source)
        target = self.resize(target)

        # 随机裁剪保持一致
        source, target = self.random_crop(source, target, self.image_size)

        source = self.to_tensor(source)
        target = self.to_tensor(target)

        source = self.normalize(source)
        target = self.normalize(target)

        source = self.scale(source)
        target = self.scale(target)

        return {
            'source': source,
            'target': target,
            'source_path': self.sources[index],
            'target_path': self.targets[index],
        }
