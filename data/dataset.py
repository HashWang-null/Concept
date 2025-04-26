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
        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.scale = transforms.Lambda(lambda x: x * 2 - 1.0)  # [0,1] -> [-1,1]

    def __len__(self):
        return len(self.sources)

    def random_crop(self, img1, img2, crop_size):
        assert img1.size == img2.size
        w, h = img1.size
        th, tw = crop_size, crop_size
        if w < tw or h < th:
            raise ValueError(f"Image too small for cropping: got ({w},{h}), crop size ({tw},{th})")
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

        # Auto upscale if short side < crop_size
        min_side = min(source.size)  # PIL image, size = (w,h)
        if min_side < self.image_size:
            scale = self.image_size / min_side
            new_size = (int(round(source.width * scale)), int(round(source.height * scale)))
            source = source.resize(new_size, resample=Image.BILINEAR)
            target = target.resize(new_size, resample=Image.BILINEAR)

        # Regular resize to (image_size, image_size) if needed
        source = self.resize(source)
        target = self.resize(target)

        # Random crop
        source, target = self.random_crop(source, target, self.image_size)

        source = self.to_tensor(source)
        target = self.to_tensor(target)

        source = self.scale(source)
        target = self.scale(target)

        return {
            'source': source,
            'target': target,
            'source_path': self.sources[index],
            'target_path': self.targets[index],
        }


class DevDataset(Dataset):
    def __init__(self, length=1000, image_size=512, **kwargs):
        super().__init__()
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = torch.randn(3, self.image_size, self.image_size)
        img = torch.clamp(img, -3, 3) / 3.0

        return {
            'source': img.clone(),
            'target': img.clone(),
            'source_path': f"dev_source_{index}.png",
            'target_path': f"dev_target_{index}.png",
        }
