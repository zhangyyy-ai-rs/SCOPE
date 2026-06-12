"""ImageFolder data helpers."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_imagefolder_loaders(data_root: str | Path, image_size: int = 224, batch_size: int = 64, workers: int = 4):
    root = Path(data_root)
    train_set = datasets.ImageFolder(root / 'train', transform=build_transforms(image_size, train=True))
    val_set = datasets.ImageFolder(root / 'val', transform=build_transforms(image_size, train=False))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, len(train_set.classes)
