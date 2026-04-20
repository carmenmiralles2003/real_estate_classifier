"""
Dataset utilities: loading, splitting, and augmentation for real-estate images.
"""
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    IMG_SIZE, IMG_MEAN, IMG_STD,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DEFAULT_HPARAMS,
)


def get_train_transforms(img_size: int = IMG_SIZE, augmentation: str = "medium") -> transforms.Compose:
    """
    Augmentation pipeline for training.

    Levels:
        - light:  resize + flip only
        - medium: + color jitter + rotation
        - heavy:  + perspective + erasing + affine
    """
    base = [transforms.Resize((img_size, img_size))]

    if augmentation == "light":
        base += [
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == "medium":
        base += [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
        ]
    elif augmentation == "heavy":
        base += [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(25),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ]

    if augmentation == "heavy":
        base.append(transforms.RandomErasing(p=0.2))

    return transforms.Compose(base)


def get_val_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic pipeline for validation and test."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


def split_dataset(
    source_dir: Path = RAW_DATA_DIR,
    dest_dir: Path = PROCESSED_DATA_DIR,
    train_ratio: float = DEFAULT_HPARAMS["train_split"],
    val_ratio: float = DEFAULT_HPARAMS["val_split"],
    seed: int = DEFAULT_HPARAMS["seed"],
):
    """
    Splits a flat ImageFolder dataset (source_dir/<class>/*.jpg)
    into train / val / test subfolders under dest_dir.
    """
    random.seed(seed)

    for split in ("train", "val", "test"):
        (dest_dir / split).mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(
            f"No class sub-folders found in {source_dir}. "
            "Place images in data/raw/<class_name>/ and re-run."
        )

    for class_dir in class_dirs:
        images = sorted(
            [f for f in class_dir.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        )
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            split_class_dir = dest_dir / split_name / class_dir.name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, split_class_dir / f.name)

    print(f"Dataset split completed → {dest_dir}")


def get_dataloaders(
    data_dir: Path = PROCESSED_DATA_DIR,
    batch_size: int = DEFAULT_HPARAMS["batch_size"],
    num_workers: int = DEFAULT_HPARAMS["num_workers"],
    img_size: int = IMG_SIZE,
    augmentation: str = DEFAULT_HPARAMS["augmentation"],
):
    """
    Returns train, val, test DataLoaders from a pre-split directory.
    """
    train_ds = datasets.ImageFolder(
        data_dir / "train", transform=get_train_transforms(img_size, augmentation),
    )
    val_ds = datasets.ImageFolder(
        data_dir / "val", transform=get_val_transforms(img_size),
    )
    test_ds = datasets.ImageFolder(
        data_dir / "test", transform=get_val_transforms(img_size),
    )

    common = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    split_dataset()
