"""
PlantHealth AI - Dataset Module
=================================
PyTorch Dataset and DataLoader utilities for the PlantVillage dataset.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .transforms import get_train_transforms, get_val_transforms


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for the PlantVillage plant disease classification dataset.
    Loads images from a directory structure: root/class_name/image.jpg
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            root_dir: Path to dataset directory (contains class subdirectories)
            transform: Albumentations transform pipeline
            class_to_idx: Optional pre-defined class-to-index mapping (for val/test sets)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Use torchvision's ImageFolder to discover classes and files
        self._folder = datasets.ImageFolder(root=str(self.root_dir))

        if class_to_idx is not None:
            # Override class_to_idx (e.g. so val/test use same mapping as train)
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx, key=class_to_idx.get)
        else:
            self.class_to_idx = self._folder.class_to_idx
            self.classes = self._folder.classes

        # Re-map sample indices to the (potentially overridden) class_to_idx
        self.samples = [
            (path, self.class_to_idx[self._folder.classes[orig_idx]])
            for path, orig_idx in self._folder.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img_np = np.array(image)
            augmented = self.transform(image=img_np)
            image = augmented["image"]
        else:
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        dist: Dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            dist[self.classes[label]] += 1
        return dist


def get_sample_images(
    dataset: PlantDiseaseDataset,
    num_samples: int = 1,
    class_name: Optional[str] = None,
) -> List[Tuple[np.ndarray, str, str]]:
    """
    Retrieve raw numpy images (no transform) from the dataset.

    Returns:
        List of (image_array, class_name, file_path) tuples
    """
    results = []

    if class_name is not None:
        if class_name not in dataset.class_to_idx:
            raise ValueError(f"Class '{class_name}' not found in dataset")
        target_label = dataset.class_to_idx[class_name]
        indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == target_label]
    else:
        indices = list(range(len(dataset)))

    selected = random.sample(indices, min(num_samples, len(indices)))

    for idx in selected:
        img_path, label = dataset.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        cls = dataset.classes[label]
        results.append((image, cls, img_path))

    return results


def get_data_loaders(
    train_dir: str,
    valid_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    augmentation_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[str]]:
    """
    Create train and validation DataLoaders.

    Returns:
        (train_loader, val_loader, class_to_idx, class_names)
    """
    train_transforms = get_train_transforms(
        image_size=image_size,
        augmentation_config=augmentation_config or {},
    )
    val_transforms = get_val_transforms(image_size=image_size)

    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transforms)
    val_dataset = PlantDiseaseDataset(
        valid_dir,
        transform=val_transforms,
        class_to_idx=train_dataset.class_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.class_to_idx, train_dataset.classes
