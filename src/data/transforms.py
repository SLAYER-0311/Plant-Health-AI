"""
PlantHealth AI - Image Transforms
====================================
Albumentations-based augmentation pipelines for training, validation, and inference.
"""

from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """
    Augmentation pipeline for training images.

    Pipeline:
        - Resize to image_size
        - Horizontal flip
        - Random rotation (±15°)
        - Color jitter (brightness, contrast, saturation, hue)
        - Gaussian blur / motion blur
        - Gaussian noise
        - Optical / grid / elastic distortion
        - CoarseDropout (cutout)
        - Normalize (ImageNet) + ToTensorV2
    """
    cfg = augmentation_config or {}

    transforms = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=cfg.get("horizontal_flip", 0.5)),
        A.Rotate(
            limit=cfg.get("rotation_degrees", 15),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.ColorJitter(
            brightness=cfg.get("brightness", 0.2),
            contrast=cfg.get("contrast", 0.2),
            saturation=cfg.get("saturation", 0.2),
            hue=cfg.get("hue", 0.1),
            p=0.5,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=1.0),
            A.GridDistortion(p=1.0),
            A.ElasticTransform(p=1.0),
        ], p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 8,
            max_width=image_size // 8,
            min_holes=1,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation / test transforms — only resize and normalize."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Inference transforms (same as val)."""
    return get_val_transforms(image_size)


def visualize_augmentations(
    image: np.ndarray,
    transform: A.Compose,
    num_samples: int = 6,
) -> List[np.ndarray]:
    """
    Apply transform N times and return list of augmented images (as uint8 HWC arrays).
    """
    augmented_images = []
    for _ in range(num_samples):
        aug = transform(image=image)
        img_tensor = aug["image"]  # CHW float tensor
        # Denormalize and convert back to HWC uint8
        img_np = denormalize(img_tensor.numpy())
        augmented_images.append(img_np)
    return augmented_images


def denormalize(
    tensor: np.ndarray,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverse ImageNet normalization.
    Accepts CHW numpy array, returns HWC uint8 numpy array.
    """
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    img = tensor * std_arr + mean_arr          # CHW float [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)              # HWC uint8
