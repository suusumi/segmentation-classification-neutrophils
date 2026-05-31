"""Albumentations transforms for blood cell image classification."""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Build training transforms for blood cell classification.

    Args:
        image_size: Target square size for resizing.

    Returns:
        Albumentations compose object producing PyTorch tensors.
    """

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Build validation transforms for blood cell classification.

    Args:
        image_size: Target square size for resizing.

    Returns:
        Albumentations compose object producing PyTorch tensors.
    """

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_segmentation_train_transforms(image_size: int = 256) -> A.Compose:
    """Build training transforms for nucleus segmentation.

    Geometric transforms are applied to the image and mask together. Color and
    noise transforms are applied only to the image by albumentations.
    """

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_REFLECT_101,
                fill_mask=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_segmentation_val_transforms(image_size: int = 256) -> A.Compose:
    """Build validation transforms for nucleus segmentation."""

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_lobe_count_train_transforms(image_size: int = 256) -> A.Compose:
    """Build online transforms for nucleus lobe count training."""

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_REFLECT_101,
                fill_mask=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"lobe_instance_mask": "mask"},
    )


def get_lobe_count_val_transforms(image_size: int = 256) -> A.Compose:
    """Build validation transforms for nucleus lobe count baselines."""

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"lobe_instance_mask": "mask"},
    )


def get_lobe_segmentation_train_transforms(image_size: int = 256) -> A.Compose:
    """Build online transforms for lobe foreground/boundary segmentation."""

    return get_lobe_count_train_transforms(image_size=image_size)


def get_lobe_segmentation_val_transforms(image_size: int = 256) -> A.Compose:
    """Build validation transforms for lobe foreground/boundary segmentation."""

    return get_lobe_count_val_transforms(image_size=image_size)
