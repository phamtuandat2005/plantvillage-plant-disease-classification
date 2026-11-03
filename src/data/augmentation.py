"""
Data augmentation pipeline for training images.
Applies transformations to raw RGB images [0, 255] before VGG16 zero-centering.
Strictly adheres to the methodology workflow:
- Lật (RandomFlip horizontal and vertical)
- Xoay ±30° (RandomRotation factor = 30 / 360)
- Cắt (RandomCrop height=200, width=200 followed by Resizing back to 224x224)
- Đổi sáng (RandomBrightness factor = 0.2)
"""

from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation_layer(
    config: Optional[Dict[str, Any]] = None,
    seed: int = 425
) -> tf.keras.Sequential:
    """
    Creates a Sequential data augmentation layer for the training pipeline.
    Strictly follows the 4 specified operations in the methodology flowchart:
    1. Lật: RandomFlip("horizontal_and_vertical")
    2. Xoay ±30°: RandomRotation(factor=30/360 ≈ 0.0833)
    3. Cắt: RandomCrop(200, 200) + Resizing(224, 224)
    4. Đổi sáng: RandomBrightness(factor=0.2, value_range=(0, 255)) & RandomContrast(0.2)

    Parameters:
        config (dict, optional): Custom augmentation hyperparameters.
        seed (int): Seed for reproducibility.

    Returns:
        tf.keras.Sequential: Augmentation model to be mapped on training images.
    """
    cfg = config or {}
    # Rotation factor: 30 degrees / 360 degrees = 1/12 ≈ 0.083333
    rotation_factor = cfg.get("rotation_factor", 30.0 / 360.0)
    crop_size = cfg.get("crop_size", (200, 200))
    target_size = cfg.get("target_size", (224, 224))
    brightness_factor = cfg.get("brightness_factor", 0.2)
    contrast_factor = cfg.get("contrast_factor", 0.2)
    flip_mode = cfg.get("flip_mode", "horizontal_and_vertical")

    aug_layers = []

    # 1. Lật ảnh
    if flip_mode:
        aug_layers.append(
            layers.RandomFlip(
                mode=flip_mode,
                seed=seed,
                name="aug_random_flip"
            )
        )

    # 2. Xoay ±30° (factor = 30/360)
    if rotation_factor > 0:
        aug_layers.append(
            layers.RandomRotation(
                factor=rotation_factor,
                fill_mode="nearest",
                seed=seed,
                name="aug_random_rotation"
            )
        )

    # 3. Cắt (RandomCrop 200x200) và chuẩn hóa lại kích thước 224x224
    if crop_size and len(crop_size) == 2:
        aug_layers.append(
            layers.RandomCrop(
                height=crop_size[0],
                width=crop_size[1],
                seed=seed,
                name="aug_random_crop"
            )
        )
        aug_layers.append(
            layers.Resizing(
                height=target_size[0],
                width=target_size[1],
                name="aug_resizing"
            )
        )

    # 4. Đổi sáng
    if brightness_factor > 0:
        aug_layers.append(
            layers.RandomBrightness(
                factor=brightness_factor,
                value_range=(0.0, 255.0),
                seed=seed,
                name="aug_random_brightness"
            )
        )

    if contrast_factor > 0:
        aug_layers.append(
            layers.RandomContrast(
                factor=contrast_factor,
                seed=seed,
                name="aug_random_contrast"
            )
        )

    return tf.keras.Sequential(aug_layers, name="data_augmentation")
