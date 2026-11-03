"""
Image loading and preprocessing pipeline for VGG16.

IMPORTANT: VGG16 expects images processed with `tf.keras.applications.vgg16.preprocess_input`.
This converts RGB [0, 255] to BGR and zero-centers each color channel with respect
to the ImageNet dataset (subtracting [103.939, 116.779, 123.68]).
DO NOT apply Rescaling(1./255) as that corrupts the expected input distribution.
"""

import os
from pathlib import Path
from typing import Tuple, Union
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


def resolve_image_path(path_str: str, base_dir: Union[str, Path] = None) -> str:
    """
    Ensures image path exists, resolving relative or cross-platform differences if necessary.

    Parameters:
        path_str (str): Stored path from CSV.
        base_dir (str or Path, optional): Workspace base directory.

    Returns:
        str: Valid absolute or resolved path.
    """
    if os.path.exists(path_str):
        return path_str

    # Normalize path separators
    norm = path_str.replace("\\", "/")
    # Try finding relative to plantvillage_dataset
    if "plantvillage_dataset" in norm:
        subpath = norm[norm.index("plantvillage_dataset"):]
        if base_dir:
            candidate = os.path.join(str(base_dir), subpath)
            if os.path.exists(candidate):
                return candidate
    return path_str


def load_raw_image(path: tf.Tensor, img_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """
    Loads and resizes an RGB image as float32 in range [0, 255].
    Does NOT apply VGG16 zero-centering. Useful for augmentation or visualization.

    Parameters:
        path (tf.Tensor): Path string tensor.
        img_size (tuple): Target (height, width).

    Returns:
        tf.Tensor: Decoded and resized image tensor (shape: [H, W, 3], range [0, 255], float32).
    """
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32)
    return image


def preprocess_vgg16_image(image: tf.Tensor) -> tf.Tensor:
    """
    Applies VGG16-specific preprocessing:
    Converts RGB [0, 255] -> BGR and subtracts ImageNet channel means.

    Parameters:
        image (tf.Tensor): Tensor with values in [0, 255].

    Returns:
        tf.Tensor: VGG16 zero-centered BGR image tensor.
    """
    return preprocess_input(image)


def load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    img_size: Tuple[int, int] = (224, 224)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    End-to-end mapping function for tf.data pipeline (evaluation/testing/inference):
    Reads file -> decodes RGB -> resizes to (224, 224) -> applies vgg16.preprocess_input.

    Parameters:
        path (tf.Tensor): Path string tensor.
        label (tf.Tensor): Integer class index.
        img_size (tuple): Target (height, width).

    Returns:
        tuple: (preprocessed_image, label).
    """
    image = load_raw_image(path, img_size=img_size)
    image = preprocess_vgg16_image(image)
    return image, label


def deprocess_vgg16_image(image_tensor: tf.Tensor) -> tf.Tensor:
    """
    Inverts VGG16 preprocessing for visualization and Grad-CAM overlay:
    Adds back ImageNet mean and converts BGR back to RGB, clipping to [0, 255].

    Parameters:
        image_tensor (tf.Tensor): Preprocessed image tensor.

    Returns:
        tf.Tensor: Unnormalized RGB image in [0, 255] uint8.
    """
    x = image_tensor.numpy() if hasattr(image_tensor, "numpy") else image_tensor
    x = x.copy()
    if x.ndim == 4:
        x = x[0]
    # Invert zero-centering
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # BGR -> RGB
    x = x[:, :, ::-1]
    return tf.clip_by_value(x, 0, 255).numpy().astype("uint8")

