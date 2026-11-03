"""
Backward compatible alias for data augmentation.
"""

from src.data.augmentation import build_augmentation_layer


def apply_augmentation():
    """Returns standard data augmentation pipeline."""
    return build_augmentation_layer()
