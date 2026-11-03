"""
Legacy configuration module maintained for full backward compatibility.
Proxies to src.utils.config.
"""

from src.utils.config import (
    BASE_DIR,
    PARENT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    NUM_CLASSES,
    load_config,
)

__all__ = [
    "BASE_DIR",
    "PARENT_DIR",
    "IMG_SIZE",
    "BATCH_SIZE",
    "SEED",
    "NUM_CLASSES",
    "load_config",
]
