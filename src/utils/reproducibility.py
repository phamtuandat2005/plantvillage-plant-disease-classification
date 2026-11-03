"""
Reproducibility utilities for setting random seeds and logging hardware/runtime environment.
"""

import os
import random
import sys
import platform
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 425, deterministic: bool = True) -> dict:
    """
    Sets seeds for Python, NumPy, and TensorFlow for experimental reproducibility.
    Optionally enables deterministic operations in TensorFlow.

    Parameters:
        seed (int): The integer random seed (default: 425).
        deterministic (bool): Whether to enforce deterministic TensorFlow ops.

    Returns:
        dict: Status report of the seed configuration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    deterministic_status = False
    if deterministic:
        try:
            # TensorFlow 2.8+ deterministic operations
            tf.config.experimental.enable_op_determinism()
            deterministic_status = True
        except Exception as e:
            deterministic_status = False
            print(f"[Warning] Could not enable deterministic ops: {e}")

    return {
        "seed": seed,
        "deterministic_enabled": deterministic_status,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def get_environment_info() -> dict:
    """
    Retrieves runtime environment details including Python, TensorFlow, OS, and GPU status.

    Returns:
        dict: Environment metadata.
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [gpu.name for gpu in gpus]

    return {
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "embedded"),
        "os": platform.system(),
        "os_release": platform.release(),
        "processor": platform.processor(),
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpu_devices": gpu_names,
    }

