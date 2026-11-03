"""
Configuration loader and path resolution utility.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict

# Workspace root directory (d:\2026-2027\computer-vision)
SRC_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = SRC_DIR.parent
PARENT_DIR = str(BASE_DIR)

DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "vgg16_config.json"

# Common defaults matching project conventions
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 425
NUM_CLASSES = 38


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads experiment configuration from JSON and resolves absolute paths.

    Parameters:
        config_path (str, optional): Custom path to config JSON file.

    Returns:
        dict: Parsed configuration dictionary with resolved paths.
    """
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve paths relative to workspace root
    raw_paths = config.get("paths", {})
    resolved_paths = {}

    for key, rel_path in raw_paths.items():
        abs_p = (BASE_DIR / rel_path).resolve()
        resolved_paths[key] = str(abs_p)
        if "dir" in key:
            abs_p.mkdir(parents=True, exist_ok=True)

    # Provide plural aliases for developer convenience
    aliases = {
        "color_dir": raw_paths.get("data_dir", "plantvillage_dataset/color"),
        "checkpoints_dir": raw_paths.get("checkpoint_dir", "outputs/vgg16/checkpoints"),
        "histories_dir": raw_paths.get("history_dir", "outputs/vgg16/histories"),
        "metrics_dir": raw_paths.get("metric_dir", "outputs/vgg16/metrics"),
        "plots_dir": raw_paths.get("plot_dir", "outputs/vgg16/plots"),
        "reports_dir": raw_paths.get("report_dir", "outputs/vgg16/reports"),
        "output_root": raw_paths.get("output_dir", "outputs/vgg16"),
    }

    for k, v in aliases.items():
        abs_v = (BASE_DIR / v).resolve()
        raw_paths.setdefault(k, str(v))
        resolved_paths.setdefault(k, str(abs_v))

    config["paths"] = raw_paths
    config["resolved_paths"] = resolved_paths

    # Alias dataset to data if needed
    if "dataset" in config and "data" not in config:
        config["data"] = config["dataset"]

    # Normalize experiment aliases
    exps = config.get("experiments", {})
    mapping = {
        "experiment_a": exps.get("exp_a", {}),
        "experiment_b": exps.get("exp_b", {}),
        "experiment_c": exps.get("exp_c", {}),
        "exp_a": exps.get("exp_a", {}),
        "exp_b": exps.get("exp_b", {}),
        "exp_c": exps.get("exp_c", {}),
    }
    for exp_key, exp_dict in mapping.items():
        if exp_dict:
            exp_dict.setdefault("early_stopping_patience", exp_dict.get("patience", 5))
            exp_dict.setdefault("reduce_lr_patience", max(1, exp_dict.get("patience", 5) // 2))
            exps[exp_key] = exp_dict
    config["experiments"] = exps

    return config
