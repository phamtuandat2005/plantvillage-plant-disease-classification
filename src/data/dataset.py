"""
Dataset loading, stratified split verification, class imbalance handling, and tf.data pipeline creation.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from src.utils.config import BASE_DIR, SEED, BATCH_SIZE, IMG_SIZE
from src.data.preprocessing import (
    load_raw_image,
    preprocess_vgg16_image,
    load_and_preprocess_image,
    resolve_image_path,
)
from src.data.augmentation import build_augmentation_layer


def load_dataset_splits(
    csv_dir: Optional[str] = None,
    reports_dir: Optional[str] = None,
    color_dir: Optional[str] = None,
    verify_leakage: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:
    """
    Loads train, validation, and test splits from CSV files.
    Verifies split integrity, creates class mappings, and persists them to JSON.

    Parameters:
        csv_dir (str, optional): Directory containing train.csv, val.csv, test.csv.
        reports_dir (str, optional): Directory to save class_indices.json.
        color_dir (str, optional): Directory of raw images (kept for compatibility).
        verify_leakage (bool): Whether to assert zero sample leakage between splits.

    Returns:
        tuple: (train_df, val_df, test_df, idx_to_class, class_to_idx)
    """
    if csv_dir is None:
        csv_dir = os.path.join(str(BASE_DIR), "plantvillage_dataset", "csv")

    train_path = os.path.join(csv_dir, "train.csv")
    val_path = os.path.join(csv_dir, "val.csv")
    test_path = os.path.join(csv_dir, "test.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split file missing: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Resolve paths if machine root has changed
    train_df["path"] = train_df["path"].apply(lambda p: resolve_image_path(p, BASE_DIR))
    val_df["path"] = val_df["path"].apply(lambda p: resolve_image_path(p, BASE_DIR))
    test_df["path"] = test_df["path"].apply(lambda p: resolve_image_path(p, BASE_DIR))

    # Construct and verify class mapping
    class_mapping_df = train_df[["label", "class"]].drop_duplicates().sort_values("label")
    idx_to_class = dict(zip(class_mapping_df["label"].astype(int), class_mapping_df["class"]))
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    # Verify val and test adhere to the same mapping
    val_map = dict(zip(val_df["label"].astype(int), val_df["class"]))
    test_map = dict(zip(test_df["label"].astype(int), test_df["class"]))
    assert idx_to_class == val_map, "Validation class mapping does not match train!"
    assert idx_to_class == test_map, "Test class mapping does not match train!"

    # Verify disjoint splits (zero leakage)
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])
    assert len(train_paths & val_paths) == 0, "Data leakage detected between train and val!"
    assert len(train_paths & test_paths) == 0, "Data leakage detected between train and test!"
    assert len(val_paths & test_paths) == 0, "Data leakage detected between val and test!"

    # Save mapping to outputs if reports_dir is provided
    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
        json_path = os.path.join(reports_dir, "class_indices.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2, ensure_ascii=False)

    return train_df, val_df, test_df, idx_to_class, class_to_idx


def compute_class_imbalance_weights(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    reports_dir: Optional[str] = None,
    num_classes: Optional[int] = None,
    **kwargs
) -> Dict[int, float]:
    """
    Computes balanced class weights strictly using the training set distribution.
    Exports sample distribution table and class weights to CSV and JSON.

    Parameters:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame, optional): Validation dataframe.
        test_df (pd.DataFrame, optional): Test dataframe.
        reports_dir (str, optional): Target reports directory.
        num_classes (int, optional): Total expected classes.

    Returns:
        dict: Mapping of class index (int) to class weight (float).
    """
    train_labels = train_df["label"].values.astype(int)
    classes = np.unique(train_labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    # Ensure all classes up to num_classes are keyed
    if num_classes is not None:
        for idx in range(num_classes):
            class_weight_dict.setdefault(idx, 1.0)

    # Build summary distribution table
    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
        stats = []
        train_counts = train_df["label"].value_counts().to_dict()
        val_counts = val_df["label"].value_counts().to_dict() if val_df is not None else {}
        test_counts = test_df["label"].value_counts().to_dict() if test_df is not None else {}

        mapping_df = train_df[["label", "class"]].drop_duplicates().sort_values("label")
        for _, row in mapping_df.iterrows():
            lbl = int(row["label"])
            c_name = row["class"]
            n_tr = train_counts.get(lbl, 0)
            n_va = val_counts.get(lbl, 0)
            n_te = test_counts.get(lbl, 0)
            n_total = n_tr + n_va + n_te
            stats.append({
                "label": lbl,
                "class_name": c_name,
                "train_count": n_tr,
                "val_count": n_va,
                "test_count": n_te,
                "total_count": n_total,
                "train_pct": round(n_tr / len(train_df) * 100, 3),
                "class_weight": round(class_weight_dict.get(lbl, 1.0), 4)
            })

        stats_df = pd.DataFrame(stats)
        stats_csv = os.path.join(reports_dir, "class_distribution_splits.csv")
        stats_df.to_csv(stats_csv, index=False)

        weights_json = os.path.join(reports_dir, "class_weights.json")
        with open(weights_json, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in class_weight_dict.items()}, f, indent=2)

    return class_weight_dict


def create_tf_dataset(
    df: pd.DataFrame,
    is_training: bool = False,
    use_augmentation: bool = False,
    batch_size: int = BATCH_SIZE,
    img_size: Tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
    aug_config: Optional[Dict[str, Any]] = None,
    cache: bool = False,
    augment: Optional[bool] = None,
    **kwargs
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset with batching, mapping, and prefetching.

    Pipeline details:
    - Train (with augmentation): load_raw -> augment -> vgg16.preprocess_input
    - Train (without augmentation): load_raw -> vgg16.preprocess_input
    - Val / Test: load_raw -> vgg16.preprocess_input (no shuffling, no augmentation)

    Parameters:
        df (pd.DataFrame): Dataset dataframe containing 'path' and 'label'.
        is_training (bool): If True, shuffles dataset.
        use_augmentation (bool): If True, maps data augmentation.
        batch_size (int): Batch size.
        img_size (tuple): Target image size.
        seed (int): Seed for shuffling.
        aug_config (dict, optional): Hyperparameters for data augmentation.
        cache (bool): Whether to cache in memory.
        augment (bool, optional): Alias for use_augmentation.

    Returns:
        tf.data.Dataset: High-performance data pipeline.
    """
    if augment is not None:
        use_augmentation = augment
    paths = df["path"].values
    labels = df["label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=seed,
            reshuffle_each_iteration=True
        )

    if is_training and use_augmentation:
        aug_layer = build_augmentation_layer(config=aug_config, seed=seed)

        def _map_train_aug(path, label):
            raw_img = load_raw_image(path, img_size=img_size)
            aug_img = aug_layer(raw_img, training=True)
            prep_img = preprocess_vgg16_image(aug_img)
            return prep_img, label

        dataset = dataset.map(_map_train_aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def _map_standard(path, label):
            return load_and_preprocess_image(path, label, img_size=img_size)

        dataset = dataset.map(_map_standard, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

