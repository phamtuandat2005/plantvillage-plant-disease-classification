"""
Model training execution engine for Transfer Learning and Fine-tuning experiments.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models

from src.models.vgg16_model import (
    build_vgg16_classifier,
    set_backbone_trainable,
    count_parameters,
    verify_forward_pass,
)
from src.training.callbacks import create_experiment_callbacks, TrainingTimerCallback
from src.utils.reproducibility import get_environment_info


def compile_and_train_vgg16(
    model: models.Model,
    train_dataset: Optional[tf.data.Dataset] = None,
    val_dataset: Optional[tf.data.Dataset] = None,
    checkpoint_path: Optional[str] = None,
    history_csv_path: Optional[str] = None,
    learning_rate: float = 1e-3,
    epochs: int = 20,
    patience_es: int = 5,
    patience_lr: int = 2,
    class_weights: Optional[Dict[int, float]] = None,
    monitor: str = "val_loss",
    mode: str = "min",
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    train_ds: Optional[tf.data.Dataset] = None,
    val_ds: Optional[tf.data.Dataset] = None,
    callbacks: Optional[List[Any]] = None,
    **kwargs
) -> Tuple[models.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    """
    Compiles and executes model training with robust checkpointing, early stopping,
    and runtime performance tracking.

    Parameters:
        model (models.Model): Model to train.
        train_dataset (tf.data.Dataset, optional): Training dataset.
        val_dataset (tf.data.Dataset, optional): Validation dataset.
        checkpoint_path (str, optional): Checkpoint target path (.keras).
        history_csv_path (str, optional): Training log CSV path.
        learning_rate (float): Initial optimizer learning rate.
        epochs (int): Maximum training epochs.
        patience_es (int): Patience for EarlyStopping.
        patience_lr (int): Patience for ReduceLROnPlateau.
        class_weights (dict, optional): Class imbalance weights.
        monitor (str): Metric to monitor.
        mode (str): 'min' or 'max'.
        steps_per_epoch (int, optional): Steps per epoch.
        validation_steps (int, optional): Validation steps.
        train_ds (tf.data.Dataset, optional): Alias for train_dataset.
        val_ds (tf.data.Dataset, optional): Alias for val_dataset.
        callbacks (list, optional): Pre-constructed list of callbacks.

    Returns:
        tuple: (trained_model, history, metadata)
    """
    if train_dataset is None and train_ds is not None:
        train_dataset = train_ds
    if val_dataset is None and val_ds is not None:
        val_dataset = val_ds

    if train_dataset is None:
        raise ValueError("train_dataset (or train_ds) must be provided.")

    checkpoint_path = checkpoint_path or "outputs/vgg16/checkpoints/best_model.keras"
    history_csv_path = history_csv_path or "outputs/vgg16/histories/training_history.csv"

    # Setup callbacks if not passed
    timer_cb = None
    if callbacks is None:
        callbacks, timer_cb = create_experiment_callbacks(
            checkpoint_path=checkpoint_path,
            history_csv_path=history_csv_path,
            monitor=monitor,
            mode=mode,
            patience_es=patience_es,
            patience_lr=patience_lr,
        )
    else:
        # Search for timer callback among passed callbacks
        for cb in callbacks:
            if isinstance(cb, TrainingTimerCallback):
                timer_cb = cb
                break
        if timer_cb is None:
            timer_cb = TrainingTimerCallback()
            callbacks.append(timer_cb)

    # Compile with Adam and SparseCategoricalCrossentropy
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )

    print(f"\n[Training] Starting training: LR={learning_rate:.1e}, Epochs={epochs}, Patience(ES)={patience_es}")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    metadata = {
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "epochs_trained": len(history.epoch),
        "parameters": count_parameters(model),
        "timing": timer_cb.get_summary() if timer_cb else {},
        "environment": get_environment_info(),
        "checkpoint_path": checkpoint_path,
        "history_csv_path": history_csv_path,
    }

    # Best epoch stats
    val_loss_hist = history.history.get("val_loss", [])
    if val_loss_hist:
        best_epoch_idx = int(tf.argmin(val_loss_hist).numpy())
        metadata["best_epoch"] = best_epoch_idx + 1
        metadata["best_val_loss"] = float(val_loss_hist[best_epoch_idx])
        if "val_accuracy" in history.history:
            metadata["best_val_accuracy"] = float(history.history["val_accuracy"][best_epoch_idx])

    # Save run metadata alongside history
    meta_path = history_csv_path.replace(".csv", "_metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: could not write metadata to {meta_path}: {e}")

    return model, history, metadata
