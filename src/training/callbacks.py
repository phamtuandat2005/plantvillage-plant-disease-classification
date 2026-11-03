"""
Training callbacks configuration and execution timing tracking.
"""

import os
import time
from datetime import datetime
from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    Callback,
)


class TrainingTimerCallback(Callback):
    """
    Callback that tracks per-epoch training durations, start time, end time,
    and total training runtime.
    """

    def __init__(self):
        super().__init__()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_iso: Optional[str] = None
        self.end_iso: Optional[str] = None
        self.epoch_times: List[float] = []
        self._epoch_start: Optional[float] = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.start_iso = datetime.now().isoformat()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - (self._epoch_start or time.time())
        self.epoch_times.append(duration)

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.end_iso = datetime.now().isoformat()

    @property
    def total_duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        if self.start_time:
            return round(time.time() - self.start_time, 2)
        return 0.0

    @property
    def total_duration(self) -> float:
        """Alias for total_duration_seconds."""
        return self.total_duration_seconds

    def get_summary(self) -> dict:
        return {
            "start_time": self.start_iso,
            "end_time": self.end_iso,
            "total_duration_seconds": self.total_duration_seconds,
            "total_epochs": len(self.epoch_times),
            "mean_epoch_seconds": round(float(sum(self.epoch_times) / max(len(self.epoch_times), 1)), 2),
        }


def create_experiment_callbacks(
    checkpoint_path: str,
    history_csv_path: Optional[str] = None,
    monitor: str = "val_loss",
    mode: str = "min",
    patience_es: int = 5,
    patience_lr: int = 2,
    lr_factor: float = 0.2,
    min_lr: float = 1e-7,
    save_weights_only: bool = False,
    log_csv_path: Optional[str] = None,
    **kwargs
) -> Tuple[List[Callback], TrainingTimerCallback]:
    """
    Assembles production-grade callbacks for model training:
    - ModelCheckpoint: Saves best checkpoint based on validation metric.
    - EarlyStopping: Halts training if no improvement after patience epochs, restores best weights.
    - ReduceLROnPlateau: Decays learning rate when plateauing.
    - CSVLogger: Logs epoch metrics to CSV.
    - TrainingTimerCallback: Captures timing data.
    """
    if history_csv_path is None and log_csv_path is not None:
        history_csv_path = log_csv_path
    if history_csv_path is None:
        history_csv_path = "outputs/vgg16/histories/training_history.csv"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=save_weights_only,
        verbose=1,
    )

    early_stopping_cb = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience_es,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=lr_factor,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1,
    )

    csv_logger_cb = CSVLogger(
        filename=history_csv_path,
        separator=",",
        append=False,
    )

    timer_cb = TrainingTimerCallback()

    callbacks = [
        checkpoint_cb,
        early_stopping_cb,
        reduce_lr_cb,
        csv_logger_cb,
        timer_cb,
    ]

    return callbacks, timer_cb
