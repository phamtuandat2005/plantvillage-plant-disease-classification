"""
Custom manual EarlyStopping implementation (Alternative to tf.keras.callbacks.EarlyStopping).
"""

from tensorflow.keras.callbacks import Callback


class ManualEarlyStopping:
    """
    Manual early stopping tracker for epoch-level evaluation loops.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"   [EarlyStopping] Validation loss did not improve: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

