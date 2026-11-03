"""
Model evaluation, comprehensive metric calculation, confusion matrix generation,
and experiment comparison utilities.
"""

from pathlib import Path
import os
import json
from typing import Dict, Any, List, Optional
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import tensorflow as tf


def evaluate_test_set(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    test_df: pd.DataFrame,
    idx_to_class: Any,
    exp_name: str = "vgg16_exp",
    output_base_dir: str = "outputs/vgg16",
    training_metadata: Optional[Dict[str, Any]] = None,
    save_plots: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates model performance on the independent test split.

    Calculates:
    - Overall Accuracy
    - Macro & Weighted Precision
    - Macro & Weighted Recall
    - Macro & Weighted F1-Score
    - Per-class precision, recall, f1-score, and support
    - 38x38 Confusion Matrix (persisted as numeric CSV and visual heatmap PNG)

    Parameters:
        model: Trained Keras model.
        test_dataset: tf.data.Dataset for test split.
        test_df: Test dataframe with ground-truth 'label' and 'class'.
        idx_to_class: Mapping from integer label to class name.
        exp_name: Identifier for experiment.
        output_base_dir: Root directory for outputs.
        training_metadata: Optional dict containing training duration, params, etc.
        save_plots: Whether to generate and save heatmap plot.

    Returns:
        dict: Full evaluation summary metrics.
    """
    if "experiment_name" in kwargs:
        exp_name = kwargs["experiment_name"]
    if "class_names" in kwargs and idx_to_class is None:
        idx_to_class = kwargs["class_names"]
    if isinstance(idx_to_class, (list, tuple)):
        idx_to_class = {i: name for i, name in enumerate(idx_to_class)}

    reports_dir = kwargs.get("reports_dir", os.path.join(output_base_dir, "reports"))
    metrics_dir = kwargs.get("metrics_dir", os.path.join(output_base_dir, "metrics"))
    plots_dir = kwargs.get("plot_dir", kwargs.get("plots_dir", os.path.join(output_base_dir, "plots")))

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n[Evaluating {exp_name}] Running model predictions on test dataset...")
    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Reconstruct true labels aligned with batch order
    if test_df is not None and "label" in test_df.columns:
        y_true = test_df["label"].values[:len(y_pred)].astype(int)
    else:
        y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)

    # Ensure class labels are ordered from 0 to NUM_CLASSES - 1
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Evaluate loss instantaneously without redundant passes over dataset
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(scce(y_true, y_pred_probs).numpy())

    print("\n" + "=" * 65)
    print(f"TEST EVALUATION RESULTS [{exp_name}]:")
    print(f"  Accuracy          : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Macro Precision   : {p_macro:.4f}")
    print(f"  Macro Recall      : {r_macro:.4f}")
    print(f"  Macro F1-Score    : {f1_macro:.4f}")
    print(f"  Weighted F1-Score : {f1_weighted:.4f}")
    print(f"  Test Loss         : {test_loss:.4f}")
    print("=" * 65)

    # Per-class Classification Report
    clf_report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    clf_report_df = pd.DataFrame(clf_report_dict).transpose()
    clf_report_csv = os.path.join(reports_dir, f"classification_report_{exp_name}.csv")
    clf_report_df.to_csv(clf_report_csv, index=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = os.path.join(metrics_dir, f"confusion_matrix_{exp_name}.csv")
    cm_df.to_csv(cm_csv, index=True)

    # Plot Confusion Matrix
    if save_plots:
        plt.figure(figsize=(22, 18))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False
        )
        plt.title(f"Confusion Matrix (38 Classes) - {exp_name}\nAccuracy: {acc:.2%}", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Predicted Label", fontsize=13, fontweight="bold")
        plt.ylabel("True Label", fontsize=13, fontweight="bold")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        cm_plot_path = os.path.join(plots_dir, f"confusion_matrix_{exp_name}.png")
        plt.savefig(cm_plot_path, dpi=200)
        plt.close()

    # Compile overall summary
    metrics_summary = {
        "experiment": exp_name,
        "test_accuracy": acc,
        "accuracy": acc,
        "test_loss": test_loss,
        "loss": test_loss,
        "precision_macro": p_macro,
        "macro_precision": p_macro,
        "precision_weighted": p_weighted,
        "weighted_precision": p_weighted,
        "recall_macro": r_macro,
        "macro_recall": r_macro,
        "recall_weighted": r_weighted,
        "weighted_recall": r_weighted,
        "f1_macro": f1_macro,
        "macro_f1": f1_macro,
        "f1_weighted": f1_weighted,
        "weighted_f1": f1_weighted,
        "total_test_samples": int(len(y_true)),
        "classification_report_csv": clf_report_csv,
        "confusion_matrix_csv": cm_csv,
    }

    if training_metadata:
        metrics_summary["training_metadata"] = training_metadata

    metrics_json = os.path.join(metrics_dir, f"metrics_{exp_name}.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def plot_training_curves(
    history_source: Any,
    exp_name: Optional[str] = "vgg16_experiment",
    output_plot_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plots training and validation loss and accuracy across epochs.

    Parameters:
        history_source: Either tf.keras.callbacks.History object or path to CSV history file.
        exp_name (str): Experiment name label.
        output_plot_path (str): Destination PNG path.
    """
    if output_plot_path is None and output_path is not None:
        output_plot_path = output_path

    # Handle case where 2nd argument is the output path
    if output_plot_path is None and isinstance(exp_name, str):
        if exp_name.endswith(".png") or "/" in exp_name or "\\" in exp_name:
            output_plot_path = exp_name
            exp_name = Path(output_plot_path).stem

    if output_plot_path is None:
        output_plot_path = "outputs/vgg16/plots/training_curves.png"
    if isinstance(history_source, str):
        df = pd.read_csv(history_source)
        loss = df["loss"].values
        val_loss = df["val_loss"].values
        acc = df["accuracy"].values
        val_acc = df["val_accuracy"].values
        epochs_range = range(1, len(loss) + 1)
    else:
        h = history_source.history
        loss = h["loss"]
        val_loss = h["val_loss"]
        acc = h["accuracy"]
        val_acc = h["val_accuracy"]
        epochs_range = range(1, len(loss) + 1)

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val_loss = float(np.min(val_loss))
    best_val_acc = float(val_acc[best_epoch - 1])

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.figure(figsize=(15, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, "o-", color="#1f77b4", label="Train Loss", linewidth=2, markersize=4)
    plt.plot(epochs_range, val_loss, "s--", color="#d62728", label="Val Loss", linewidth=2, markersize=4)
    plt.axvline(x=best_epoch, color="#2ca02c", linestyle=":", linewidth=2, label=f"Best Val Epoch ({best_epoch})")
    plt.title(f"Loss Curve - {exp_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, "o-", color="#1f77b4", label="Train Accuracy", linewidth=2, markersize=4)
    plt.plot(epochs_range, val_acc, "s--", color="#d62728", label="Val Accuracy", linewidth=2, markersize=4)
    plt.axvline(x=best_epoch, color="#2ca02c", linestyle=":", linewidth=2, label=f"Best Val Epoch ({best_epoch})")
    plt.title(f"Accuracy Curve - {exp_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()


def generate_experiment_comparison_table(
    experiments_list: List[Dict[str, Any]],
    output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Constructs a comparative summary table across all conducted experiments.

    Parameters:
        experiments_list: List of metric dictionaries from evaluate_test_set.
        output_csv_path: Optional file path to export comparison CSV.

    Returns:
        pd.DataFrame: Comparative table.
    """
    rows = []
    for exp in experiments_list:
        meta = exp.get("training_metadata", {})
        timing = meta.get("timing", {})
        params = meta.get("parameters", {})

        row = {
            "Experiment": exp.get("experiment", "N/A"),
            "Test Accuracy (%)": round(exp.get("test_accuracy", 0.0) * 100, 2),
            "Test Loss": round(exp.get("test_loss", 0.0), 4),
            "Macro Precision": round(exp.get("precision_macro", 0.0), 4),
            "Macro Recall": round(exp.get("recall_macro", 0.0), 4),
            "Macro F1": round(exp.get("f1_macro", 0.0), 4),
            "Weighted F1": round(exp.get("f1_weighted", 0.0), 4),
            "Trainable Params": params.get("trainable_params", "N/A"),
            "Total Params": params.get("total_params", "N/A"),
            "Total Epochs": timing.get("total_epochs", "N/A"),
            "Training Time (s)": timing.get("total_duration_seconds", "N/A"),
        }
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_comp.to_csv(output_csv_path, index=False)

    return df_comp
