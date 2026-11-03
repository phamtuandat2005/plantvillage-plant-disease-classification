"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for VGG16.
Visualizes discriminative regions activated for both correct and misclassified predictions.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models

from src.data.preprocessing import (
    load_raw_image,
    preprocess_vgg16_image,
    resolve_image_path,
)


def compute_gradcam_heatmap(
    img_array: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: str = "block5_conv3",
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Computes normalized Grad-CAM 2D heatmap for a single image tensor.

    Parameters:
        img_array: Preprocessed image tensor of shape (1, 224, 224, 3).
        model: VGG16 classifier model.
        last_conv_layer_name: Name of target convolutional layer (default: block5_conv3).
        pred_index: Target class index to explain. If None, uses top predicted class.

    Returns:
        tuple: (heatmap_2d [H, W], predicted_class_idx, confidence_score)
    """
    # Detect base VGG16 submodel if nested
    base_model = None
    for layer in model.layers:
        if "vgg16" in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        conv_feature_model = models.Model(base_model.inputs, last_conv_layer.output)
        pool_layer = base_model.get_layer("block5_pool")

        # Remaining classifier layers
        gap_layer = None
        dense_layer = None
        for l in model.layers:
            if "global_avg" in l.name or isinstance(l, tf.keras.layers.GlobalAveragePooling2D):
                gap_layer = l
            if "prediction" in l.name or ("dense" in l.name and getattr(l, "units", None) is not None):
                dense_layer = l

        with tf.GradientTape() as tape:
            conv_outputs = conv_feature_model(img_array)
            tape.watch(conv_outputs)
            x = pool_layer(conv_outputs)
            if gap_layer is not None:
                x = gap_layer(x)
            else:
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            preds = dense_layer(x) if dense_layer is not None else model(img_array)

            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            confidence = float(preds[0, pred_index])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Flat model architecture
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = models.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            confidence = float(preds[0, pred_index])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients: channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive contribution to the target class
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), int(pred_index), confidence


def overlay_heatmap_on_image(
    original_rgb_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Superimposes a Grad-CAM heatmap onto the original RGB image.

    Parameters:
        original_rgb_img: Original image array (uint8, range [0, 255], RGB order).
        heatmap: 2D float array in range [0, 1].
        alpha: Overlay blending weight (0.0 to 1.0).
        colormap: OpenCV colormap enum.

    Returns:
        np.ndarray: Blended RGB image (uint8, [0, 255]).
    """
    h, w = original_rgb_img.shape[:2]
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_scaled = np.uint8(255 * heatmap_resized)

    # Colorize using colormap (OpenCV returns BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap_scaled, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending
    blended = np.float32(original_rgb_img) * (1.0 - alpha) + np.float32(heatmap_rgb) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def explain_predictions_with_gradcam(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    idx_to_class: Dict[int, str],
    output_dir: str = "outputs/vgg16/gradcam",
    num_correct: int = 4,
    num_misclassified: int = 4,
    last_conv_layer_name: str = "block5_conv3",
    random_seed: int = 425
) -> pd.DataFrame:
    """
    Systematically selects test samples (both correct and misclassified),
    computes Grad-CAM heatmaps, generates comparative 3-panel figures,
    and logs predictions to a summary CSV.

    Parameters:
        model: Trained VGG16 model.
        test_df: Test split dataframe.
        idx_to_class: Label to class name dictionary.
        output_dir: Destination directory.
        num_correct: Number of correct test samples to explain.
        num_misclassified: Number of misclassified test samples to explain.
        last_conv_layer_name: Conv layer for CAM.
        random_seed: Seed for reproducible sample selection.

    Returns:
        pd.DataFrame: Analysis summary table.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    print(f"\n[Grad-CAM] Explaining model predictions using layer '{last_conv_layer_name}'...")

    # Predict on test samples to identify correct and misclassified
    # Sample a batch of 150 test instances to find clean and erroneous predictions
    sample_pool_size = min(len(test_df), 200)
    indices = rng.choice(len(test_df), size=sample_pool_size, replace=False)
    pool_df = test_df.iloc[indices].copy()

    records = []
    correct_samples = []
    misclassified_samples = []

    for idx, row in pool_df.iterrows():
        p = resolve_image_path(row["path"])
        true_lbl = int(row["label"])
        raw_tensor = load_raw_image(p, img_size=(224, 224))
        prep_tensor = preprocess_vgg16_image(raw_tensor)
        inp = tf.expand_dims(prep_tensor, axis=0)

        pred_probs = model(inp, training=False).numpy()[0]
        pred_lbl = int(np.argmax(pred_probs))
        conf = float(pred_probs[pred_lbl])

        item = {
            "path": p,
            "true_label": true_lbl,
            "true_class": idx_to_class[true_lbl],
            "pred_label": pred_lbl,
            "pred_class": idx_to_class[pred_lbl],
            "confidence": conf,
            "raw_tensor": raw_tensor,
            "prep_tensor": prep_tensor,
            "is_correct": (true_lbl == pred_lbl)
        }

        if item["is_correct"] and len(correct_samples) < num_correct:
            correct_samples.append(item)
        elif not item["is_correct"] and len(misclassified_samples) < num_misclassified:
            misclassified_samples.append(item)

        if len(correct_samples) >= num_correct and len(misclassified_samples) >= num_misclassified:
            break

    selected_samples = correct_samples + misclassified_samples
    summary_rows = []

    for i, item in enumerate(selected_samples):
        inp = tf.expand_dims(item["prep_tensor"], axis=0)
        heatmap, pred_idx, conf = compute_gradcam_heatmap(
            inp,
            model,
            last_conv_layer_name=last_conv_layer_name,
            pred_index=item["pred_label"]
        )

        orig_rgb = item["raw_tensor"].numpy().astype(np.uint8)
        overlay = overlay_heatmap_on_image(orig_rgb, heatmap, alpha=0.45)

        # Plot 3-panel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        status_str = "CORRECT" if item["is_correct"] else "MISCLASSIFIED"
        color_status = "green" if item["is_correct"] else "red"

        # 1. Original
        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"Original Image\nTrue: {item['true_class']}", fontsize=10)
        axes[0].axis("off")

        # 2. Heatmap
        im_hm = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title(f"Grad-CAM Heatmap\nLayer: {last_conv_layer_name}", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im_hm, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Superimposed
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Superimposed [{status_str}]\nPred: {item['pred_class']} ({conf*100:.1f}%)",
            fontsize=10,
            color=color_status,
            fontweight="bold"
        )
        axes[2].axis("off")

        fig.suptitle(
            f"Sample #{i+1} [{status_str}] - Ground Truth: {item['true_class']}",
            fontsize=12,
            fontweight="bold",
            y=1.02
        )
        plt.tight_layout()

        fname = f"gradcam_sample_{i+1:02d}_{status_str.lower()}_{item['true_label']}_vs_{item['pred_label']}.png"
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        summary_rows.append({
            "sample_index": i + 1,
            "status": status_str,
            "true_label": item["true_label"],
            "true_class": item["true_class"],
            "pred_label": item["pred_label"],
            "pred_class": item["pred_class"],
            "confidence": round(conf, 4),
            "image_path": item["path"],
            "visualization_path": out_path,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "gradcam_analysis_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Grad-CAM] Completed {len(summary_rows)} sample explanations. Results saved to: {output_dir}")

    return summary_df

