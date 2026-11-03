"""
Pipeline Verification & Smoke Test Script for PlantVillage VGG16 Experiments.
Verifies end-to-end functionality across data loading, model instantiation,
training loop, evaluation metrics, and Grad-CAM interpretability.
"""

import sys
import os
from pathlib import Path

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.utils.reproducibility import set_seed, get_environment_info
from src.utils.config import load_config
from src.data.dataset import load_dataset_splits, compute_class_imbalance_weights, create_tf_dataset
from src.models.vgg16_model import build_vgg16_classifier, set_backbone_trainable, count_parameters, verify_forward_pass
from src.training.callbacks import create_experiment_callbacks
from src.training.train import compile_and_train_vgg16
from src.evaluation.evaluate import evaluate_test_set
from src.evaluation.gradcam import compute_gradcam_heatmap, overlay_heatmap_on_image


def run_smoke_test():
    print("=" * 70)
    print("VGG16 PIPELINE VERIFICATION & SMOKE TEST")
    print("=" * 70)

    # 1. Environment & Seeds
    env = get_environment_info()
    print(f"[1/7] Environment: Python {env['python_version']} | TF {env['tensorflow_version']} | Keras {env['keras_version']}")
    print(f"      GPU Available: {env['gpu_available']} (Devices: {env['gpu_devices']})")
    set_seed(425)

    # 2. Config & Splits
    cfg = load_config()
    dataset_cfg = cfg.get("dataset", cfg.get("data", {}))
    paths_cfg = cfg.get("paths", {})
    img_size = tuple(dataset_cfg.get("img_size", [224, 224]))
    batch_size = dataset_cfg.get("batch_size", 32)
    print(f"[2/7] Configuration loaded. Image size: {img_size}, Batch size: {batch_size}")

    color_dir = paths_cfg.get("data_dir", paths_cfg.get("color_dir", "plantvillage_dataset/color"))
    csv_dir = paths_cfg.get("csv_dir", "plantvillage_dataset/csv")
    train_df, val_df, test_df, idx_to_class, class_to_idx = load_dataset_splits(
        csv_dir=csv_dir,
        color_dir=color_dir,
        verify_leakage=True
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    print(f"      Dataset: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}, Classes={num_classes}")

    # 3. Class Weights & tf.data
    weights = compute_class_imbalance_weights(train_df, num_classes=num_classes)
    print(f"[3/7] Class weights computed (min={min(weights.values()):.3f}, max={max(weights.values()):.3f})")

    # Fast subset for smoke test
    quick_train_df = train_df.head(64)
    quick_val_df = val_df.head(32)
    quick_test_df = test_df.head(32)

    train_ds = create_tf_dataset(quick_train_df, batch_size=16, is_training=True, augment=True)
    val_ds = create_tf_dataset(quick_val_df, batch_size=16, is_training=False, augment=False)
    test_ds = create_tf_dataset(quick_test_df, batch_size=16, is_training=False, augment=False)

    for images, labels in train_ds.take(1):
        print(f"      Batch shape: Images={images.shape}, Labels={labels.shape}")
        break

    # 4. Model Architecture & Parameters
    print("[4/7] Instantiating VGG16 Transfer Learning model...")
    model, base_model = build_vgg16_classifier(num_classes=num_classes, dropout_rate=0.3)
    p_frozen = count_parameters(model)
    print(f"      Feature Extraction: Trainable={p_frozen['trainable_params']:,}, Total={p_frozen['total_params']:,}")
    verify_forward_pass(model, input_shape=(2, 224, 224, 3), num_classes=num_classes)

    set_backbone_trainable(model, unfreeze_from_block=5)
    p_unfrozen = count_parameters(model)
    print(f"      Fine-tuning (Block 5): Trainable={p_unfrozen['trainable_params']:,}, Total={p_unfrozen['total_params']:,}")

    # Reset to frozen for smoke training
    set_backbone_trainable(model, unfreeze_from_block=None)

    # 5. Training loop (1 step)
    print("[5/7] Running 1-epoch smoke training pass...")
    os.makedirs("outputs/vgg16/smoke_test", exist_ok=True)
    cbs, timer = create_experiment_callbacks(
        checkpoint_path="outputs/vgg16/smoke_test/smoke_model.keras",
        log_csv_path="outputs/vgg16/histories/smoke_test_history.csv",
        patience_es=2,
        patience_lr=1
    )
    trained_model, history, metadata = compile_and_train_vgg16(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=1,
        learning_rate=1e-3,
        class_weights=weights,
        callbacks=cbs
    )
    print(f"      Smoke step completed in {timer.total_duration:.2f}s")

    # 6. Evaluation metrics & Confusion matrix
    print("[6/7] Evaluating smoke test metrics...")
    eval_res = evaluate_test_set(
        model=trained_model,
        test_dataset=test_ds,
        test_df=quick_test_df,
        idx_to_class=idx_to_class,
        exp_name="smoke_test",
        output_base_dir="outputs/vgg16"
    )
    print(f"      Smoke Test Accuracy: {eval_res['test_accuracy']*100:.2f}%, Test Loss: {eval_res['test_loss']:.4f}")

    # 7. Grad-CAM Interpretability
    print("[7/7] Testing Grad-CAM heatmap generation...")
    test_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    heatmap, pred_idx, conf = compute_gradcam_heatmap(test_img, trained_model, last_conv_layer_name="block5_conv3")
    overlay = overlay_heatmap_on_image(np.full((224, 224, 3), 128, dtype=np.uint8), heatmap)
    print(f"      Grad-CAM verified. Heatmap shape: {heatmap.shape}, Overlay shape: {overlay.shape}")

    print("\n" + "=" * 70)
    print("ALL 7 PIPELINE STAGES VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()

