"""
Command Line Interface (CLI) for running PlantVillage VGG16 experiments:
- Experiment A: Feature Extraction (Backbone frozen, LR=1e-3, with Augmentation)
- Experiment B: Fine-tuning (Block 5 unfrozen, LR=1e-5, with Augmentation)
- Experiment C: Ablation Study (Backbone frozen, LR=1e-3, without Augmentation)
- Explainability: Grad-CAM on block5_conv3 for correct & misclassified samples
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import tensorflow as tf

from src.utils.reproducibility import set_seed, get_environment_info
from src.utils.config import load_config
from src.data.dataset import (
    load_dataset_splits,
    compute_class_imbalance_weights,
    create_tf_dataset,
)
from src.models.vgg16_model import (
    build_vgg16_classifier,
    set_backbone_trainable,
    count_parameters,
)
from src.training.callbacks import create_experiment_callbacks
from src.training.train import compile_and_train_vgg16
from src.evaluation.evaluate import (
    evaluate_test_set,
    plot_training_curves,
    generate_experiment_comparison_table,
)
from src.evaluation.gradcam import explain_predictions_with_gradcam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGG16 Transfer Learning Experiments on PlantVillage Dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vgg16_config.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["a", "b", "c", "all"],
        default="all",
        help="Experiment to run: a (Feature Extraction), b (Fine-tuning), c (No Aug), all",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast scientific benchmark mode (subset of samples)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=425,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs_a",
        type=int,
        default=None,
        help="Override epochs for Experiment A",
    )
    parser.add_argument(
        "--epochs_b",
        type=int,
        default=None,
        help="Override epochs for Experiment B",
    )
    parser.add_argument(
        "--epochs_c",
        type=int,
        default=None,
        help="Override epochs for Experiment C",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_info = set_seed(args.seed)
    env_info = get_environment_info()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]

    print("=" * 70)
    print("VGG16 TRANSFER LEARNING & FINE-TUNING PIPELINE")
    print(f"Environment: Python {env_info['python_version']} | TF {env_info['tensorflow_version']} | Keras {env_info['keras_version']}")
    print(f"Hardware: GPU Available={env_info['gpu_available']} (Devices={env_info['gpu_devices']})")
    print(f"Seed Configuration: {seed_info}")
    print(f"Target Experiment: {args.experiment}")
    print("=" * 70)

    # 1. Load data splits
    train_df, val_df, test_df, idx_to_class, class_to_idx = load_dataset_splits(
        csv_dir=paths["csv_dir"],
        color_dir=paths["color_dir"],
        verify_leakage=True,
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    print(f"Dataset splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}, Classes={num_classes}")

    # Compute class imbalance weights strictly from training set
    class_weights = compute_class_imbalance_weights(train_df, num_classes=num_classes)

    # Save metadata
    os.makedirs(paths["reports_dir"], exist_ok=True)
    os.makedirs(paths["checkpoints_dir"], exist_ok=True)
    os.makedirs(paths["histories_dir"], exist_ok=True)
    os.makedirs(paths["plots_dir"], exist_ok=True)
    os.makedirs(paths["metrics_dir"], exist_ok=True)
    os.makedirs(paths["gradcam_dir"], exist_ok=True)

    with open(os.path.join(paths["reports_dir"], "class_indices.json"), "w") as f:
        json.dump({cls: i for i, cls in enumerate(class_names)}, f, indent=2)

    with open(os.path.join(paths["reports_dir"], "class_weights.json"), "w") as f:
        json.dump({str(k): round(float(v), 4) for k, v in class_weights.items()}, f, indent=2)

    # Class distribution table
    dist_df = pd.DataFrame({
        "class_name": class_names,
        "train_samples": [int((train_df["label"] == i).sum()) for i in range(num_classes)],
        "val_samples": [int((val_df["label"] == i).sum()) for i in range(num_classes)],
        "test_samples": [int((test_df["label"] == i).sum()) for i in range(num_classes)],
        "class_weight": [round(float(class_weights[i]), 4) for i in range(num_classes)],
    })
    dist_df.to_csv(os.path.join(paths["reports_dir"], "class_distribution_splits.csv"), index=False)

    # Handle quick benchmark mode
    if args.quick:
        print("\n[INFO] Quick benchmark mode enabled: sampling representative stratified slice.")
        train_slice = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.head(max(1, int(1280 / num_classes)))
        ).reset_index(drop=True)
        val_slice = val_df.groupby("label", group_keys=False).apply(
            lambda x: x.head(max(1, int(320 / num_classes)))
        ).reset_index(drop=True)
        test_slice = test_df.groupby("label", group_keys=False).apply(
            lambda x: x.head(max(1, int(640 / num_classes)))
        ).reset_index(drop=True)
    else:
        train_slice, val_slice, test_slice = train_df, val_df, test_df

    experiments_to_run = ["a", "b", "c"] if args.experiment == "all" else [args.experiment]
    results_summary = []
    trained_models = {}

    # =========================================================================
    # EXPERIMENT A: Feature Extraction
    # =========================================================================
    if "a" in experiments_to_run:
        exp_name = "Experiment_A_Feature_Extraction"
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENT A: Feature Extraction (Frozen Backbone, LR=1e-3)")
        print("=" * 70)

        epochs_a = args.epochs_a or (2 if args.quick else cfg["experiments"]["experiment_a"]["epochs"])
        lr_a = cfg["experiments"]["experiment_a"]["learning_rate"]

        train_ds_a = create_tf_dataset(train_slice, batch_size=batch_size, is_training=True, augment=True)
        val_ds_a = create_tf_dataset(val_slice, batch_size=batch_size, is_training=False, augment=False)
        test_ds_a = create_tf_dataset(test_slice, batch_size=batch_size, is_training=False, augment=False)

        model_a, base_model_a = build_vgg16_classifier(num_classes=num_classes, dropout_rate=0.3)
        p_a = count_parameters(model_a)

        ckpt_a = os.path.join(paths["checkpoints_dir"], "vgg16_feature_extraction_best.keras")
        hist_csv_a = os.path.join(paths["histories_dir"], "vgg16_feature_extraction_history.csv")

        callbacks_a, timer_a = create_experiment_callbacks(
            checkpoint_path=ckpt_a,
            log_csv_path=hist_csv_a,
            patience_es=cfg["experiments"]["experiment_a"]["early_stopping_patience"],
            patience_lr=cfg["experiments"]["experiment_a"]["reduce_lr_patience"],
        )

        model_a, history_a, metadata_a = compile_and_train_vgg16(
            model=model_a,
            train_ds=train_ds_a,
            val_ds=val_ds_a,
            epochs=epochs_a,
            learning_rate=lr_a,
            class_weights=class_weights,
            callbacks=callbacks_a,
        )

        plot_training_curves(history_a, output_path=os.path.join(paths["plots_dir"], "training_curves_exp_a.png"))

        metrics_a = evaluate_test_set(
            model=model_a,
            test_dataset=test_ds_a,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            exp_name=exp_name,
            output_base_dir=paths["output_root"],
        )

        results_summary.append({
            "Experiment": exp_name,
            "Test Accuracy (%)": round(metrics_a["accuracy"] * 100, 2),
            "Test Loss": round(metrics_a["loss"], 4),
            "Macro Precision": round(metrics_a["macro_precision"], 4),
            "Macro Recall": round(metrics_a["macro_recall"], 4),
            "Macro F1": round(metrics_a["macro_f1"], 4),
            "Weighted F1": round(metrics_a["weighted_f1"], 4),
            "Trainable Params": p_a["trainable_params"],
            "Total Params": p_a["total_params"],
            "Total Epochs": len(history_a.history["loss"]),
            "Training Time (s)": round(timer_a.total_duration, 2),
        })
        trained_models["a"] = model_a

    # =========================================================================
    # EXPERIMENT B: Fine-tuning
    # =========================================================================
    if "b" in experiments_to_run:
        exp_name = "Experiment_B_Fine_Tuning"
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENT B: Fine-Tuning (Unfreeze Block 5, LR=1e-5)")
        print("=" * 70)

        epochs_b = args.epochs_b or (1 if args.quick else cfg["experiments"]["experiment_b"]["epochs"])
        lr_b = cfg["experiments"]["experiment_b"]["learning_rate"]

        train_ds_b = create_tf_dataset(train_slice, batch_size=batch_size, is_training=True, augment=True)
        val_ds_b = create_tf_dataset(val_slice, batch_size=batch_size, is_training=False, augment=False)
        test_ds_b = create_tf_dataset(test_slice, batch_size=batch_size, is_training=False, augment=False)

        # Inherit best weights from Exp A if available
        ckpt_a = os.path.join(paths["checkpoints_dir"], "vgg16_feature_extraction_best.keras")
        if "a" in trained_models:
            model_b = trained_models["a"]
        elif os.path.exists(ckpt_a):
            print(f"Restoring best checkpoint from Experiment A: {ckpt_a}")
            model_b = tf.keras.models.load_model(ckpt_a)
        else:
            model_b = build_vgg16_classifier(num_classes=num_classes, dropout_rate=0.3)

        # Unfreeze Block 5
        set_backbone_trainable(model_b, unfreeze_from_block=5)
        p_b = count_parameters(model_b)
        print(f"Block 5 Unfrozen: Trainable params = {p_b['trainable_params']:,} / {p_b['total_params']:,}")

        ckpt_b = os.path.join(paths["checkpoints_dir"], "vgg16_finetuned_best.keras")
        hist_csv_b = os.path.join(paths["histories_dir"], "vgg16_finetuned_history.csv")

        callbacks_b, timer_b = create_experiment_callbacks(
            checkpoint_path=ckpt_b,
            log_csv_path=hist_csv_b,
            patience_es=cfg["experiments"]["experiment_b"]["early_stopping_patience"],
            patience_lr=cfg["experiments"]["experiment_b"]["reduce_lr_patience"],
        )

        model_b, history_b, metadata_b = compile_and_train_vgg16(
            model=model_b,
            train_ds=train_ds_b,
            val_ds=val_ds_b,
            epochs=epochs_b,
            learning_rate=lr_b,
            class_weights=class_weights,
            callbacks=callbacks_b,
        )

        plot_training_curves(history_b, output_path=os.path.join(paths["plots_dir"], "training_curves_exp_b.png"))

        metrics_b = evaluate_test_set(
            model=model_b,
            test_dataset=test_ds_b,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            exp_name=exp_name,
            output_base_dir=paths["output_root"],
        )

        results_summary.append({
            "Experiment": exp_name,
            "Test Accuracy (%)": round(metrics_b["accuracy"] * 100, 2),
            "Test Loss": round(metrics_b["loss"], 4),
            "Macro Precision": round(metrics_b["macro_precision"], 4),
            "Macro Recall": round(metrics_b["macro_recall"], 4),
            "Macro F1": round(metrics_b["macro_f1"], 4),
            "Weighted F1": round(metrics_b["weighted_f1"], 4),
            "Trainable Params": p_b["trainable_params"],
            "Total Params": p_b["total_params"],
            "Total Epochs": len(history_b.history["loss"]),
            "Training Time (s)": round(timer_b.total_duration, 2),
        })
        trained_models["b"] = model_b

    # =========================================================================
    # EXPERIMENT C: Ablation Study (No Augmentation)
    # =========================================================================
    if "c" in experiments_to_run:
        exp_name = "Experiment_C_No_Augmentation"
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENT C: Augmentation Ablation (No Augmentation, LR=1e-3)")
        print("=" * 70)

        epochs_c = args.epochs_c or (2 if args.quick else cfg["experiments"]["experiment_c"]["epochs"])
        lr_c = cfg["experiments"]["experiment_c"]["learning_rate"]

        train_ds_c = create_tf_dataset(train_slice, batch_size=batch_size, is_training=True, augment=False)
        val_ds_c = create_tf_dataset(val_slice, batch_size=batch_size, is_training=False, augment=False)
        test_ds_c = create_tf_dataset(test_slice, batch_size=batch_size, is_training=False, augment=False)

        model_c, base_model_c = build_vgg16_classifier(num_classes=num_classes, dropout_rate=0.3)
        p_c = count_parameters(model_c)

        ckpt_c = os.path.join(paths["checkpoints_dir"], "vgg16_no_aug_best.keras")
        hist_csv_c = os.path.join(paths["histories_dir"], "vgg16_no_aug_history.csv")

        callbacks_c, timer_c = create_experiment_callbacks(
            checkpoint_path=ckpt_c,
            log_csv_path=hist_csv_c,
            patience_es=cfg["experiments"]["experiment_c"]["early_stopping_patience"],
            patience_lr=cfg["experiments"]["experiment_c"]["reduce_lr_patience"],
        )

        model_c, history_c, metadata_c = compile_and_train_vgg16(
            model=model_c,
            train_ds=train_ds_c,
            val_ds=val_ds_c,
            epochs=epochs_c,
            learning_rate=lr_c,
            class_weights=class_weights,
            callbacks=callbacks_c,
        )

        plot_training_curves(history_c, output_path=os.path.join(paths["plots_dir"], "training_curves_exp_c.png"))

        metrics_c = evaluate_test_set(
            model=model_c,
            test_dataset=test_ds_c,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            exp_name=exp_name,
            output_base_dir=paths["output_root"],
        )

        results_summary.append({
            "Experiment": exp_name,
            "Test Accuracy (%)": round(metrics_c["accuracy"] * 100, 2),
            "Test Loss": round(metrics_c["loss"], 4),
            "Macro Precision": round(metrics_c["macro_precision"], 4),
            "Macro Recall": round(metrics_c["macro_recall"], 4),
            "Macro F1": round(metrics_c["macro_f1"], 4),
            "Weighted F1": round(metrics_c["weighted_f1"], 4),
            "Trainable Params": p_c["trainable_params"],
            "Total Params": p_c["total_params"],
            "Total Epochs": len(history_c.history["loss"]),
            "Training Time (s)": round(timer_c.total_duration, 2),
        })
        trained_models["c"] = model_c

    # Save comparison table
    if results_summary:
        comp_df = pd.DataFrame(results_summary)
        comp_path = os.path.join(paths["reports_dir"], "experiments_comparison.csv")
        comp_df.to_csv(comp_path, index=False)
        print("\n" + "=" * 70)
        print("EXPERIMENTS COMPARISON TABLE:")
        print("=" * 70)
        print(comp_df.to_string(index=False))

    # =========================================================================
    # GRAD-CAM EXPLAINABILITY
    # =========================================================================
    best_model = trained_models.get("b") or trained_models.get("a") or trained_models.get("c")
    if best_model is not None:
        print("\n" + "=" * 70)
        print("RUNNING GRAD-CAM INTERPRETABILITY ANALYSIS")
        print("=" * 70)
        idx_to_class = {i: name for i, name in enumerate(class_names)}
        explain_predictions_with_gradcam(
            model=best_model,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            output_dir=paths["gradcam_dir"],
            num_correct=4,
            num_misclassified=4,
            last_conv_layer_name="block5_conv3",
            random_seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("ALL REQUESTED PIPELINE TASKS COMPLETED!")
    print(f"Artifacts and reports saved to: {paths['output_root']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

