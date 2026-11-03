"""
PlantVillage VGG16 Deep Learning Studio & Training CLI (Gemini CLI Style).
Interactive terminal user interface (TUI) powered by Rich and argparse.

Supports:
- Interactive Wizard Mode: Guided prompts, presets, live telemetry, and menu selection.
- Headless / Script Mode: Full flags support for automated batch runs & pipelines.
- Paradigms: Two-Stage Transfer Learning, Feature Extraction, Fine-Tuning, Ablation Study.
- Post-training Evaluation & Grad-CAM explainability.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure UTF-8 output on Windows terminals & suppress TF verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Rich UI Components
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


def print_cli_banner():
    """Renders clean header panel without emojis."""
    banner_text = Text()
    banner_text.append("PLANTVILLAGE DEEP LEARNING STUDIO\n", style="bold green")
    banner_text.append("VGG16 Transfer Learning | Fine-Tuning | Grad-CAM Explainability\n", style="bold cyan")
    banner_text.append("Academic Research Pipeline | TensorFlow & Keras 3", style="dim white")

    if HAS_RICH:
        console.print(Panel(banner_text, border_style="bright_blue", expand=False, padding=(1, 3)))
    else:
        print("=" * 70)
        print("PLANTVILLAGE VGG16 DEEP LEARNING STUDIO")
        print("=" * 70)


def print_hardware_status(env_info: Dict[str, Any]):
    """Displays hardware and environment telemetry table."""
    if not HAS_RICH:
        print(f"Device: {'GPU' if env_info['gpu_available'] else 'CPU'} | Python: {env_info['python_version']} | TF: {env_info['tensorflow_version']}")
        return

    table = Table(title="System Telemetry & Compute Device", title_style="bold yellow", border_style="dim")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Specification", style="white")
    table.add_column("Status", justify="center")

    gpu_status = "[bold green]ACCELERATED[/bold green]" if env_info["gpu_available"] else "[yellow]CPU MODE[/yellow]"
    table.add_row("Compute Device", f"{'NVIDIA GPU' if env_info['gpu_available'] else 'Multi-core CPU'} ({len(env_info['gpu_devices'])} devices)", gpu_status)
    table.add_row("Python Runtime", env_info["python_version"], "[green]OK[/green]")
    table.add_row("Deep Learning", f"TensorFlow {env_info['tensorflow_version']} / Keras {env_info['keras_version']}", "[green]OK[/green]")
    table.add_row("Operating System", f"{env_info['os']} {env_info['os_release']}", "[dim]ACTIVE[/dim]")
    console.print(table)


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for headless / batch execution."""
    parser = argparse.ArgumentParser(
        prog="python train.py",
        description="PlantVillage VGG16 Deep Learning Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive Gemini CLI studio wizard.",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["feature_extraction", "fine_tuning", "two_stage"],
        default="two_stage",
        help="Training mode: 'two_stage' (recommended), 'feature_extraction', or 'fine_tuning'.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing .keras checkpoint to resume/fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum epochs for single-stage mode.",
    )
    parser.add_argument(
        "--epochs_stage1",
        type=int,
        default=15,
        help="Stage 1 (Feature Extraction) epochs in two_stage mode.",
    )
    parser.add_argument(
        "--epochs_stage2",
        type=int,
        default=15,
        help="Stage 2 (Fine-Tuning) epochs in two_stage mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr", "--learning_rate",
        type=float,
        default=None,
        help="Learning rate override.",
    )
    parser.add_argument(
        "--lr_stage1",
        type=float,
        default=1e-3,
        help="Stage 1 learning rate in two_stage mode.",
    )
    parser.add_argument(
        "--lr_stage2",
        type=float,
        default=1e-5,
        help="Stage 2 learning rate in two_stage mode.",
    )
    parser.add_argument(
        "--unfreeze_block",
        type=int,
        choices=[3, 4, 5],
        default=5,
        help="Convolutional block to unfreeze (5 unfreezes block5_conv1..3).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability before final dense layer.",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Disable data augmentation on training set (ablation mode).",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable balanced class weighting.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification mode: runs on a small stratified subset.",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        default=True,
        help="Evaluate model on independent test set after training.",
    )
    parser.add_argument(
        "--no_eval_test",
        action="store_false",
        dest="eval_test",
        help="Skip test evaluation.",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        default=True,
        help="Generate Grad-CAM heatmaps for test predictions.",
    )
    parser.add_argument(
        "--no_gradcam",
        action="store_false",
        dest="gradcam",
        help="Skip Grad-CAM explainability.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="vgg16_run",
        help="Prefix identifier for checkpoints, logs, and figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=425,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vgg16_config.json",
        help="Path to JSON configuration file.",
    )
    return parser


def run_interactive_wizard():
    """Interactive Gemini CLI Studio Wizard without emojis."""
    if sys.stdin.isatty():
        os.system("cls" if os.name == "nt" else "clear")
    print_cli_banner()

    # Telemetry
    from src.utils.reproducibility import get_environment_info
    env_info = get_environment_info()
    print_hardware_status(env_info)

    # Main Action Menu
    console.print("\n[bold cyan]Select an Action:[/bold cyan]")
    menu_table = Table(show_header=False, box=None, padding=(0, 2))
    menu_table.add_row("[bold green][1][/bold green]", "[bold white]Two-Stage Transfer Learning[/bold white]", "[dim](Stage 1: Feature Extraction -> Stage 2: Fine-Tuning Block 5)[/dim]")
    menu_table.add_row("[bold green][2][/bold green]", "[bold white]Feature Extraction Only[/bold white]", "[dim](Train custom classifier head, backbone frozen, LR=1e-3)[/dim]")
    menu_table.add_row("[bold green][3][/bold green]", "[bold white]Fine-Tuning Only[/bold white]", "[dim](Unfreeze convolutional Block 5 with lower LR=1e-5)[/dim]")
    menu_table.add_row("[bold green][4][/bold green]", "[bold white]Augmentation Ablation Study[/bold white]", "[dim](Train without augmentation to measure regularization impact)[/dim]")
    menu_table.add_row("[bold green][5][/bold green]", "[bold white]Quick Benchmark / Smoke Test[/bold white]", "[dim](Rapid 1-2 min verification on representative stratified sample)[/dim]")
    menu_table.add_row("[bold green][6][/bold green]", "[bold white]Evaluate Existing Checkpoint & Grad-CAM[/bold white]", "[dim](Inspect metrics & generate heatmaps for an existing model)[/dim]")
    menu_table.add_row("[bold red][7][/bold red]", "[bold white]Exit[/bold white]", "")
    console.print(menu_table)

    choice = Prompt.ask("\n[bold yellow]Your choice[/bold yellow]", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")

    if choice == "7":
        console.print("[dim]Exiting PlantVillage Studio. Goodbye![/dim]")
        sys.exit(0)

    if choice == "6":
        run_standalone_evaluation_wizard()
        return

    # Map choice to mode
    mode_map = {
        "1": "two_stage",
        "2": "feature_extraction",
        "3": "fine_tuning",
        "4": "feature_extraction",  # with no_aug
        "5": "two_stage",           # with quick
    }
    mode = mode_map[choice]
    quick_mode = (choice == "5")
    no_aug = (choice == "4")

    # Preset selection
    console.print("\n[bold cyan]Select Hyperparameter Configuration Preset:[/bold cyan]")
    preset_table = Table(show_header=False, box=None, padding=(0, 2))
    preset_table.add_row("[bold green][1][/bold green]", "[bold white]Fast Verification[/bold white]", "[dim]Epochs: 2, Batch: 32 (Ideal for quick pipeline test)[/dim]")
    preset_table.add_row("[bold green][2][/bold green]", "[bold white]Standard Academic[/bold white]", "[dim]Epochs: 15 per stage, Batch: 32, LR: 1e-3 / 1e-5 (Recommended)[/dim]")
    preset_table.add_row("[bold green][3][/bold green]", "[bold white]Full Convergence[/bold white]", "[dim]Epochs: 25 per stage, Batch: 32, Patience: 7 (Maximum accuracy)[/dim]")
    preset_table.add_row("[bold green][4][/bold green]", "[bold white]Custom Advanced[/bold white]", "[dim]Configure every parameter manually[/dim]")
    console.print(preset_table)

    preset_choice = Prompt.ask("[bold yellow]Preset[/bold yellow]", choices=["1", "2", "3", "4"], default="1" if quick_mode else "2")

    epochs = 20
    epochs_s1 = 15
    epochs_s2 = 15
    batch_size = 32
    lr_s1 = 1e-3
    lr_s2 = 1e-5
    run_name = f"vgg16_{mode}"

    if preset_choice == "1":
        quick_mode = True
        epochs = 2
        epochs_s1 = 2
        epochs_s2 = 1
        run_name = f"vgg16_{mode}_quick"
    elif preset_choice == "2":
        epochs = 15
        epochs_s1 = 15
        epochs_s2 = 15
        run_name = f"vgg16_{mode}_standard"
    elif preset_choice == "3":
        epochs = 25
        epochs_s1 = 25
        epochs_s2 = 25
        run_name = f"vgg16_{mode}_converged"
    elif preset_choice == "4":
        run_name = Prompt.ask("[cyan]Enter run identifier name[/cyan]", default=f"vgg16_{mode}")
        batch_size = IntPrompt.ask("[cyan]Batch size[/cyan]", default=32)
        if mode == "two_stage":
            epochs_s1 = IntPrompt.ask("[cyan]Stage 1 (Feature Extraction) epochs[/cyan]", default=15)
            epochs_s2 = IntPrompt.ask("[cyan]Stage 2 (Fine-Tuning) epochs[/cyan]", default=15)
            lr_s1 = FloatPrompt.ask("[cyan]Stage 1 Learning rate[/cyan]", default=1e-3)
            lr_s2 = FloatPrompt.ask("[cyan]Stage 2 Learning rate[/cyan]", default=1e-5)
        else:
            epochs = IntPrompt.ask("[cyan]Epochs[/cyan]", default=15)
            default_lr = 1e-3 if mode == "feature_extraction" else 1e-5
            lr_s1 = FloatPrompt.ask("[cyan]Learning rate[/cyan]", default=default_lr)
        no_aug = not Confirm.ask("[cyan]Enable Data Augmentation?[/cyan]", default=True)
        quick_mode = Confirm.ask("[cyan]Run on fast subset (Quick mode)?[/cyan]", default=False)

    eval_test = Confirm.ask("\n[bold cyan]Run test evaluation & 38x38 confusion matrix after training?[/bold cyan]", default=True)
    gradcam = Confirm.ask("[bold cyan]Generate Grad-CAM heatmaps for predictions?[/bold cyan]", default=True) if eval_test else False

    # Confirmation Table
    summary_table = Table(title="Experiment Execution Plan", title_style="bold green", border_style="bright_blue")
    summary_table.add_column("Parameter", style="cyan")
    summary_table.add_column("Value", style="bold white")

    summary_table.add_row("Training Paradigm", mode.upper())
    summary_table.add_row("Run Identifier", run_name)
    if mode == "two_stage":
        summary_table.add_row("Stage 1 (Feature Extraction)", f"{epochs_s1} epochs @ LR={lr_s1}")
        summary_table.add_row("Stage 2 (Fine-Tuning)", f"{epochs_s2} epochs @ LR={lr_s2} (Unfreeze Block 5)")
    else:
        summary_table.add_row("Epochs & Learning Rate", f"{epochs} epochs @ LR={lr_s1}")
    summary_table.add_row("Batch Size", str(batch_size))
    summary_table.add_row("Data Augmentation", "[red]Disabled (Ablation)[/red]" if no_aug else "[green]Enabled (Flip, Rotate, Zoom, Brightness)[/green]")
    summary_table.add_row("Class Weights", "[green]Enabled (Balanced from Train set)[/green]")
    summary_table.add_row("Quick Benchmark Mode", "[yellow]True (Representative subset)[/yellow]" if quick_mode else "[green]False (Full 54k images)[/green]")
    summary_table.add_row("Post-Training Eval", "[green]Enabled[/green]" if eval_test else "[red]Disabled[/red]")
    summary_table.add_row("Grad-CAM Explainability", "[green]Enabled[/green]" if gradcam else "[red]Disabled[/red]")

    console.print(summary_table)

    if not Confirm.ask("\n[bold green]Ready to launch training?[/bold green]", default=True):
        console.print("[dim]Operation aborted by user.[/dim]")
        return

    # Execute training
    execute_training_pipeline(
        mode=mode,
        run_name=run_name,
        epochs=epochs,
        epochs_s1=epochs_s1,
        epochs_s2=epochs_s2,
        batch_size=batch_size,
        lr_s1=lr_s1,
        lr_s2=lr_s2,
        unfreeze_block=5,
        no_augmentation=no_aug,
        no_class_weights=False,
        quick=quick_mode,
        eval_test=eval_test,
        gradcam=gradcam,
        seed=425,
        config_path="configs/vgg16_config.json"
    )


def run_standalone_evaluation_wizard():
    """Standalone checkpoint evaluator and Grad-CAM generator."""
    from src.utils.config import load_config
    cfg = load_config()
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    ckpts = list(ckpt_dir.glob("*.keras")) + list(Path("models").glob("*.keras"))
    if not ckpts:
        console.print("[bold red]No .keras checkpoints found in outputs/vgg16/checkpoints/ or models/![/bold red]")
        return

    console.print("\n[bold cyan]Available Checkpoints:[/bold cyan]")
    for i, c in enumerate(ckpts, start=1):
        size_mb = c.stat().st_size / (1024 * 1024)
        console.print(f"  [bold green][{i}][/bold green] {c.name} [dim]({size_mb:.1f} MB)[/dim]")

    idx = IntPrompt.ask("[bold yellow]Select checkpoint index[/bold yellow]", default=1)
    if idx < 1 or idx > len(ckpts):
        console.print("[red]Invalid selection![/red]")
        return
    chosen_ckpt = str(ckpts[idx - 1])

    console.print(f"\n[green]Selected Checkpoint:[/green] {chosen_ckpt}")
    quick = Confirm.ask("[cyan]Evaluate on quick test subset (faster)?[/cyan]", default=True)

    import tensorflow as tf
    from src.data.dataset import load_dataset_splits, create_tf_dataset
    from src.evaluation.evaluate import evaluate_test_set
    from src.evaluation.gradcam import explain_predictions_with_gradcam

    console.print("[dim]Loading test dataset...[/dim]")
    _, _, test_df, idx_to_class, _ = load_dataset_splits(verify_leakage=False)
    test_slice = test_df.groupby("label", group_keys=False).apply(lambda x: x.head(15)).reset_index(drop=True) if quick else test_df
    test_ds = create_tf_dataset(test_slice, batch_size=32, is_training=False, augment=False)

    console.print(f"[dim]Loading model from {chosen_ckpt}...[/dim]")
    model = tf.keras.models.load_model(chosen_ckpt)

    eval_res = evaluate_test_set(
        model=model,
        test_dataset=test_ds,
        test_df=test_slice,
        idx_to_class=idx_to_class,
        exp_name=Path(chosen_ckpt).stem,
        output_base_dir=cfg["paths"]["output_dir"]
    )

    console.print("\n" + "=" * 50)
    console.print(f"[bold green]Accuracy:[/bold green] {eval_res['test_accuracy']*100:.2f}% | [bold yellow]Loss:[/bold yellow] {eval_res['test_loss']:.4f}")
    console.print(f"[bold cyan]Macro F1:[/bold cyan] {eval_res['f1_macro']:.4f} | [bold magenta]Weighted F1:[/bold magenta] {eval_res['f1_weighted']:.4f}")
    console.print("=" * 50)

    if Confirm.ask("\n[cyan]Generate Grad-CAM heatmaps for this model?[/cyan]", default=True):
        out_gcam = os.path.join(cfg["paths"]["gradcam_dir"], Path(chosen_ckpt).stem)
        explain_predictions_with_gradcam(
            model=model,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            output_dir=out_gcam,
            num_correct=4,
            num_misclassified=4,
            last_conv_layer_name="block5_conv3"
        )
        console.print(f"[bold green][OK] Grad-CAM visualizations saved to:[/bold green] {out_gcam}")


def execute_training_pipeline(
    mode: str,
    run_name: str,
    epochs: int,
    epochs_s1: int,
    epochs_s2: int,
    batch_size: int,
    lr_s1: float,
    lr_s2: float,
    unfreeze_block: int = 5,
    no_augmentation: bool = False,
    no_class_weights: bool = False,
    quick: bool = False,
    eval_test: bool = True,
    gradcam: bool = True,
    seed: int = 425,
    config_path: str = "configs/vgg16_config.json",
    resume_path: Optional[str] = None,
):
    """Core execution engine for training and evaluation."""
    import tensorflow as tf
    from src.utils.reproducibility import set_seed, get_environment_info
    from src.utils.config import load_config
    from src.data.dataset import load_dataset_splits, compute_class_imbalance_weights, create_tf_dataset
    from src.models.vgg16_model import build_vgg16_classifier, set_backbone_trainable, count_parameters, verify_forward_pass
    from src.training.train import compile_and_train_vgg16
    from src.evaluation.evaluate import evaluate_test_set, plot_training_curves
    from src.evaluation.gradcam import explain_predictions_with_gradcam

    set_seed(seed)
    cfg = load_config(config_path)
    paths = cfg["paths"]

    # 1. Dataset splits
    if HAS_RICH:
        console.print("\n[bold cyan][1/5] Loading & Verifying Stratified Dataset Splits...[/bold cyan]")
    else:
        print("\n[1/5] Loading dataset splits...")

    train_df, val_df, test_df, idx_to_class, class_to_idx = load_dataset_splits(
        csv_dir=paths["csv_dir"],
        color_dir=paths["color_dir"],
        verify_leakage=True
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    class_weights = None
    if not no_class_weights:
        class_weights = compute_class_imbalance_weights(train_df, num_classes=num_classes, reports_dir=paths["reports_dir"])

    if quick:
        train_slice = train_df.groupby("label", group_keys=False).apply(lambda x: x.head(max(1, int(1280 / num_classes)))).reset_index(drop=True)
        val_slice = val_df.groupby("label", group_keys=False).apply(lambda x: x.head(max(1, int(320 / num_classes)))).reset_index(drop=True)
        test_slice = test_df.groupby("label", group_keys=False).apply(lambda x: x.head(max(1, int(640 / num_classes)))).reset_index(drop=True)
    else:
        train_slice, val_slice, test_slice = train_df, val_df, test_df

    # 2. tf.data Pipelines
    if HAS_RICH:
        console.print("[bold cyan][2/5] Building tf.data Streaming Pipelines...[/bold cyan]")
    train_ds = create_tf_dataset(train_slice, batch_size=batch_size, is_training=True, augment=(not no_augmentation))
    val_ds = create_tf_dataset(val_slice, batch_size=batch_size, is_training=False, augment=False)
    test_ds = create_tf_dataset(test_slice, batch_size=batch_size, is_training=False, augment=False)

    ckpt_dir = paths["checkpoints_dir"]
    hist_dir = paths["histories_dir"]
    plots_dir = paths["plots_dir"]

    # 3. Model setup
    if HAS_RICH:
        console.print("[bold cyan][3/5] Instantiating VGG16 Neural Network...[/bold cyan]")
    if resume_path and os.path.exists(resume_path):
        model = tf.keras.models.load_model(resume_path)
        base_model = None
        for l in model.layers:
            if "vgg16" in l.name.lower():
                base_model = l
                break
    else:
        model, base_model = build_vgg16_classifier(num_classes=num_classes, dropout_rate=0.3)

    verify_forward_pass(model, input_shape=(2, 224, 224, 3), num_classes=num_classes)

    best_model = None
    final_ckpt = None

    # 4. Train by paradigm
    if mode == "feature_extraction":
        set_backbone_trainable(model, base_model=base_model, unfreeze_from_layer=None)
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_fe_best.keras")
        hist_path = os.path.join(hist_dir, f"{run_name}_fe_history.csv")
        trained_model, history, meta = compile_and_train_vgg16(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_path=ckpt_path,
            history_csv_path=hist_path,
            learning_rate=lr_s1,
            epochs=epochs,
            class_weights=class_weights,
        )
        plot_training_curves(history, exp_name=f"{run_name}_fe", output_plot_path=os.path.join(plots_dir, f"curves_{run_name}_fe.png"))
        best_model = trained_model
        final_ckpt = ckpt_path

    elif mode == "fine_tuning":
        unfreeze_layer = f"block{unfreeze_block}_conv1"
        set_backbone_trainable(model, base_model=base_model, unfreeze_from_layer=unfreeze_layer)
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_ft_best.keras")
        hist_path = os.path.join(hist_dir, f"{run_name}_ft_history.csv")
        trained_model, history, meta = compile_and_train_vgg16(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_path=ckpt_path,
            history_csv_path=hist_path,
            learning_rate=lr_s2,
            epochs=epochs,
            class_weights=class_weights,
        )
        plot_training_curves(history, exp_name=f"{run_name}_ft", output_plot_path=os.path.join(plots_dir, f"curves_{run_name}_ft.png"))
        best_model = trained_model
        final_ckpt = ckpt_path

    elif mode == "two_stage":
        # Stage 1: Feature Extraction
        if HAS_RICH:
            console.print("\n[bold magenta]======== STAGE 1: FEATURE EXTRACTION (WARM-UP) ========[/bold magenta]")
        set_backbone_trainable(model, base_model=base_model, unfreeze_from_layer=None)
        ckpt_s1 = os.path.join(ckpt_dir, f"{run_name}_stage1_fe_best.keras")
        hist_s1 = os.path.join(hist_dir, f"{run_name}_stage1_fe_history.csv")
        model_s1, history_s1, _ = compile_and_train_vgg16(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_path=ckpt_s1,
            history_csv_path=hist_s1,
            learning_rate=lr_s1,
            epochs=epochs_s1,
            class_weights=class_weights,
        )
        plot_training_curves(history_s1, exp_name=f"{run_name}_stage1", output_plot_path=os.path.join(plots_dir, f"curves_{run_name}_stage1.png"))

        # Stage 2: Fine-Tuning
        if HAS_RICH:
            console.print("\n[bold magenta]======== STAGE 2: FINE-TUNING (UNFREEZE BLOCK 5) ========[/bold magenta]")
        unfreeze_layer = f"block{unfreeze_block}_conv1"
        set_backbone_trainable(model_s1, unfreeze_from_layer=unfreeze_layer)
        ckpt_s2 = os.path.join(ckpt_dir, f"{run_name}_stage2_ft_best.keras")
        hist_s2 = os.path.join(hist_dir, f"{run_name}_stage2_ft_history.csv")
        model_s2, history_s2, _ = compile_and_train_vgg16(
            model=model_s1,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_path=ckpt_s2,
            history_csv_path=hist_s2,
            learning_rate=lr_s2,
            epochs=epochs_s2,
            class_weights=class_weights,
        )
        plot_training_curves(history_s2, exp_name=f"{run_name}_stage2", output_plot_path=os.path.join(plots_dir, f"curves_{run_name}_stage2.png"))
        best_model = model_s2
        final_ckpt = ckpt_s2

    # 5. Post-training evaluation
    if eval_test and best_model is not None:
        if HAS_RICH:
            console.print("\n[bold cyan][4/5] Evaluating Best Checkpoint on Test Set...[/bold cyan]")
        eval_metrics = evaluate_test_set(
            model=best_model,
            test_dataset=test_ds,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            exp_name=run_name,
            output_base_dir=paths["output_root"]
        )

        if HAS_RICH:
            res_table = Table(title="Independent Test Set Evaluation Results", title_style="bold green", border_style="bright_blue")
            res_table.add_column("Metric", style="cyan")
            res_table.add_column("Score", style="bold white", justify="right")
            res_table.add_row("Test Accuracy", f"[bold green]{eval_metrics['test_accuracy'] * 100:.2f}%[/bold green]")
            res_table.add_row("Test Loss", f"[yellow]{eval_metrics['test_loss']:.4f}[/yellow]")
            res_table.add_row("Macro Precision", f"{eval_metrics['precision_macro']:.4f}")
            res_table.add_row("Macro Recall", f"{eval_metrics['recall_macro']:.4f}")
            res_table.add_row("Macro F1-Score", f"[bold cyan]{eval_metrics['f1_macro']:.4f}[/bold cyan]")
            res_table.add_row("Weighted F1-Score", f"[bold magenta]{eval_metrics['f1_weighted']:.4f}[/bold magenta]")
            res_table.add_row("Total Test Samples", f"{eval_metrics['total_test_samples']:,}")
            console.print(res_table)

    # 6. Grad-CAM
    if gradcam and best_model is not None:
        if HAS_RICH:
            console.print("\n[bold cyan][5/5] Generating Grad-CAM Heatmap Visualizations...[/bold cyan]")
        gradcam_dir = os.path.join(paths["gradcam_dir"], run_name)
        explain_predictions_with_gradcam(
            model=best_model,
            test_df=test_slice,
            idx_to_class=idx_to_class,
            output_dir=gradcam_dir,
            num_correct=4,
            num_misclassified=4,
            last_conv_layer_name="block5_conv3",
            random_seed=seed
        )
        if HAS_RICH:
            console.print(f"[bold green][OK] Grad-CAM visualizations saved to:[/bold green] {gradcam_dir}")

    if HAS_RICH:
        console.print(Panel(
            f"[bold green]Pipeline Completed Successfully![/bold green]\n"
            f"Checkpoints saved to : [cyan]{ckpt_dir}[/cyan]\n"
            f"Best Checkpoint File : [bold yellow]{final_ckpt}[/bold yellow]\n"
            f"Training Histories   : [cyan]{hist_dir}[/cyan]\n"
            f"Evaluation Plots     : [cyan]{plots_dir}[/cyan]",
            title="Run Completed",
            border_style="green",
            expand=False
        ))


def main():
    parser = build_arg_parser()

    # If launched with no arguments, launch the Interactive Wizard
    if len(sys.argv) == 1 or "-i" in sys.argv or "--interactive" in sys.argv:
        run_interactive_wizard()
        return

    # Otherwise, execute in batch / headless mode with passed args
    args = parser.parse_args()
    print_cli_banner()
    execute_training_pipeline(
        mode=args.mode,
        run_name=args.run_name,
        epochs=args.epochs,
        epochs_s1=args.epochs_stage1,
        epochs_s2=args.epochs_stage2,
        batch_size=args.batch_size,
        lr_s1=args.lr if args.lr is not None else args.lr_stage1,
        lr_s2=args.lr if args.lr is not None else args.lr_stage2,
        unfreeze_block=args.unfreeze_block,
        no_augmentation=args.no_augmentation,
        no_class_weights=args.no_class_weights,
        quick=args.quick,
        eval_test=args.eval_test,
        gradcam=args.gradcam,
        seed=args.seed,
        config_path=args.config,
        resume_path=args.resume
    )


if __name__ == "__main__":
    main()
