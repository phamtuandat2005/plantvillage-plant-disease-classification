"""
Universal Model Loader, Evaluator, and Comparator.
Supports loading and comparing models across formats:
- .keras (Keras 3 / TensorFlow native format)
- .h5 (Legacy Keras / TensorFlow HDF5 format)
- .pth / .pt (PyTorch models: TorchScript, full models, or state dictionaries)

Computes comprehensive diagnostic metrics:
- Overall Accuracy (% correct)
- Error Rate (% wrong)
- Macro & Weighted F1-score, Precision, Recall
- Inference Latency (ms / sample)
- Visual 9-image diagnosis grid with confidence scores and error highlights.

==============================================================================
UNIFIED TEAM PREPROCESSING STANDARD (Group Consensus)
==============================================================================
Pipeline: RGB image (H=224, W=224, C=3) with pixel values in [0, 255]

Step 1 — Resize  : 224 × 224 pixels (PIL BILINEAR, RGB, 3 channels)
Step 2 — Scale   : divide by 255.0  →  [0.0, 1.0]
Step 3 — Normalize:
    x = (x - mean) / std
    mean = [0.485, 0.456, 0.406]   (ImageNet RGB mean)
    std  = [0.229, 0.224, 0.225]   (ImageNet RGB std)

This matches the torchvision / tf.keras.applications standard for
VGG16, ResNet50, MobileNetV2 when used with transfer learning.

NOTE: Models that already embed an internal preprocessing layer
      (e.g. a Rescaling or custom bgr-conversion Conv2D baked into
      the .keras / .h5 file) are auto-detected and receive raw [0, 255]
      pixel arrays — avoiding double-preprocessing artifacts.
==============================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore[missing-import]
from PIL import Image

# Suppress TF verbose logs and avoid OpenMP duplicate lib collision on Windows
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def enable_legacy_keras2_compatibility():
    """
    Enables seamless loading of legacy Keras 2 .h5 models into Keras 3
    by patching known serialization incompatibilities (input_axes, quantization_config, renorm, etc.).
    """
    try:
        # pyrefly: ignore[missing-import]
        import keras
        import inspect

        # 1. Patch initializers to ignore legacy 'input_axes' and 'output_axes'
        for init_name in dir(keras.initializers):
            init_cls = getattr(keras.initializers, init_name)
            if isinstance(init_cls, type) and issubclass(init_cls, keras.initializers.Initializer):
                orig_init = init_cls.__init__
                def make_patched_init(orig_fn):
                    def patched(self, *args, **kwargs):
                        kwargs.pop("input_axes", None)
                        kwargs.pop("output_axes", None)
                        return orig_fn(self, *args, **kwargs)
                    return patched
                init_cls.__init__ = make_patched_init(orig_init)

        # 2. Patch Layer.from_config to filter out unsupported legacy Keras 2 kwargs
        orig_layer_from_config = keras.layers.Layer.from_config

        @classmethod
        def patched_layer_from_config(cls, config):
            for bad_key in ["quantization_config", "renorm", "renorm_clipping", "renorm_momentum", "synchronized"]:
                config.pop(bad_key, None)
            try:
                return orig_layer_from_config.__func__(cls, config)
            except (TypeError, ValueError) as e:
                sig = inspect.signature(cls.__init__)
                params = sig.parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    raise e
                valid_keys = set(params.keys())
                filtered_config = {k: v for k, v in config.items() if k in valid_keys}
                return orig_layer_from_config.__func__(cls, filtered_config)

        keras.layers.Layer.from_config = patched_layer_from_config
    except Exception:
        pass


# Activate legacy compatibility on module load
enable_legacy_keras2_compatibility()


class UniversalModelWrapper:
    """
    Standardized inference wrapper encapsulating models across Keras (.keras, .h5)
    and PyTorch (.pth, .pt) frameworks.
    """

    def __init__(
        self,
        model: Any,
        framework: str,
        name: str,
        model_path: str,
        num_classes: int = 38,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.framework = framework.lower()  # 'tensorflow' or 'pytorch'
        self.name = name
        self.model_path = model_path
        
        # Auto-detect true num_classes from model output if available
        if hasattr(model, "output_shape") and model.output_shape:
            out_dim = model.output_shape[-1]
            if isinstance(out_dim, int) and out_dim > 0:
                num_classes = out_dim
        
        self.num_classes = num_classes
        if class_names and len(class_names) == num_classes:
            self.class_names = class_names
        else:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]
            
        self.device = device
        self._file_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0
        # Auto-detect preprocessing scheme required by model
        self.preprocess_mode = self._detect_preprocess_mode()
        self._param_count = self._compute_param_count()

    def _compute_param_count(self) -> int:
        """Returns total parameter count of the underlying model."""
        try:
            if self.framework == "tensorflow":
                return int(self.model.count_params())
            elif self.framework == "pytorch":
                return sum(p.numel() for p in self.model.parameters())
        except Exception:
            return 0
        return 0

    @property
    def param_count(self) -> int:
        return self._param_count

    @property
    def file_size_mb(self) -> float:
        return self._file_size_mb

    def _detect_preprocess_mode(self) -> str:
        """
        Determines the exact preprocessing required by the model:
        - 'imagenet_norm': (x/255 - mean) / std (ImageNet RGB standard, used by PyTorch & custom ResNet50)
        - 'tf_vgg_bgr': tf.keras.applications.vgg16.preprocess_input (BGR zero-centering, used by TF VGG16)
        - 'raw': raw [0, 255] pixels (for models that have full input preprocessing layers inside)
        """
        if self.framework == "pytorch":
            return "imagenet_norm"

        # Check TF model
        m_name = (self.name or "").lower()
        p_name = (self.model_path or "").lower()

        # If model has internal BGR conversion layer like 'imagenet_rgb_to_keras_bgr':
        # Notice that this layer expects normalized RGB: (x/255 - mean)/std !
        # Its weights are std_scaler and mean_bias designed to transform normalized RGB into Keras BGR.
        for layer in getattr(self.model, "layers", []):
            lname = (layer.name or "").lower()
            if "imagenet_rgb_to_keras_bgr" in lname or "rgb_to" in lname:
                return "imagenet_norm"

        # If model is VGG16 (TF / Keras applications)
        if "vgg16" in m_name or "vgg16" in p_name:
            return "tf_vgg_bgr"

        # If model has other builtin preprocessing (e.g. Rescaling(1./255))
        for layer in getattr(self.model, "layers", []):
            ltype = type(layer).__name__.lower()
            lname = (layer.name or "").lower()
            if any(k in ltype for k in ["rescaling", "normalization"]):
                return "raw"
            if any(k in lname for k in ["rescal", "normaliz"]):
                return "raw"

        # Default fallback
        return "imagenet_norm"

    def preprocess_images(self, images_np: np.ndarray) -> Any:
        """
        Applies model-specific preprocessing to ensure optimal inference accuracy.

        Input: numpy array (N, 224, 224, 3), dtype float32, RGB pixel values [0, 255].
        """
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.preprocess_mode == "tf_vgg_bgr":
            # TF VGG16 application standard: RGB -> BGR, zero-center by ImageNet mean
            # pyrefly: ignore[missing-import]
            import tensorflow as tf
            return tf.keras.applications.vgg16.preprocess_input(images_np.copy())

        elif self.preprocess_mode == "raw":
            return images_np.astype(np.float32)

        elif self.preprocess_mode == "imagenet_norm":
            if self.framework == "tensorflow":
                # (x/255 - mean) / std (RGB)
                x = images_np.astype(np.float32) / 255.0
                return (x - IMAGENET_MEAN) / IMAGENET_STD
            elif self.framework == "pytorch":
                # pyrefly: ignore[missing-import]
                import torch
                tensor = torch.from_numpy(images_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
                mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
                std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
                tensor = (tensor - mean) / std
                if self.device != "cpu":
                    tensor = tensor.to(self.device)
                return tensor

        return images_np

    def predict_proba(self, images_np: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generates class probability distributions of shape (N, num_classes).
        Guaranteed to sum to 1.0 per sample.
        Automatically batches inputs to prevent GPU/CPU Out-of-Memory (ResourceExhaustedError).
        """
        n_samples = len(images_np)
        if n_samples == 0:
            return np.empty((0, self.num_classes), dtype=np.float32)

        # Automatic chunking for large sample sets to prevent OOM
        if n_samples > batch_size:
            chunk_probs = []
            for i in range(0, n_samples, batch_size):
                batch_arr = images_np[i : i + batch_size]
                chunk_probs.append(self.predict_proba(batch_arr, batch_size=batch_size))
            return np.vstack(chunk_probs)

        if self.framework == "tensorflow":
            import tensorflow as tf
            processed = self.preprocess_images(images_np)
            raw_out = self.model(processed, training=False)
            if hasattr(raw_out, "numpy"):
                raw_out = raw_out.numpy()
            # If model output is not softmaxed (e.g. logits)
            if not np.allclose(np.sum(raw_out, axis=-1), 1.0, atol=1e-2):
                exp = np.exp(raw_out - np.max(raw_out, axis=-1, keepdims=True))
                raw_out = exp / np.sum(exp, axis=-1, keepdims=True)
            return raw_out.astype(np.float32)

        elif self.framework == "pytorch":
            # pyrefly: ignore[missing-import]
            import torch
            processed = self.preprocess_images(images_np)
            self.model.eval()
            with torch.no_grad():
                raw_out = self.model(processed)
                probs = torch.softmax(raw_out, dim=-1)
                return probs.cpu().numpy().astype(np.float32)

        raise ValueError(f"Unsupported framework: {self.framework}")

    def predict_classes(self, images_np: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Returns predicted class indices of shape (N,)."""
        probs = self.predict_proba(images_np, batch_size=batch_size)
        return np.argmax(probs, axis=-1)

    def predict_single_image(self, image_input: Union[str, np.ndarray, Image.Image], top_k: int = 5) -> Dict[str, Any]:
        """
        Runs prediction on a single image and returns top-k class rankings with probabilities.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
            img = img.resize((224, 224))
            img_arr = np.array(img, dtype=np.float32)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB").resize((224, 224))
            img_arr = np.array(img, dtype=np.float32)
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[:2] != (224, 224):
                img = Image.fromarray(image_input.astype(np.uint8)).resize((224, 224))
                img_arr = np.array(img, dtype=np.float32)
            else:
                img_arr = image_input.astype(np.float32)
        else:
            raise TypeError("Unsupported image input type.")

        batch_arr = np.expand_dims(img_arr, axis=0)
        start_t = time.perf_counter()
        probs = self.predict_proba(batch_arr)[0]
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "rank": rank + 1,
                "class_idx": int(idx),
                "class_name": self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}",
                "confidence": float(probs[idx]),
                "confidence_pct": f"{probs[idx] * 100.0:.2f}%",
            }
            for rank, idx in enumerate(top_indices)
        ]

        return {
            "model_name": self.name,
            "predicted_class_idx": int(top_indices[0]),
            "predicted_class_name": top_predictions[0]["class_name"],
            "top_confidence": float(probs[top_indices[0]]),
            "top_confidence_pct": top_predictions[0]["confidence_pct"],
            "top_predictions": top_predictions,
            "latency_ms": latency_ms,
        }


def load_universal_model(
    model_path: Union[str, Path],
    model_name: Optional[str] = None,
    num_classes: int = 38,
    class_names: Optional[List[str]] = None,
    device: str = "auto",
    pytorch_arch: str = "vgg16",
) -> UniversalModelWrapper:
    """
    Universal model loader supporting:
    - .keras (Keras 3 / TensorFlow)
    - .h5 (Legacy Keras / TensorFlow HDF5)
    - .pth / .pt (PyTorch: TorchScript, full model, or state_dict)
    """
    model_path_str = str(model_path)
    if not os.path.exists(model_path_str):
        raise FileNotFoundError(f"Model file not found at: {model_path_str}")

    ext = Path(model_path_str).suffix.lower()
    base_name = model_name or Path(model_path_str).stem

    # Detect compute device
    if device == "auto":
        try:
            # pyrefly: ignore[missing-import]
            import torch
            pt_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pt_device = "cpu"
    else:
        pt_device = device

    # 1. Keras / TensorFlow (.keras or .h5)
    if ext in [".keras", ".h5"]:
        import tensorflow as tf
        enable_legacy_keras2_compatibility()
        model = tf.keras.models.load_model(model_path_str, compile=False)
        return UniversalModelWrapper(
            model=model,
            framework="tensorflow",
            name=base_name,
            model_path=model_path_str,
            num_classes=num_classes,
            class_names=class_names,
            device="gpu" if tf.config.list_physical_devices("GPU") else "cpu",
        )

    # 2. PyTorch (.pth or .pt)
    elif ext in [".pth", ".pt"]:
        # pyrefly: ignore[missing-import]
        import torch
        # pyrefly: ignore[missing-import]
        import torchvision.models as tv_models

        # Case A: TorchScript
        try:
            model = torch.jit.load(model_path_str, map_location=pt_device)
            model.eval()
            return UniversalModelWrapper(
                model=model,
                framework="pytorch",
                name=base_name,
                model_path=model_path_str,
                num_classes=num_classes,
                class_names=class_names,
                device=pt_device,
            )
        except Exception:
            pass

        # Case B: Standard PyTorch model or state_dict
        loaded_obj = torch.load(model_path_str, map_location=pt_device, weights_only=False)

        # If it's already an nn.Module
        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj.to(pt_device)
            model.eval()
            return UniversalModelWrapper(
                model=model,
                framework="pytorch",
                name=base_name,
                model_path=model_path_str,
                num_classes=num_classes,
                class_names=class_names,
                device=pt_device,
            )

        # If it's a state_dict or checkpoint dictionary
        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj.get("model_state_dict", loaded_obj.get("state_dict", loaded_obj))

            # Auto-detect number of classes from final layer
            fc_keys = [k for k in state_dict if ("classifier" in k or "fc" in k) and "weight" in k]
            if fc_keys:
                num_classes = int(state_dict[fc_keys[-1]].shape[0])

            # Auto-detect architecture from name or state_dict shapes
            name_lower = Path(model_path_str).stem.lower()
            arch = pytorch_arch.lower()
            if arch in ["auto", "vgg16"]:
                if "alexnet" in name_lower or ("features.0.weight" in state_dict and state_dict["features.0.weight"].shape[-1] == 11):
                    arch = "alexnet"
                elif "resnet" in name_lower or "conv1.weight" in state_dict:
                    arch = "resnet50"
                elif "mobilenet" in name_lower or "features.0.0.weight" in state_dict:
                    arch = "mobilenet_v3"

            if arch == "alexnet":
                model = tv_models.alexnet(weights=None)
                model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
            elif arch == "resnet50":
                model = tv_models.resnet50(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            else:
                model = tv_models.vgg16(weights=None)
                model.classifier[6] = torch.nn.Linear(4096, num_classes)

            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                try:
                    model.load_state_dict(state_dict, strict=False)
                except Exception as inner_e:
                    print(f"[Warning] PyTorch state_dict loading issue: {inner_e}")

            model = model.to(pt_device)
            model.eval()
            return UniversalModelWrapper(
                model=model,
                framework="pytorch",
                name=base_name,
                model_path=model_path_str,
                num_classes=num_classes,
                class_names=class_names,
                device=pt_device,
            )

    raise ValueError(f"Unsupported model extension: {ext}. Expected .keras, .h5, .pth, or .pt.")


def evaluate_model_on_dataset(
    model_wrapper: UniversalModelWrapper,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluates a UniversalModelWrapper on a dataset of images and integer labels.
    Computes exact counts and percentages of correct and wrong predictions.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    n_samples = len(images)
    all_preds = []
    all_probs = []

    start_time = time.perf_counter()
    for i in range(0, n_samples, batch_size):
        batch_img = images[i : i + batch_size]
        probs = model_wrapper.predict_proba(batch_img)
        preds = np.argmax(probs, axis=-1)
        all_preds.extend(preds)
        all_probs.extend(probs)
    total_time = time.perf_counter() - start_time

    all_preds = np.array(all_preds, dtype=np.int32)
    all_probs = np.array(all_probs, dtype=np.float32)

    is_correct = (all_preds == labels)
    correct_count = int(np.sum(is_correct))
    wrong_count = int(n_samples - correct_count)

    accuracy_pct = (correct_count / n_samples) * 100.0 if n_samples > 0 else 0.0
    error_pct = (wrong_count / n_samples) * 100.0 if n_samples > 0 else 0.0

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, all_preds, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, all_preds, average="weighted", zero_division=0
    )

    latency_per_sample_ms = (total_time / n_samples) * 1000.0 if n_samples > 0 else 0.0

    return {
        "model_name": model_wrapper.name,
        "framework": model_wrapper.framework,
        "format": Path(model_wrapper.model_path).suffix,
        "total_samples": n_samples,
        "correct_count": correct_count,
        "correct_pct": accuracy_pct,
        "wrong_count": wrong_count,
        "wrong_pct": error_pct,
        "accuracy": accuracy_score(labels, all_preds),
        "macro_f1": float(f1_m),
        "weighted_f1": float(f1_w),
        "macro_precision": float(precision_m),
        "macro_recall": float(recall_m),
        "latency_ms_per_sample": latency_per_sample_ms,
        "total_eval_time_sec": total_time,
        "param_count": model_wrapper.param_count,
        "file_size_mb": model_wrapper.file_size_mb,
        "predictions": all_preds,
        "probabilities": all_probs,
        "is_correct": is_correct,
    }


def compare_multiple_models(
    models: List[UniversalModelWrapper],
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates multiple models side-by-side and produces a formatted comparison table.
    """
    results_dict = {}
    summary_rows = []

    for wrapper in models:
        print(f"---> Evaluating {wrapper.name} on {len(images)} images...")
        res = evaluate_model_on_dataset(wrapper, images, labels, batch_size=batch_size)
        results_dict[wrapper.name] = res
        summary_rows.append({
            "Model": res["model_name"],
            "Format": res["format"],
            "Framework": res["framework"],
            "Total": res["total_samples"],
            "Correct": res["correct_count"],
            "Correct (%)": f"{res['correct_pct']:.2f}%",
            "Wrong": res["wrong_count"],
            "Wrong (%)": f"{res['wrong_pct']:.2f}%",
            "Accuracy": f"{res['accuracy'] * 100.0:.2f}%",
            "Macro F1": f"{res['macro_f1']:.4f}",
            "Latency (ms/img)": f"{res['latency_ms_per_sample']:.2f}",
            "Params": f"{res['param_count']:,}",
            "Size (MB)": f"{res['file_size_mb']:.1f}",
        })
        print(f"---> {wrapper.name} evaluation completed.")

    comparison_df = pd.DataFrame(summary_rows)
    return comparison_df, results_dict


def visualize_9_predictions(
    models: List[UniversalModelWrapper],
    images: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    eval_results: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: str = "random",  # 'random', 'misclassified', 'disagreement', 'correct'
    random_seed: int = 42,
    figsize: Tuple[int, int] = (15, 15),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualizes a 3x3 grid (9 images) displaying:
    - Original RGB plant leaf image
    - Ground Truth class
    - Predictions and confidence scores from all models
    - Visual indicators (Green if correct, Red if wrong)
    - Accuracy percentage and error percentage for the 9 displayed images.
    """
    np.random.seed(random_seed)
    n_total = len(images)
    if n_total < 9:
        raise ValueError(f"Need at least 9 images to display a 3x3 grid, found {n_total}.")

    # Gather predictions for each model if not pre-computed
    preds_by_model = {}
    correct_by_model = {}
    for wrapper in models:
        if eval_results and wrapper.name in eval_results:
            preds_by_model[wrapper.name] = eval_results[wrapper.name]["predictions"]
            correct_by_model[wrapper.name] = eval_results[wrapper.name]["is_correct"]
        else:
            probs = wrapper.predict_proba(images)
            preds = np.argmax(probs, axis=-1)
            preds_by_model[wrapper.name] = preds
            correct_by_model[wrapper.name] = (preds == labels)

    # Select candidate indices according to mode
    if mode == "misclassified":
        # Any model made an error
        any_wrong = np.zeros(n_total, dtype=bool)
        for name in preds_by_model:
            any_wrong |= ~correct_by_model[name]
        candidates = np.where(any_wrong)[0]
    elif mode == "disagreement":
        # Models disagree with each other
        if len(models) >= 2:
            model_names = list(preds_by_model.keys())
            disagree = (preds_by_model[model_names[0]] != preds_by_model[model_names[1]])
            candidates = np.where(disagree)[0]
        else:
            candidates = np.arange(n_total)
    elif mode == "correct":
        # All models correct
        all_correct = np.ones(n_total, dtype=bool)
        for name in preds_by_model:
            all_correct &= correct_by_model[name]
        candidates = np.where(all_correct)[0]
    else:  # 'random'
        candidates = np.arange(n_total)

    # Fallback to random if insufficient candidates
    if len(candidates) < 9:
        candidates = np.arange(n_total)

    selected_indices = np.random.choice(candidates, size=9, replace=False)

    # Create 3x3 Figure
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()

    # Track 9-image statistics per model
    stats_9 = {m.name: {"correct": 0, "wrong": 0} for m in models}

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        img_np = images[idx].astype(np.uint8)
        true_lbl = int(labels[idx])
        true_name = class_names[true_lbl] if true_lbl < len(class_names) else f"Class_{true_lbl}"

        # Clean display image
        ax.imshow(img_np)
        ax.set_xticks([])
        ax.set_yticks([])

        # Build multi-line caption for predictions
        title_lines = [f"True: {true_name}"]
        is_any_wrong = False

        for wrapper in models:
            pred_lbl = int(preds_by_model[wrapper.name][idx])
            pred_name = class_names[pred_lbl] if pred_lbl < len(class_names) else f"Class_{pred_lbl}"
            is_corr = (pred_lbl == true_lbl)

            if is_corr:
                stats_9[wrapper.name]["correct"] += 1
                status_tag = "[OK]"
            else:
                stats_9[wrapper.name]["wrong"] += 1
                status_tag = "[ERR]"
                is_any_wrong = True

            # Shorten model name if long
            m_label = wrapper.name[:16]
            title_lines.append(f"{status_tag} {m_label}: {pred_name}")

        caption = "\n".join(title_lines)
        # Highlight border if any error
        border_color = "red" if is_any_wrong else "green"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        ax.set_title(caption, fontsize=9, loc="left", pad=4)

    # Construct overall summary banner for the 9 displayed samples
    summary_banner_lines = [f"Visual Diagnosis of 9 Samples (Mode: {mode.upper()})"]
    for m in models:
        corr = stats_9[m.name]["correct"]
        wrg = stats_9[m.name]["wrong"]
        pct_corr = (corr / 9.0) * 100.0
        pct_wrg = (wrg / 9.0) * 100.0
        summary_banner_lines.append(
            f"{m.name}: {corr}/9 Correct ({pct_corr:.1f}%) | {wrg}/9 Wrong ({pct_wrg:.1f}%)"
        )

    fig.suptitle("\n".join(summary_banner_lines), fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_model_comparison_charts(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generates academic bar plots comparing:
    1. Correct vs Wrong percentage breakdown.
    2. Macro F1-score & Inference latency with clear combined legend and value tags.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    models = comparison_df["Model"].tolist()
    # Parse percentages back to float if strings
    correct_pcts = [float(str(x).replace("%", "")) for x in comparison_df["Correct (%)"]]
    wrong_pcts = [float(str(x).replace("%", "")) for x in comparison_df["Wrong (%)"]]
    f1_scores = [float(x) for x in comparison_df["Macro F1"]]
    latencies = [float(x) for x in comparison_df["Latency (ms/img)"]]

    x = np.arange(len(models))
    width = 0.35

    # -------------------------------------------------------------
    # Biểu đồ 1: Tỷ lệ dự đoán Đúng (%) vs Sai (%)
    # -------------------------------------------------------------
    b1 = ax1.bar(x - width/2, correct_pcts, width, label="Đoán Đúng / Correct (%)", color="#2ecc71")
    b2 = ax1.bar(x + width/2, wrong_pcts, width, label="Đoán Sai / Error (%)", color="#e74c3c")
    ax1.set_ylabel("Tỷ Lệ Phần Trăm (%)", fontweight="bold", fontsize=11)
    ax1.set_title("1. Tỷ Lệ Dự Đoán Đúng vs Sai Của Từng Mô Hình", fontweight="bold", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Hiển thị số liệu trực tiếp trên đỉnh cột
    for i, (c, w) in enumerate(zip(correct_pcts, wrong_pcts)):
        ax1.text(i - width/2, c + 1.5, f"{c:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#1e8449")
        ax1.text(i + width/2, w + 1.5, f"{w:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#c0392b")

    # -------------------------------------------------------------
    # Biểu đồ 2: Chất lượng phân loại (F1-score) vs Tốc độ (Latency)
    # -------------------------------------------------------------
    ax2_twin = ax2.twinx()
    p1 = ax2.bar(x - width/2, f1_scores, width, label="Cột Xanh: Macro F1-Score (Chất lượng)", color="#2980b9")
    p2 = ax2_twin.bar(x + width/2, latencies, width, label="Cột Cam: Latency ms/ảnh (Tốc độ)", color="#e67e22")

    # Trục trái (Xanh): F1-score
    ax2.set_ylabel("Macro F1-Score (0.0 -> 1.0)", color="#2980b9", fontweight="bold", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0, 1.15)

    # Trục phải (Cam): Latency
    max_lat = max(latencies) if latencies else 10.0
    ax2_twin.set_ylabel("Thời Gian Xử Lý / Latency (ms / ảnh)", color="#e67e22", fontweight="bold", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#e67e22")
    ax2_twin.set_ylim(0, max_lat * 1.25 if max_lat > 0 else 10.0)

    ax2.set_title("2. Chất Lượng Phân Loại (F1) vs Tốc Độ Suy Luận (Latency)", fontweight="bold", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Hiển thị số liệu trực tiếp trên đỉnh cột
    for i, (f1_val, lat_val) in enumerate(zip(f1_scores, latencies)):
        ax2.text(i - width/2, f1_val + 0.02, f"{f1_val:.3f}", ha="center", fontsize=9, fontweight="bold", color="#1b4f72")
        ax2_twin.text(i + width/2, lat_val + (max_lat * 0.02), f"{lat_val:.1f} ms", ha="center", fontsize=9, fontweight="bold", color="#a04000")

    # Hộp CHÚ THÍCH (Legend) kết hợp cả 2 cột rõ ràng
    combined_bars = [p1, p2]
    combined_labels = [b.get_label() for b in combined_bars]
    ax2.legend(combined_bars, combined_labels, loc="upper left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
