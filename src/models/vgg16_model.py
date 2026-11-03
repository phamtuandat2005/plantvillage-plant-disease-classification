"""
VGG16 model architecture definition, backbone freezing/unfreezing, and parameter inspection.
"""

from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
# pyrefly: ignore[missing-import]
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models


def build_vgg16_classifier(
    num_classes: int = 38,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    weights: str = "imagenet",
    trainable_backbone: bool = False
) -> Tuple[models.Model, VGG16]:
    """
    Constructs the VGG16 Transfer Learning model:
    - Backbone: VGG16 pre-trained on ImageNet without top dense layers.
    - GlobalAveragePooling2D
    - Dropout(dropout_rate)
    - Dense(num_classes, activation='softmax')

    Parameters:
        num_classes (int): Number of target classes.
        input_shape (tuple): (height, width, channels).
        dropout_rate (float): Dropout probability (default: 0.3).
        weights (str): Pretrained weights source ('imagenet' or None).
        trainable_backbone (bool): Whether backbone weights are initially trainable.

    Returns:
        tuple: (full_model, base_model)
    """
    base_model = VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_backbone

    # If backbone is frozen, ensure all individual layers are frozen
    if not trainable_backbone:
        for layer in base_model.layers:
            layer.trainable = False

    # Functional API assembly for full inspectability and Grad-CAM integration
    inputs = layers.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VGG16_PlantVillage")

    return model, base_model


def set_backbone_trainable(
    model: Any,
    base_model: Optional[Any] = None,
    unfreeze_from_layer: Optional[str] = "block5_conv1",
    unfreeze_from_block: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Configures fine-tuning by unfreezing layers from a given layer name upwards,
    keeping earlier convolutional blocks strictly frozen.
    """
    if isinstance(model, (tuple, list)):
        if base_model is None and len(model) > 1:
            base_model = model[1]
        model = model[0]

    if unfreeze_from_block is not None:
        unfreeze_from_layer = f"block{unfreeze_from_block}_conv1" if unfreeze_from_block else None

    if isinstance(base_model, str):
        unfreeze_from_layer = base_model
        base_model = None

    if base_model is None:
        for layer in model.layers:
            if "vgg16" in layer.name.lower():
                base_model = layer
                break

    if base_model is None:
        base_model = model

    if unfreeze_from_layer is None:
        base_model.trainable = False
        for layer in getattr(base_model, "layers", []):
            layer.trainable = False
        unfrozen_layers = []
    else:
        base_model.trainable = True
        unfreeze = False
        unfrozen_layers = []

        for layer in getattr(base_model, "layers", []):
            if layer.name == unfreeze_from_layer:
                unfreeze = True

            if unfreeze:
                layer.trainable = True
                unfrozen_layers.append(layer.name)
            else:
                layer.trainable = False

    param_info = count_parameters(model)
    return {
        "unfrozen_layers": unfrozen_layers,
        "unfrozen_layer_count": len(unfrozen_layers),
        "trainable_params": param_info["trainable_params"],
        "non_trainable_params": param_info["non_trainable_params"],
        "total_params": param_info["total_params"],
    }


def count_parameters(model: Any) -> Dict[str, int]:
    """
    Accurately counts total, trainable, and non-trainable parameters.

    Parameters:
        model: Keras model (or tuple).

    Returns:
        dict: Parameter breakdown.
    """
    if isinstance(model, (tuple, list)):
        model = model[0]

    trainable_count = sum(p.numpy().size for p in model.trainable_variables)
    non_trainable_count = sum(p.numpy().size for p in model.non_trainable_variables)
    total_count = trainable_count + non_trainable_count

    return {
        "total_params": int(total_count),
        "trainable_params": int(trainable_count),
        "non_trainable_params": int(non_trainable_count),
    }


def verify_forward_pass(
    model: Any,
    input_shape: Any = (224, 224, 3),
    batch_size: int = 2,
    num_classes: int = 38
) -> bool:
    """
    Runs a test forward pass with a dummy tensor to verify shapes and prevent runtime errors.
    """
    if isinstance(model, (tuple, list)):
        model = model[0]

    if isinstance(input_shape, tuple) and len(input_shape) == 4:
        batch_size = input_shape[0]
        input_shape = input_shape[1:]

    dummy_input = tf.random.normal([batch_size, *input_shape])
    dummy_output = model(dummy_input, training=False)
    expected_shape = (batch_size, num_classes)

    if tuple(dummy_output.shape) != expected_shape:
        raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {dummy_output.shape}")

    return True

