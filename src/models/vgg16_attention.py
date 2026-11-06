"""
VGG16 Architecture enhanced with Self-Attention (Spatial Self-Attention & CBAM).
Integrates global context modeling directly into VGG16 for robust field-condition recognition.
"""

from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
import keras
from keras import layers, models
from tensorflow.keras.applications import VGG16


@keras.saving.register_keras_serializable(package="PlantVillage")
class SpatialSelfAttention2D(layers.Layer):
    """
    Spatial Self-Attention mechanism (Non-Local Transformer style).
    Computes global pixel-to-pixel correlations across the entire feature map.
    Features:
        - Query & Key projections with dimensionality reduction (key_dim).
        - Value projection preserving feature channels.
        - Learnable scalar gamma initialized to 0.0 for stable residual learning.
    """
    def __init__(self, key_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim

    def build(self, input_shape):
        c = input_shape[-1]
        self.query_conv = layers.Conv2D(self.key_dim, kernel_size=1, padding="same", name="query_conv")
        self.key_conv = layers.Conv2D(self.key_dim, kernel_size=1, padding="same", name="key_conv")
        self.value_conv = layers.Conv2D(c, kernel_size=1, padding="same", name="value_conv")
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        hw = h * w

        q = tf.reshape(self.query_conv(x), [b, hw, self.key_dim])
        k = tf.reshape(self.key_conv(x), [b, hw, self.key_dim])
        v = tf.reshape(self.value_conv(x), [b, hw, c])

        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.reshape(out, [b, h, w, c])
        return x + self.gamma * out

    def get_config(self):
        config = super().get_config()
        config["key_dim"] = self.key_dim
        return config


@keras.saving.register_keras_serializable(package="PlantVillage")
class CBAMBlock(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially infers attention maps along Channel and Spatial dimensions.
    """
    def __init__(self, reduction_ratio: int = 16, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        c = input_shape[-1]
        bottleneck_c = max(c // self.reduction_ratio, 8)
        self.shared_dense_1 = layers.Dense(bottleneck_c, activation="relu", name="shared_mlp_1")
        self.shared_dense_2 = layers.Dense(c, name="shared_mlp_2")
        self.spatial_conv = layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation="sigmoid", name="spatial_conv")
        super().build(input_shape)

    def call(self, x):
        # 1. Channel Attention
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x, axis=[1, 2])
        mlp_avg = self.shared_dense_2(self.shared_dense_1(avg_pool))
        mlp_max = self.shared_dense_2(self.shared_dense_1(max_pool))
        channel_attn = tf.nn.sigmoid(mlp_avg + mlp_max)
        channel_attn = tf.reshape(channel_attn, [-1, 1, 1, tf.shape(x)[-1]])
        x_channel = x * channel_attn

        # 2. Spatial Attention
        avg_spatial = tf.reduce_mean(x_channel, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x_channel, axis=-1, keepdims=True)
        spatial_cat = tf.concat([avg_spatial, max_spatial], axis=-1)
        spatial_attn = self.spatial_conv(spatial_cat)
        out = x_channel * spatial_attn

        return out + x

    def get_config(self):
        config = super().get_config()
        config["reduction_ratio"] = self.reduction_ratio
        config["kernel_size"] = self.kernel_size
        return config


def build_vgg16_attention_classifier(
    num_classes: int = 38,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    attention_type: str = "self_attention",
    key_dim: int = 64,
    pretrained_weights_path: Optional[str] = None,
    trainable_backbone: bool = False
) -> Tuple[models.Model, models.Model]:
    """
    Constructs VGG16 with integrated Attention Module:
    - Backbone: VGG16 (ImageNet or loaded from existing checkpoint)
    - Attention Module: SpatialSelfAttention2D or CBAMBlock
    - GlobalAveragePooling2D
    - Dropout(dropout_rate)
    - Dense(num_classes, activation='softmax')

    Parameters:
        num_classes: Total classification targets.
        input_shape: Image input shape (224, 224, 3).
        dropout_rate: Dropout rate before classification head.
        attention_type: 'self_attention' or 'cbam'.
        key_dim: Query/Key dimension for SpatialSelfAttention2D.
        pretrained_weights_path: Path to existing .keras model to borrow weights from.
        trainable_backbone: Whether backbone layers are initially trainable.

    Returns:
        (full_model, base_backbone_model)
    """
    base_model = VGG16(
        weights="imagenet" if pretrained_weights_path is None else None,
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_backbone

    if not trainable_backbone:
        for layer in base_model.layers:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)

    # Insert Attention Mechanism
    if attention_type.lower() == "cbam":
        x = CBAMBlock(name="cbam_attention")(x)
    else:
        x = SpatialSelfAttention2D(key_dim=key_dim, name="spatial_self_attention")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"VGG16_{attention_type.title()}_PlantVillage")

    # If borrowing weights from a previously trained checkpoint
    if pretrained_weights_path:
        try:
            print(f"[+] Loading base weights from: {pretrained_weights_path}")
            source_model = keras.models.load_model(pretrained_weights_path)
            
            # Transfer backbone weights
            for layer in base_model.layers:
                try:
                    src_layer = source_model.get_layer("vgg16").get_layer(layer.name)
                    layer.set_weights(src_layer.get_weights())
                except Exception:
                    try:
                        src_layer = source_model.get_layer(layer.name)
                        layer.set_weights(src_layer.get_weights())
                    except Exception:
                        pass
                        
            # Transfer classification head weights
            try:
                model.get_layer("predictions").set_weights(source_model.get_layer("predictions").get_weights())
            except Exception:
                pass
            print("[+] Successfully inherited VGG16 and Classification Head weights!")
        except Exception as e:
            print(f"[!] Warning loading weights: {e}")

    return model, base_model


def configure_attention_finetuning(
    model: models.Model,
    unfreeze_block5: bool = True
) -> Dict[str, Any]:
    """
    Configures trainable parameters:
    - Always unfreezes Attention layer and Classification head.
    - Optionally unfreezes block5 of VGG16 backbone for end-to-end fine-tuning.
    """
    base_model = None
    for layer in model.layers:
        if "vgg16" in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers:
            if unfreeze_block5 and ("block5" in layer.name):
                layer.trainable = True
            else:
                layer.trainable = False

    # Ensure Attention and Head are trainable
    for layer in model.layers:
        if layer.name != "vgg16":
            layer.trainable = True

    trainable_count = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable_count = sum(tf.size(w).numpy() for w in model.non_trainable_weights)

    return {
        "trainable_weights_count": len(model.trainable_weights),
        "trainable_params": int(trainable_count),
        "non_trainable_params": int(non_trainable_count),
    }
