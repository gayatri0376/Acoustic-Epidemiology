"""
Convolutional Neural Network Architectures for Pediatric Auscultation.

Guaranteed compliance with Django backend:
- Expected input shape: (None, 128, 128, 1)
- Explicit 'last_conv_layer' layer name for authentic Grad-CAM extraction
- Softmax output dimension matching verified_classes exactly
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# Ensure Cnn model and project root are in sys.path for direct execution and IDE analysis
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import (
        CNN_INPUT_SHAPE,
        TARGET_CONV_LAYER_NAME,
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        SOUND_CLASSES_7CLASS,
    )
except ImportError:
    try:
        from .config import (
            CNN_INPUT_SHAPE,
            TARGET_CONV_LAYER_NAME,
            DISEASE_CLASSES,
            SOUND_CLASSES_4CLASS,
            SOUND_CLASSES_7CLASS,
        )
    except (ImportError, ValueError):
        CNN_INPUT_SHAPE = (128, 128, 1)
        TARGET_CONV_LAYER_NAME = "last_conv_layer"
        DISEASE_CLASSES = ["Healthy Control", "Pneumonia", "Wheezing Disorders"]
        SOUND_CLASSES_4CLASS = ["Normal", "CAS", "DAS", "CAS & DAS"]
        SOUND_CLASSES_7CLASS = [
            "Normal",
            "Fine Crackle",
            "Wheeze",
            "Coarse Crackle",
            "Rhonchi",
            "Wheeze+Crackle",
            "Stridor",
        ]

logger = logging.getLogger("CNNArchitecture")

# Ensure writable KERAS_HOME in sandboxed or restricted permission environments
if "KERAS_HOME" not in os.environ:
    import tempfile
    _safe_keras = os.path.join(tempfile.gettempdir(), ".keras")
    try:
        os.makedirs(_safe_keras, exist_ok=True)
        os.environ["KERAS_HOME"] = _safe_keras
    except Exception:
        pass

# Top-level safe TensorFlow / Keras import
try:
    import tensorflow as tf
    from tensorflow import keras
    layers = tf.keras.layers
    models = tf.keras.models
    regularizers = tf.keras.regularizers
    optimizers = tf.keras.optimizers
except Exception:
    tf = None
    layers = None
    models = None
    regularizers = None
    optimizers = None


def get_sparse_focal_loss(gamma: float = 2.0, alpha: Optional[List[float]] = None):
    """
    Computes Sparse Categorical Focal Loss to address severe pediatric disease class imbalance.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - Down-weights easy majority examples (Pneumonia ~77%)
    - Focuses training gradients on hard minority classes (Healthy Control ~7%, Wheezing ~16%)
    """
    global tf
    if tf is None:
        import tensorflow as tf

    alpha_tensor = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        ce = -tf.math.log(y_pred)
        focal_weight = tf.math.pow(1.0 - y_pred, gamma)
        loss = focal_weight * ce

        if alpha_tensor is not None:
            loss = loss * alpha_tensor

        loss = tf.reduce_sum(loss * y_true_one_hot, axis=-1)
        return tf.reduce_mean(loss)

    focal_loss.__name__ = "focal_loss"
    return focal_loss


def build_throat_disease_cnn(
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    num_classes: int = len(DISEASE_CLASSES),
    learning_rate: float = 3e-4,
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.25,
    loss: str = "focal",
    focal_gamma: float = 2.0,
    focal_alpha: Optional[List[float]] = None,
):
    """
    Builds a 3-block 2D-CNN tailored for Pediatric Throat & Respiratory Disease Prediction
    incorporating Dual Pooling (GAP + GMP) and Focal Loss to combat severe class imbalance.

    Classes:
      0: Healthy Control    (SPRSound control group ~7%)
      1: Pneumonia          (lung consolidation, crackles ~77%)
      2: Wheezing Disorders (Asthma + Bronchitis, airway constriction, wheezing ~16%)

    Special structural features:
      1. Dual Pooling: GAP captures continuous wheezing rhythms; GMP retains sharp crackle bursts.
      2. Focal Loss: Prevents the 77% Pneumonia majority from dominating training gradients.
      3. Layer 'last_conv_layer' is explicitly named for authentic Grad-CAM saliency heatmaps.
    """
    global tf, layers, models, regularizers, optimizers
    if tf is None or layers is None:
        try:
            import tensorflow as _tf
            tf = _tf
            layers = _tf.keras.layers
            models = _tf.keras.models
            regularizers = _tf.keras.regularizers
            optimizers = _tf.keras.optimizers
        except ImportError as exc:
            raise ImportError(
                f"TensorFlow is required to build CNN architectures. {exc}\n"
                "Install with: pip install tensorflow"
            )

    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")

    # Block 1: Low-level acoustic spectro-temporal feature extraction
    x = layers.Conv2D(
        32, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv_b1"
    )(inputs)
    x = layers.BatchNormalization(name="bn_b1")(x)
    x = layers.ReLU(name="relu_b1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b1")(x)
    x = layers.Dropout(dropout_rate, name="drop_b1")(x)

    # Block 2: Intermediate harmonic and adventitious patterns
    x = layers.Conv2D(
        64, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv_b2"
    )(x)
    x = layers.BatchNormalization(name="bn_b2")(x)
    x = layers.ReLU(name="relu_b2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b2")(x)
    x = layers.Dropout(dropout_rate, name="drop_b2")(x)

    # Block 3: High-level semantic features & Grad-CAM target
    x = layers.Conv2D(
        128, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=TARGET_CONV_LAYER_NAME  # 'last_conv_layer'
    )(x)
    x = layers.BatchNormalization(name="bn_b3")(x)
    x = layers.ReLU(name="relu_b3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b3")(x)

    # Dual Pooling: Global Average + Global Max captures transient crackles and harmonic wheezes
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
    pooled = layers.Concatenate(name="dual_pool")([gap, gmp])

    # Classification Head
    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_fc"
    )(pooled)
    x = layers.Dropout(0.3, name="drop_fc")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax",
        name="disease_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PediatricThroatDiseaseCNN")

    if loss == "focal":
        alpha = focal_alpha if focal_alpha is not None else [1.5, 0.6, 1.0]
        model_loss = get_sparse_focal_loss(gamma=focal_gamma, alpha=alpha)
    else:
        model_loss = "sparse_categorical_crossentropy"

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=model_loss,
        metrics=["accuracy"]
    )

    return model


def build_throat_sound_cnn(
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    num_classes: int = len(SOUND_CLASSES_4CLASS),
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
    loss: str = "crossentropy",
    focal_gamma: float = 2.0,
    focal_alpha: Optional[List[float]] = None,
):
    """
    Builds a 4-block 2D-CNN for Respiratory Sound Classification.
    Works for 4-class (Normal, CAS, DAS, CAS&DAS) and 7-class (including Stridor throat obstruction).
    """
    global tf, layers, models, regularizers, optimizers
    if tf is None or layers is None:
        try:
            import tensorflow as _tf
            tf = _tf
            layers = _tf.keras.layers
            models = _tf.keras.models
            regularizers = _tf.keras.regularizers
            optimizers = _tf.keras.optimizers
        except ImportError as exc:
            raise ImportError(f"TensorFlow is required: {exc}")

    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv_b1")(inputs)
    x = layers.BatchNormalization(name="bn_b1")(x)
    x = layers.ReLU(name="relu_b1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b1")(x)
    x = layers.Dropout(0.2, name="drop_b1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv_b2")(x)
    x = layers.BatchNormalization(name="bn_b2")(x)
    x = layers.ReLU(name="relu_b2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b2")(x)
    x = layers.Dropout(0.2, name="drop_b2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv_b3")(x)
    x = layers.BatchNormalization(name="bn_b3")(x)
    x = layers.ReLU(name="relu_b3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b3")(x)
    x = layers.Dropout(0.3, name="drop_b3")(x)

    # Block 4 — Grad-CAM target
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name=TARGET_CONV_LAYER_NAME)(x)
    x = layers.BatchNormalization(name="bn_b4")(x)
    x = layers.ReLU(name="relu_b4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="drop_b4")(x)

    # Head
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_fc")(x)
    x = layers.Dropout(0.4, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="sound_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PediatricRespiratorySoundCNN")

    if loss == "focal":
        alpha = focal_alpha if focal_alpha is not None else [1.5, 0.6, 1.0]
        model_loss = get_sparse_focal_loss(gamma=focal_gamma, alpha=alpha)
    else:
        model_loss = "sparse_categorical_crossentropy"

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=model_loss,
        metrics=["accuracy"]
    )

    return model


def verify_model_for_backend(
    model,
    expected_classes: List[str],
    target_conv_layer: str = TARGET_CONV_LAYER_NAME
) -> Dict[str, Any]:
    """
    Validates that a compiled/loaded model strictly meets all backend requirements.
    Prevents runtime errors in InferenceService, DiseaseInferenceService, and GradCamService.
    """
    import numpy as np

    # 1. Check input shape
    input_shape = getattr(model, 'input_shape', None)
    if input_shape:
        # None is batch dimension: (None, 128, 128, 1)
        expected_3d = (128, 128, 1)
        actual_3d = tuple(input_shape[1:])
        if actual_3d != expected_3d:
            raise ValueError(
                f"Model input shape mismatch: expected (None, {expected_3d}), got {input_shape}."
            )

    # 2. Check output neuron count
    output_shape = getattr(model, 'output_shape', None)
    expected_count = len(expected_classes)
    if output_shape:
        actual_count = output_shape[-1]
        if actual_count != expected_count:
            raise ValueError(
                f"Model output dimension mismatch: output has {actual_count} neurons, "
                f"but verified_classes has {expected_count} ({expected_classes})."
            )

    # 3. Check target Grad-CAM convolutional layer exists
    layer_names = [layer.name for layer in model.layers]
    if target_conv_layer not in layer_names:
        raise ValueError(
            f"Required Grad-CAM layer '{target_conv_layer}' not found in model layers: {layer_names}."
        )

    # 4. Dry-run forward pass with dummy tensor
    dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
    output = model.predict(dummy_input, verbose=0)

    if output.shape != (1, expected_count):
        raise ValueError(f"Unexpected prediction output shape: {output.shape}")

    # Softmax sums to 1.0
    prob_sum = float(np.sum(output[0]))
    if not np.isclose(prob_sum, 1.0, atol=1e-3):
        raise ValueError(f"Output probabilities do not sum to 1.0: sum={prob_sum}")

    return {
        "status": "VERIFIED",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "verified_classes": expected_classes,
        "target_conv_layer": target_conv_layer,
        "parameter_count": model.count_params(),
    }
