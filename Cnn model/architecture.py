"""
Convolutional Neural Network Architectures for Pediatric Auscultation.

Guaranteed compliance with Django backend:
- Expected input shape: (None, 128, 128, 1)
- Explicit 'last_conv_layer' layer name for authentic Grad-CAM extraction
- Softmax output dimension matching verified_classes exactly
"""

import logging
from typing import Tuple, List, Optional, Dict, Any

try:
    from .config import (
        CNN_INPUT_SHAPE,
        TARGET_CONV_LAYER_NAME,
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        SOUND_CLASSES_7CLASS,
    )
except (ImportError, ValueError):
    from config import (
        CNN_INPUT_SHAPE,
        TARGET_CONV_LAYER_NAME,
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        SOUND_CLASSES_7CLASS,
    )

logger = logging.getLogger("CNNArchitecture")


def build_throat_disease_cnn(
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    num_classes: int = len(DISEASE_CLASSES),
    learning_rate: float = 3e-4,
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.25,
):
    """
    Builds a 3-block 2D-CNN tailored for Pediatric Throat & Respiratory Disease Prediction.

    Classes:
      0: Healthy Control    (SPRSound control group)
      1: Pneumonia          (lung consolidation, crackles)
      2: Wheezing Disorders (Asthma + Bronchitis, airway constriction, wheezing)

    Special structural feature:
      Layer 'last_conv_layer' is explicitly named to allow backend/explanations/services.py
      to compute Grad-CAM saliency heatmaps with zero configuration.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers, optimizers
    except ImportError as exc:
        raise ImportError(
            f"TensorFlow is required to build CNN architectures. {exc}\n"
            "Install with: pip install tensorflow keras"
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
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="drop_gap")(x)

    # Classification Head
    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_fc"
    )(x)
    x = layers.Dropout(0.4, name="drop_fc")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax",
        name="disease_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PediatricThroatDiseaseCNN")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_throat_sound_cnn(
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    num_classes: int = len(SOUND_CLASSES_4CLASS),
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
):
    """
    Builds a 4-block 2D-CNN for Respiratory Sound Classification.
    Works for 4-class (Normal, CAS, DAS, CAS&DAS) and 7-class (including Stridor throat obstruction).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers, optimizers
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

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
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
    layer_names = [l.name for l in model.layers]
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
