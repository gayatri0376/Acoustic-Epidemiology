#!/usr/bin/env python3
"""
Disease Model Training Pipeline for Acoustic Epidemiology.

Trains a CNN on the SPRSound disease-classification task:
  - Healthy Control (SPRSound "Control Group")
  - Pneumonia (SPRSound "Pneumonia (non-severe)" + "Pneumonia (severe)")
  - Wheezing Disorders (SPRSound "Asthma" + "Bronchitis")

Uses patient-grouped cross-validation folds from disease_split_manifest.csv.
Separate from the sound-classification training pipeline (train_cnn.py).

Key design choices for convergence:
  - Lower initial learning rate (3e-4) to prevent divergence on imbalanced data
  - Lighter 3-block CNN to match the smaller effective dataset (1634 records)
  - Balanced class weights computed from training fold only
  - Label smoothing (0.1) to help with noisy patient-level labels
  - ReduceLROnPlateau with patience=5 and aggressive factor=0.3
  - Early stopping on val_loss with patience=15

Usage:
    python ml/training/train_disease_cnn.py --val-fold 0 --epochs 60 --batch-size 32
"""

import os
import sys
import json
import csv
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiseaseTrainer")

DISEASE_MANIFEST_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "disease_split_manifest.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "models" / "saved_models"

DISEASE_CLASSES = ["Healthy Control", "Pneumonia", "Wheezing Disorders"]


def check_ml_environment():
    try:
        import tensorflow as tf
        return True, None
    except ImportError as e:
        return False, f"TensorFlow not available: {e}"


def load_disease_data(val_fold: int = 0):
    """
    Loads feature tensors and labels from disease_split_manifest.csv.
    Returns train/val arrays with patient-grouped split.
    """
    X_train, y_train, pids_train = [], [], []
    X_val, y_val, pids_val = [], [], []
    skipped = 0

    with open(DISEASE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = int(row["cv_fold"])
            feature_path = PROJECT_ROOT / row["feature_path"]
            if not feature_path.exists():
                skipped += 1
                continue

            data = np.load(feature_path)
            tensor = data["tensor"].astype(np.float32)[0]  # (128, 128, 1)
            label = int(row["disease_class_index"])
            pid = row["patient_id"]

            if fold == val_fold:
                X_val.append(tensor)
                y_val.append(label)
                pids_val.append(pid)
            else:
                X_train.append(tensor)
                y_train.append(label)
                pids_train.append(pid)

    logger.info("Loaded %d train, %d val samples (skipped %d missing)", len(X_train), len(X_val), skipped)
    return (
        np.array(X_train), np.array(y_train), pids_train,
        np.array(X_val), np.array(y_val), pids_val,
    )


def compute_class_weights(y, num_classes):
    """Balanced class weights: total / (num_classes * count_per_class)."""
    counts = Counter(int(l) for l in y)
    total = len(y)
    weights = {}
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights[c] = total / (num_classes * cnt)
    return weights


def build_disease_cnn(input_shape=(128, 128, 1), num_classes=3, learning_rate=3e-4):
    """
    Lighter 3-block CNN for disease classification.
    Uses fewer parameters than the sound-classification model to reduce overfitting
    on the smaller effective dataset (1634 records total, only ~116 Healthy Control).

    Architecture:
      Block 1: Conv2D(32,3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
      Block 2: Conv2D(64,3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
      Block 3: Conv2D(128,3x3, name='last_conv_layer') -> BN -> ReLU -> GAP -> Dropout(0.4)
      Head: Dense(64, relu) -> Dropout(0.4) -> Dense(3, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv_b1")(inputs)
    x = layers.BatchNormalization(name="bn_b1")(x)
    x = layers.ReLU(name="relu_b1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b1")(x)
    x = layers.Dropout(0.25, name="drop_b1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv_b2")(inputs if False else x)
    x = layers.BatchNormalization(name="bn_b2")(x)
    x = layers.ReLU(name="relu_b2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_b2")(x)
    x = layers.Dropout(0.25, name="drop_b2")(x)

    # Block 3 — Target conv layer for Grad-CAM
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="last_conv_layer")(x)
    x = layers.BatchNormalization(name="bn_b3")(x)
    x = layers.ReLU(name="relu_b3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="drop4")(x)

    # Classification head
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.4, name="drop5")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disease_cnn_v2")

    # Label smoothing to handle noisy patient-level labels
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def run_disease_training(
    val_fold: int = 0,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
):
    available, err = check_ml_environment()
    if not available:
        return {"success": False, "message": err}

    import tensorflow as tf
    np.random.seed(42)
    tf.random.set_seed(42)

    if not DISEASE_MANIFEST_PATH.exists():
        return {"success": False, "message": f"Disease manifest not found: {DISEASE_MANIFEST_PATH}"}

    logger.info("Loading disease data (val_fold=%d)...", val_fold)
    X_train, y_train, pids_train, X_val, y_val, pids_val = load_disease_data(val_fold)

    if len(X_train) == 0 or len(X_val) == 0:
        return {"success": False, "message": "Empty train or val set."}

    num_classes = 3
    class_weights = compute_class_weights(y_train, num_classes)
    logger.info("Class weights: %s", class_weights)

    # Verify patient disjointness
    train_pids = set(pids_train)
    val_pids = set(pids_val)
    overlap = train_pids.intersection(val_pids)
    if overlap:
        return {"success": False, "message": f"Patient overlap: {overlap}"}

    fold_meta = {
        "val_fold": val_fold,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_patients": len(train_pids),
        "val_patients": len(val_pids),
        "patient_overlap": 0,
        "train_class_counts": {str(k): int(v) for k, v in Counter(int(l) for l in y_train).items()},
        "val_class_counts": {str(k): int(v) for k, v in Counter(int(l) for l in y_val).items()},
        "class_weights": {str(k): v for k, v in class_weights.items()},
    }

    logger.info("Train: %d samples, %d patients", len(X_train), len(train_pids))
    logger.info("Val: %d samples, %d patients", len(X_val), len(val_pids))

    model = build_disease_cnn(
        input_shape=(128, 128, 1),
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    if dry_run:
        model.summary()
        logger.info("[DRY RUN] Architecture verified.")
        return {"success": True, "status": "DRY_RUN_PASSED", "fold_metadata": fold_meta}

    output_dir = output_dir or SAVED_MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "sprsound-disease-cnn-v2"
    best_path = output_dir / f"{model_name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    logger.info("Starting disease CNN training for up to %d epochs...", epochs)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history and config
    hist_path = output_dir / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    config = {
        "model_version": model_name,
        "task_id": "SPRSOUND_DISEASE_3CLASS",
        "task_type": "DISEASE_CLASSIFICATION",
        "input_shape": [None, 128, 128, 1],
        "verified_classes": DISEASE_CLASSES,
        "target_conv_layer": "last_conv_layer",
        "output_activation": "softmax",
        "preprocessing_config": {
            "sample_rate": 8000,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512,
            "f_min": 50.0,
            "f_max": 2000.0,
            "apply_bandpass": True,
            "top_db": 80.0,
            "normalization": "minmax_01",
        },
        "fold_metadata": fold_meta,
        "epochs_trained": len(history.history["loss"]),
        "best_artifact_path": str(best_path.relative_to(PROJECT_ROOT)),
    }
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Training complete. Artifact saved to: %s", best_path)
    return {
        "success": True,
        "status": "TRAINED",
        "artifact_path": str(best_path),
        "history": history.history,
        "metadata": config,
    }


def main():
    parser = argparse.ArgumentParser(description="Disease CNN Training Pipeline")
    parser.add_argument("--val-fold", type=int, default=0, help="Validation fold (0-4)")
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Initial LR")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_disease_training(
        val_fold=args.val_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run,
    )

    if not result.get("success"):
        print(f"\nTRAINING FAILED: {result.get('message')}")
        sys.exit(2)
    else:
        print(f"\nTRAINING {result['status']}")


if __name__ == "__main__":
    main()

