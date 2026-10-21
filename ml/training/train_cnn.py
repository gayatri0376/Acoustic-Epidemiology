#!/usr/bin/env python3
"""
Acoustic Epidemiology: CNN Training Pipeline
Automated Classification of Pediatric Respiratory Diseases Using CNNs and Time-Frequency Feature Extraction

Trains a genuine 2D Convolutional Neural Network on Log Mel-Spectrogram features extracted from
the SPRSound pediatric respiratory dataset.

Key Principles:
1. Strict Patient Disjointness: Folds are grouped by patient_id using StratifiedGroupKFold.
2. Verified Ground Truth: Uses deterministic acoustic record-level mapping (Normal, CAS, DAS, CAS&DAS).
3. Fair Evaluation: Training weights computed exclusively from training data; independent test partitions
   (2022 inter-subject test, 2023 test) are NEVER touched during training or hyperparameter selection.
4. Grad-CAM Compatible: Final convolutional layer is explicitly named for gradient explainability.
5. No Dummy Artifacts: Fails cleanly with transparent environment requirements if ML framework is unavailable.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CNNTrainer")

LABEL_MAPPING_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "label_mapping.json"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "dataset_split_manifest.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "models" / "saved_models"


def check_ml_environment() -> Tuple[bool, Optional[str]]:
    """
    Checks if TensorFlow/Keras is available in the current environment.
    Returns (is_available, error_message).
    """
    try:
        import tensorflow as tf
        import keras
        return True, None
    except ImportError as e:
        msg = (
            "TensorFlow/Keras is not installed in the active environment:\n"
            f"  Interpreter: {sys.executable}\n"
            f"  Error: {e}\n\n"
            "To install the recommended ML dependencies for this Apple Silicon environment:\n"
            "  pip install tensorflow keras h5py\n"
        )
        return False, msg


def load_label_mapping(task_id: str = "SPRSOUND_RECORD_4CLASS") -> Dict[str, Any]:
    """Loads deterministic label configuration from ml/features/extracted/label_mapping.json."""
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found at {LABEL_MAPPING_PATH}. "
            "Run feature extraction to generate deterministic label mappings first."
        )
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if task_id == "SPRSOUND_RECORD_4CLASS":
        return data["primary_target"]
    elif task_id == "SPRSOUND_RECORD_5CLASS_WITH_QUALITY":
        return data["quality_detection_target"]
    elif task_id == "SPRSOUND_RECORD_TERNARY":
        return data["ternary_target"]
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def build_respiratory_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 4,
    learning_rate: float = 1e-3
):
    """
    Builds a 4-block 2D Convolutional Neural Network tailored for Time-Frequency Log Mel-Spectrograms.
    Architecture:
      ConvBlock 1: Conv2D(32) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.2)
      ConvBlock 2: Conv2D(64) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.2)
      ConvBlock 3: Conv2D(128) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.3)
      ConvBlock 4: Conv2D(256, name='last_conv_layer') -> BatchNorm -> ReLU -> GlobalAvgPool -> Dropout(0.4)
      Classifier: Dense(128, relu) -> Dropout(0.4) -> Dense(num_classes, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")

    # Block 1: Low-level acoustic feature extraction
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv2d_block1")(inputs)
    x = layers.BatchNormalization(name="bn_block1")(x)
    x = layers.ReLU(name="relu_block1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_block1")(x)
    x = layers.Dropout(0.2, name="dropout_block1")(x)

    # Block 2: Intermediate spectro-temporal patterns
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv2d_block2")(x)
    x = layers.BatchNormalization(name="bn_block2")(x)
    x = layers.ReLU(name="relu_block2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_block2")(x)
    x = layers.Dropout(0.2, name="dropout_block2")(x)

    # Block 3: Adventitious acoustic structures (crackles / wheezes)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv2d_block3")(x)
    x = layers.BatchNormalization(name="bn_block3")(x)
    x = layers.ReLU(name="relu_block3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_block3")(x)
    x = layers.Dropout(0.3, name="dropout_block3")(x)

    # Block 4: High-level semantic representation (Target Conv Layer for Grad-CAM)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="last_conv_layer")(x)
    x = layers.BatchNormalization(name="bn_block4")(x)
    x = layers.ReLU(name="relu_block4")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.4, name="dropout_block4")(x)

    # Classification Head
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PediatricAcousticCNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_dataset_for_fold(
    features_dir: Path,
    val_fold: int = 0,
    task_id: str = "SPRSOUND_RECORD_4CLASS"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads extracted .npz record tensors and labels partitioned by patient-grouped CV fold.
    Returns (X_train, y_train, X_val, y_val, metadata).
    """
    import csv
    label_cfg = load_label_mapping(task_id)
    class_to_idx = label_cfg["class_to_index"]

    records_dir = features_dir / "records"
    if not records_dir.exists():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")

    split_manifest = features_dir / "dataset_split_manifest.csv"
    if not split_manifest.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {split_manifest}. Run feature extraction with --generate-splits first."
        )

    # Read manifest entries for train_cv
    train_meta = {}
    with open(split_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split_category") == "train_cv":
                train_meta[row["record_id"]] = {
                    "cv_fold": int(row.get("cv_fold", -1)),
                    "patient_id": row["patient_id"],
                    "record_annotation": row["record_annotation"]
                }

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    train_patients, val_patients = set(), set()

    for npz_file in sorted(records_dir.glob("train_cv_*_record.npz")):
        data = np.load(npz_file)
        stem = str(data["record_id"])
        raw_label = str(data["label"])
        pid = str(data["patient_id"])

        # Quality handling: if primary 4-class, exclude Poor Quality
        if task_id == "SPRSOUND_RECORD_4CLASS" and raw_label == "Poor Quality":
            continue

        if raw_label not in class_to_idx:
            continue

        label_idx = class_to_idx[raw_label]
        tensor = data["tensor"]  # shape (1, 128, 128, 1)
        # Squeeze batch dimension for array accumulation: (128, 128, 1)
        tensor_sample = np.squeeze(tensor, axis=0)

        meta = train_meta.get(stem, {})
        fold = meta.get("cv_fold", -1)

        if fold == val_fold:
            X_val_list.append(tensor_sample)
            y_val_list.append(label_idx)
            val_patients.add(pid)
        else:
            X_train_list.append(tensor_sample)
            y_train_list.append(label_idx)
            train_patients.add(pid)

    # Disjointness check
    leakage = train_patients.intersection(val_patients)
    if leakage:
        raise ValueError(f"CRITICAL: Patient leakage detected between train and val fold {val_fold}: {leakage}")

    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.empty((0, 128, 128, 1))
    y_train = np.array(y_train_list, dtype=np.int32) if y_train_list else np.empty((0,))
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.empty((0, 128, 128, 1))
    y_val = np.array(y_val_list, dtype=np.int32) if y_val_list else np.empty((0,))

    metadata = {
        "val_fold": val_fold,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "train_class_counts": {str(k): int(v) for k, v in Counter(y_train).items()},
        "val_class_counts": {str(k): int(v) for k, v in Counter(y_val).items()},
        "task_id": task_id,
        "classes": label_cfg["index_to_class"]
    }
    return X_train, y_train, X_val, y_val, metadata


def compute_balanced_class_weights(y_train: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Computes balanced inverse-frequency class weights based strictly on training labels."""
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_train)
    if len(unique_classes) == 0:
        return {i: 1.0 for i in range(num_classes)}

    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    for c in range(num_classes):
        if c not in weight_dict:
            weight_dict[c] = 1.0
    return weight_dict


def run_training_pipeline(
    task_id: str = "SPRSOUND_RECORD_4CLASS",
    val_fold: int = 0,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_dir: Optional[Path] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Executes training pipeline or dry run.
    Guarantees no dummy weights or fake metrics are generated.
    """
    is_available, err_msg = check_ml_environment()
    if not is_available:
        logger.error("ML Environment Audit Failed:\n%s", err_msg)
        return {
            "success": False,
            "status": "BLOCKED_DEPENDENCY",
            "message": err_msg
        }

    import tensorflow as tf

    label_cfg = load_label_mapping(task_id)
    num_classes = label_cfg["num_classes"]
    verified_classes = [label_cfg["index_to_class"][str(i)] for i in range(num_classes)]

    logger.info("Task: %s (%d classes: %s)", task_id, num_classes, verified_classes)

    features_dir = PROJECT_ROOT / "ml" / "features" / "extracted"
    X_train, y_train, X_val, y_val, fold_meta = load_dataset_for_fold(
        features_dir=features_dir,
        val_fold=val_fold,
        task_id=task_id
    )

    logger.info("Fold metadata: %s", fold_meta)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            f"Insufficient extracted feature tensors in {features_dir}. "
            "Run feature extraction for the training partition first: "
            "'python ml/features/extract_features.py --mode train'"
        )

    # Class weights from training fold only
    class_weights = compute_balanced_class_weights(y_train, num_classes)
    logger.info("Balanced class weights: %s", class_weights)

    # Build model
    model = build_respiratory_cnn(
        input_shape=(128, 128, 1),
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    if dry_run:
        logger.info("[DRY RUN] Architecture and fold partitioning verified successfully.")
        return {
            "success": True,
            "status": "DRY_RUN_PASSED",
            "model_summary": model.summary(print_fn=logger.info),
            "fold_metadata": fold_meta,
            "class_weights": class_weights
        }

    output_dir = output_dir or SAVED_MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_version_name = f"sprsound-cnn-v1-fold{val_fold}"
    best_artifact_path = output_dir / f"{model_version_name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_artifact_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info("Starting genuine CNN training for %d epochs...", epochs)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history and configuration
    history_path = output_dir / f"{model_version_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    config_path = output_dir / f"{model_version_name}_config.json"
    meta_record = {
        "model_version": model_version_name,
        "task_id": task_id,
        "input_shape": [None, 128, 128, 1],
        "verified_classes": verified_classes,
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
            "normalization": "minmax_01"
        },
        "fold_metadata": fold_meta,
        "epochs_trained": len(history.history["loss"]),
        "best_artifact_path": str(best_artifact_path.relative_to(PROJECT_ROOT))
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2)

    logger.info("Genuine training complete. Artifact saved to: %s", best_artifact_path)
    return {
        "success": True,
        "status": "TRAINED",
        "artifact_path": str(best_artifact_path),
        "history": history.history,
        "metadata": meta_record
    }


def main():
    parser = argparse.ArgumentParser(description="Acoustic Epidemiology CNN Training Pipeline")
    parser.add_argument(
        "--task",
        choices=["SPRSOUND_RECORD_4CLASS", "SPRSOUND_RECORD_5CLASS_WITH_QUALITY", "SPRSOUND_RECORD_TERNARY"],
        default="SPRSOUND_RECORD_4CLASS",
        help="Target classification task"
    )
    parser.add_argument("--val-fold", type=int, default=0, help="Validation fold (0 to 4)")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading, fold integrity, and architecture without training")
    args = parser.parse_args()

    result = run_training_pipeline(
        task_id=args.task,
        val_fold=args.val_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run
    )

    if not result.get("success", False):
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE HALTED (EXPECTED ENVIRONMENT REQUIREMENT)")
        print("=" * 70)
        print(result.get("message", "Training halted."))
        sys.exit(2)
    else:
        print("\n" + "=" * 70)
        print(f"TRAINING PIPELINE SUCCESS: {result['status']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
