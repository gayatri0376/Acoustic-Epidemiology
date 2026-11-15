"""
Patient-Grouped Dataset Loaders for Pediatric Breathing Sounds.

Guarantees:
1. Strict Patient Disjointness: Zero data leakage between train and val folds.
2. Balanced Class Weighting: Mitigates severe pediatric clinical imbalance.
3. SpecAugment Spectrogram Augmentation for robust training.
"""

import os
import csv
import logging
from collections import Counter
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

try:
    from .config import (
        PROJECT_ROOT,
        DISEASE_MANIFEST_PATH,
        DATASET_SPLIT_MANIFEST_PATH,
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
    )
except (ImportError, ValueError):
    from config import (
        PROJECT_ROOT,
        DISEASE_MANIFEST_PATH,
        DATASET_SPLIT_MANIFEST_PATH,
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
    )

logger = logging.getLogger("DatasetLoader")


def load_disease_dataset(
    val_fold: int = 0,
    manifest_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Dict[int, float]]:
    """
    Loads feature tensors from disease_split_manifest.csv.
    Strictly groups recordings by patient_id across folds.

    Returns:
      (X_train, y_train, pids_train, X_val, y_val, pids_val, class_weights)
    """
    path = manifest_path or DISEASE_MANIFEST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Disease split manifest not found at: {path}")

    X_train, y_train, pids_train = [], [], []
    X_val, y_val, pids_val = [], [], []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = int(row["cv_fold"])
            rel_feature_path = row["feature_path"].replace("\\", "/")
            full_feature_path = PROJECT_ROOT / rel_feature_path

            if not full_feature_path.exists():
                skipped += 1
                continue

            try:
                data = np.load(full_feature_path)
                tensor = data["tensor"].astype(np.float32)
                # Ensure shape is (128, 128, 1)
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                elif tensor.ndim == 2:
                    tensor = np.expand_dims(tensor, axis=-1)

                label = int(row["disease_class_index"])
                pid = str(row["patient_id"])

                if fold == val_fold:
                    X_val.append(tensor)
                    y_val.append(label)
                    pids_val.append(pid)
                else:
                    X_train.append(tensor)
                    y_train.append(label)
                    pids_train.append(pid)
            except Exception as e:
                logger.warning("Error reading %s: %s", full_feature_path, e)
                skipped += 1

    # Verify strict patient disjointness
    train_patients = set(pids_train)
    val_patients = set(pids_val)
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise ValueError(
            f"PATIENT LEAKAGE DETECTED! {len(overlap)} patients overlap between train and val: {overlap}"
        )

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    X_val_arr = np.array(X_val, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.int32)

    # Compute balanced class weights: total / (num_classes * count_per_class)
    counts = Counter(y_train)
    total = len(y_train)
    num_classes = len(DISEASE_CLASSES)
    class_weights = {}
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        class_weights[c] = float(total / (num_classes * cnt))

    logger.info(
        "Disease dataset loaded (val_fold=%d): %d train (%d patients), %d val (%d patients). Skipped: %d",
        val_fold, len(X_train), len(train_patients), len(X_val), len(val_patients), skipped
    )
    logger.info("Training class distribution: %s", dict(counts))
    logger.info("Balanced class weights: %s", class_weights)

    return (
        X_train_arr, y_train_arr, pids_train,
        X_val_arr, y_val_arr, pids_val,
        class_weights
    )


def load_sound_dataset(
    val_fold: int = 0,
    manifest_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Dict[int, float]]:
    """
    Loads feature tensors from dataset_split_manifest.csv (4-class sound classification).
    Excludes Poor Quality records per verified project policy.
    """
    path = manifest_path or DATASET_SPLIT_MANIFEST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sound split manifest not found at: {path}")

    class_to_idx = {name: idx for idx, name in enumerate(SOUND_CLASSES_4CLASS)}

    X_train, y_train, pids_train = [], [], []
    X_val, y_val, pids_val = [], [], []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split_category") != "train_cv":
                continue

            annotation = row["record_annotation"].strip()
            if annotation == "Poor Quality" or annotation not in class_to_idx:
                continue

            fold = int(row["cv_fold"])
            rec_id = row["record_id"].strip()
            feature_rel = f"ml/features/extracted/records/train_cv_{rec_id}_record.npz"
            full_feature_path = PROJECT_ROOT / feature_rel

            if not full_feature_path.exists():
                skipped += 1
                continue

            try:
                data = np.load(full_feature_path)
                tensor = data["tensor"].astype(np.float32)
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                elif tensor.ndim == 2:
                    tensor = np.expand_dims(tensor, axis=-1)

                label = class_to_idx[annotation]
                pid = str(row["patient_id"])

                if fold == val_fold:
                    X_val.append(tensor)
                    y_val.append(label)
                    pids_val.append(pid)
                else:
                    X_train.append(tensor)
                    y_train.append(label)
                    pids_train.append(pid)
            except Exception as e:
                skipped += 1

    train_patients = set(pids_train)
    val_patients = set(pids_val)
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise ValueError(f"Patient leakage detected in sound dataset: {len(overlap)} overlapping patients.")

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    X_val_arr = np.array(X_val, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.int32)

    counts = Counter(y_train)
    total = len(y_train)
    num_classes = len(SOUND_CLASSES_4CLASS)
    class_weights = {}
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        class_weights[c] = float(total / (num_classes * cnt))

    logger.info(
        "Sound dataset loaded (val_fold=%d): %d train (%d patients), %d val (%d patients). Skipped: %d",
        val_fold, len(X_train), len(train_patients), len(X_val), len(val_patients), skipped
    )

    return (
        X_train_arr, y_train_arr, pids_train,
        X_val_arr, y_val_arr, pids_val,
        class_weights
    )


def augment_spectrogram(
    tensor: np.ndarray,
    freq_mask_max: int = 12,
    time_mask_max: int = 12,
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Applies SpecAugment-style time masking, frequency masking, and subtle noise.
    Prevents CNN overfitting on small acoustic sets without distorting pitch.
    """
    aug = tensor.copy()
    h, w, _ = aug.shape

    # Frequency masking
    f_len = np.random.randint(0, freq_mask_max)
    f_start = np.random.randint(0, max(1, h - f_len))
    aug[f_start:f_start + f_len, :, :] = 0.0

    # Time masking
    t_len = np.random.randint(0, time_mask_max)
    t_start = np.random.randint(0, max(1, w - t_len))
    aug[:, t_start:t_start + t_len, :] = 0.0

    # Subtle Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, aug.shape).astype(np.float32)
        aug = np.clip(aug + noise, 0.0, 1.0)

    return aug
