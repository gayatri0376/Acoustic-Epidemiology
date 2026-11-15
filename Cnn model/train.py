"""
End-to-End CNN Training Pipeline for Pediatric Respiratory & Throat Models.

Usage:
  # Train disease prediction model (Healthy Control, Pneumonia, Wheezing Disorders)
  python train.py --task disease --val-fold 0 --epochs 50 --batch-size 32

  # Train sound classification model (Normal, CAS, DAS, CAS & DAS)
  python train.py --task sound --val-fold 0 --epochs 40 --batch-size 32

  # Dry-run pipeline verification
  python train.py --task disease --dry-run
"""

import os
import sys
import json
import logging
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Ensure Cnn model and project root are importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .config import (
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        PREPROCESSING_CONFIG,
        BACKEND_SAVED_MODELS_DIR,
        LOCAL_SAVED_MODELS_DIR,
        TRAINING_CONFIG,
        TARGET_CONV_LAYER_NAME,
    )
    from .architecture import (
        build_throat_disease_cnn,
        build_throat_sound_cnn,
        verify_model_for_backend,
    )
    from .dataset_loader import (
        load_disease_dataset,
        load_sound_dataset,
        augment_spectrogram,
    )
except (ImportError, ValueError):
    from config import (
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        PREPROCESSING_CONFIG,
        BACKEND_SAVED_MODELS_DIR,
        LOCAL_SAVED_MODELS_DIR,
        TRAINING_CONFIG,
        TARGET_CONV_LAYER_NAME,
    )
    from architecture import (
        build_throat_disease_cnn,
        build_throat_sound_cnn,
        verify_model_for_backend,
    )
    from dataset_loader import (
        load_disease_dataset,
        load_sound_dataset,
        augment_spectrogram,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CNNTrainPipeline")


def train_model(
    task: str = "disease",
    val_fold: int = 0,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    dry_run: bool = False,
    augment: bool = True,
    output_dir: Path = LOCAL_SAVED_MODELS_DIR,
) -> Dict[str, Any]:
    """
    Executes full training, validation evaluation, artifact serialization,
    and backend verification.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            ReduceLROnPlateau,
            CSVLogger,
        )
    except ImportError as e:
        logger.error(
            "TensorFlow is not installed in the current environment.\n"
            "Please install: pip install tensorflow keras\n%s", e
        )
        sys.exit(1)

    logger.info("=================================================================")
    logger.info("Starting CNN Training Pipeline: Task = %s, Val Fold = %d", task.upper(), val_fold)
    logger.info("TensorFlow Version: %s", tf.__version__)
    logger.info("=================================================================")

    # 1. Load Dataset
    if task == "disease":
        classes = DISEASE_CLASSES
        X_train, y_train, pids_train, X_val, y_val, pids_val, class_weights = load_disease_dataset(val_fold=val_fold)
        model_version_name = f"sprsound-disease-cnn-v1-fold{val_fold}"
        architecture_desc = "3Block-ThroatDisease-2DCNN"
        register_cmd_type = "register_disease_model"
    else:
        classes = SOUND_CLASSES_4CLASS
        X_train, y_train, pids_train, X_val, y_val, pids_val, class_weights = load_sound_dataset(val_fold=val_fold)
        model_version_name = f"sprsound-sound-cnn-v1-fold{val_fold}"
        architecture_desc = "4Block-ThroatSound-2DCNN"
        register_cmd_type = "register_model"

    num_classes = len(classes)
    logger.info("Target classes (%d): %s", num_classes, classes)
    logger.info("Train shape: %s | Val shape: %s", X_train.shape, X_val.shape)

    # 2. Build Model
    if task == "disease":
        model = build_throat_disease_cnn(
            input_shape=(128, 128, 1),
            num_classes=num_classes,
            learning_rate=learning_rate
        )
    else:
        model = build_throat_sound_cnn(
            input_shape=(128, 128, 1),
            num_classes=num_classes,
            learning_rate=learning_rate
        )

    model.summary(print_fn=logger.info)

    # 3. Handle Dry Run
    if dry_run:
        logger.info("DRY RUN MODE: Testing forward pass and verification without full training.")
        verify_result = verify_model_for_backend(model, expected_classes=classes)
        logger.info("Model verification passed: %s", verify_result)
        return {"status": "DRY_RUN_PASSED", "verification": verify_result}

    # 4. Data Augmentation
    if augment:
        logger.info("Generating augmented training samples...")
        augmented_X, augmented_y = [], []
        for x_samp, y_samp in zip(X_train, y_train):
            augmented_X.append(x_samp)
            augmented_y.append(y_samp)
            # Add one augmented variation per sample
            augmented_X.append(augment_spectrogram(x_samp))
            augmented_y.append(y_samp)
        X_train_final = np.array(augmented_X, dtype=np.float32)
        y_train_final = np.array(augmented_y, dtype=np.int32)
        logger.info("Augmentation complete: expanded from %d to %d samples.", len(X_train), len(X_train_final))
    else:
        X_train_final = X_train
        y_train_final = y_train

    # 5. Callbacks
    local_artifact_path = output_dir / f"{model_version_name}.keras"
    backend_artifact_path = BACKEND_SAVED_MODELS_DIR / f"{model_version_name}.keras"
    log_csv_path = output_dir / f"{model_version_name}_history.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(local_artifact_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=TRAINING_CONFIG[task]["patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=TRAINING_CONFIG[task]["reduce_lr_factor"],
            patience=TRAINING_CONFIG[task]["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(str(log_csv_path))
    ]

    # 6. Model Training
    logger.info("Commencing training for %d epochs...", epochs)
    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Post-Training Validation Evaluation
    logger.info("Evaluating model on validation fold (%d)...", val_fold)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    raw_preds = model.predict(X_val, verbose=0)
    pred_indices = np.argmax(raw_preds, axis=1)

    # Confusion matrix & per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(
        y_val, pred_indices,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    conf_mat = confusion_matrix(y_val, pred_indices).tolist()

    logger.info("Validation Accuracy: %.4f | Validation Loss: %.4f", val_acc, val_loss)
    logger.info("Classification Report:\n%s", classification_report(y_val, pred_indices, target_names=classes, zero_division=0))

    # 8. Verify Model Compatibility with Backend
    verify_info = verify_model_for_backend(model, expected_classes=classes)
    logger.info("Backend compatibility verified: %s", verify_info)

    # 9. Sync Model Artifact to Backend Directory
    shutil.copy2(local_artifact_path, backend_artifact_path)
    logger.info("Model artifact copied to backend storage: %s", backend_artifact_path)

    # 10. Save Evaluation Metadata
    rel_artifact_path = f"ml/models/saved_models/{model_version_name}.keras"
    metadata = {
        "model_version": model_version_name,
        "task": task,
        "task_type": "DISEASE_CLASSIFICATION" if task == "disease" else "SOUND_CLASSIFICATION",
        "architecture": architecture_desc,
        "artifact_path": rel_artifact_path,
        "verified_classes": classes,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "val_fold": val_fold,
        "validation_metrics": {
            "val_loss": round(float(val_loss), 4),
            "val_accuracy": round(float(val_acc), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "confusion_matrix": conf_mat,
            "per_class_metrics": {
                c: {
                    "precision": round(float(report[c]["precision"]), 4),
                    "recall": round(float(report[c]["recall"]), 4),
                    "f1_score": round(float(report[c]["f1-score"]), 4),
                    "support": int(report[c]["support"])
                }
                for c in classes
            }
        }
    }

    eval_json_path = output_dir / f"{model_version_name}_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Evaluation metadata saved to: %s", eval_json_path)

    # 11. Print Backend Registration Command
    classes_str = " ".join([f'"{c}"' for c in classes])
    logger.info("=================================================================")
    logger.info("MODEL READY FOR BACKEND REGISTRATION!")
    logger.info("Execute this Django command to activate the model:")
    logger.info(
        "python backend/manage.py %s \\\n"
        "  --model-version %s \\\n"
        "  --artifact-path %s \\\n"
        "  --classes %s \\\n"
        "  --status ACTIVE \\\n"
        "  --architecture %s \\\n"
        "  --target-conv-layer %s \\\n"
        "  --evaluation-json %s \\\n"
        "  --set-active --overwrite",
        register_cmd_type,
        model_version_name,
        rel_artifact_path,
        classes_str,
        architecture_desc,
        TARGET_CONV_LAYER_NAME,
        str(eval_json_path).replace("\\", "/")
    )
    logger.info("=================================================================")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Pediatric Throat & Respiratory Predictions")
    parser.add_argument("--task", choices=["disease", "sound"], default="disease", help="Target task: 'disease' or 'sound'")
    parser.add_argument("--val-fold", type=int, default=0, help="Validation fold (0 to 4)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-augment", action="store_true", help="Disable spectrogram augmentation")
    parser.add_argument("--dry-run", action="store_true", help="Verify pipeline and forward pass without full training")

    args = parser.parse_args()

    train_model(
        task=args.task,
        val_fold=args.val_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment=not args.no_augment,
        dry_run=args.dry_run,
    )
