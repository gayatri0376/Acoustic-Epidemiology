#!/usr/bin/env python3
"""
Honest evaluation of the SPRSound disease-classification model on the validation fold.
Uses fold 0 as the held-out validation set (the fold used during training).

Reports:
- Per-class precision, recall, F1
- Macro-F1, balanced accuracy
- Confusion matrix
- Patient-level majority-vote accuracy
- Per-class sample counts

IMPORTANT: This uses the VALIDATION fold (fold 0), NOT a fully held-out independent
test set. The disease_split_manifest.csv does not include a separate test split.
Results should be interpreted as internal cross-validation performance, not
independent generalization performance.
"""
import sys
import json
import csv
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "saved_models" / "sprsound-disease-cnn-v1.keras"
MANIFEST_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "disease_split_manifest.csv"
CONFIG_PATH = PROJECT_ROOT / "ml" / "models" / "saved_models" / "sprsound-disease-cnn-v1_config.json"
OUTPUT_PATH = PROJECT_ROOT / "ml" / "evaluation" / "reports" / "disease_model_validation_fold0_evaluation.json"

DISEASE_CLASSES = ["Healthy Control", "Pneumonia", "Wheezing Disorders"]
VAL_FOLD = 0


def load_validation_data():
    """Loads feature tensors and labels for validation fold 0 from the manifest."""
    X_val, y_val, patient_ids, record_ids = [], [], [], []
    skipped = 0

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["cv_fold"]) != VAL_FOLD:
                continue

            feature_path = PROJECT_ROOT / row["feature_path"]
            if not feature_path.exists():
                skipped += 1
                continue

            data = np.load(feature_path)
            tensor = data["tensor"].astype(np.float32)
            label = int(row["disease_class_index"])

            X_val.append(tensor[0])  # shape (128, 128, 1)
            y_val.append(label)
            patient_ids.append(row["patient_id"])
            record_ids.append(row["record_id"])

    print(f"  Loaded {len(X_val)} validation samples (skipped {skipped} missing files)")
    return np.array(X_val), np.array(y_val), patient_ids, record_ids


def compute_metrics(y_true, y_pred, class_names):
    """Computes per-class precision, recall, F1, macro-F1, balanced accuracy."""
    n = len(class_names)
    results = {}
    per_class = {}

    for i, cls in enumerate(class_names):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        support = int(np.sum(y_true == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_f1 = float(np.mean([per_class[c]["f1"] for c in class_names]))
    macro_precision = float(np.mean([per_class[c]["precision"] for c in class_names]))
    macro_recall = float(np.mean([per_class[c]["recall"] for c in class_names]))

    class_recalls = [per_class[c]["recall"] for c in class_names]
    balanced_accuracy = float(np.mean(class_recalls))

    overall_accuracy = float(np.mean(y_true == y_pred))

    # Confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }


def patient_majority_vote(y_pred, patient_ids, y_true):
    """Computes patient-level majority-vote accuracy."""
    patient_preds = defaultdict(list)
    patient_labels = {}

    for pred, pid, true_label in zip(y_pred, patient_ids, y_true):
        patient_preds[pid].append(pred)
        patient_labels[pid] = true_label  # all recordings for a patient share same label

    correct = 0
    total = len(patient_preds)
    for pid, preds in patient_preds.items():
        # majority vote
        counts = np.bincount(np.array(preds), minlength=3)
        majority = int(np.argmax(counts))
        if majority == patient_labels[pid]:
            correct += 1

    return {
        "total_patients": total,
        "correct_patients": correct,
        "patient_accuracy": round(correct / total, 4) if total > 0 else 0.0,
    }


def run_evaluation():
    print("\n=== Disease Model Evaluation — Validation Fold 0 ===")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Manifest: {MANIFEST_PATH}")

    print("\nLoading validation data...")
    X_val, y_val, patient_ids, record_ids = load_validation_data()

    if len(X_val) == 0:
        print("ERROR: No validation samples found. Cannot evaluate.")
        return None

    # Class distribution
    unique, counts = np.unique(y_val, return_counts=True)
    print("\nValidation set class distribution:")
    for idx, cnt in zip(unique, counts):
        print(f"  {DISEASE_CLASSES[idx]}: {cnt} samples")

    print("\nLoading model...")
    try:
        import tensorflow as tf
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print(f"  Model loaded: input={model.input_shape}, output={model.output_shape}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    print(f"\nRunning inference on {len(X_val)} samples...")
    raw_outputs = model.predict(X_val, verbose=1, batch_size=32)
    y_pred = np.argmax(raw_outputs, axis=1)

    print("\nComputing metrics...")
    metrics = compute_metrics(y_val, y_pred, DISEASE_CLASSES)
    patient_metrics = patient_majority_vote(y_pred, patient_ids, y_val)

    # Load training config
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    report = {
        "evaluation_protocol": {
            "status": "VALIDATION_FOLD_ONLY",
            "note": (
                "This evaluation uses validation fold 0 — the fold held out during training. "
                "The disease_split_manifest.csv does not include a separate independent test set. "
                "Results represent internal cross-validation performance, NOT independent generalization. "
                "An independent test set evaluation requires a separate patient-disjoint test split "
                "not present in the current dataset preparation."
            ),
            "fold": VAL_FOLD,
            "total_val_samples": len(X_val),
            "unique_val_patients": len(set(patient_ids)),
        },
        "dataset": {
            "source": "SPRSound (SJTU Paediatric Respiratory Sound Database)",
            "license": "CC BY 4.0",
            "population": "Pediatric patients (0–16.2 years)",
            "label_provenance": "Patient-level hospital admission/discharge diagnoses",
            "label_mapping": {
                "0": "Healthy Control",
                "1": "Pneumonia",
                "2": "Wheezing Disorders",
            },
            "label_grouping_rationale": {
                "Pneumonia": "Merges 'Pneumonia (non-severe)' and 'Pneumonia (severe)' from SPRSound summaries.",
                "Wheezing Disorders": (
                    "Merges 'Asthma' and 'Bronchitis' because both produce wheezing (CAS) and "
                    "the dataset does not provide acoustic evidence to distinguish them. "
                    "Do NOT separate these as asthma vs bronchitis predictions."
                ),
                "Healthy Control": "Maps to 'Control Group' patients in SPRSound summaries.",
            },
        },
        "model_config": config,
        "recording_level_results": metrics,
        "patient_level_results": patient_metrics,
        "class_sample_counts": {
            DISEASE_CLASSES[idx]: int(cnt)
            for idx, cnt in zip(unique, counts)
        },
        "limitations": [
            "Evaluation uses validation fold 0 only (not an independent held-out test set).",
            "Disease labels are patient-level; individual auscultation sites may show "
            "acoustic patterns inconsistent with the systemic diagnosis.",
            "Training data is heavily imbalanced (Pneumonia ~77% of records).",
            "Class weights were applied during training but Healthy Control remains very sparse (83 train, 33 val).",
            "Model trained for only 7 epochs; validation loss was increasing, "
            "suggesting the model did not fully converge on all classes.",
            "Pediatric population only (0–16.2 years). Do not apply to adult recordings.",
            "Wheezing Disorders merges Asthma and Bronchitis — cannot be separated acoustically from SPRSound labels.",
            "Results are a research screening output and must NOT be used for clinical diagnosis.",
        ],
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Validation Fold 0)")
    print("="*60)
    print(f"\nOverall Accuracy  : {metrics['overall_accuracy']:.4f}")
    print(f"Macro F1-Score    : {metrics['macro_f1']:.4f}")
    print(f"Macro Precision   : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall      : {metrics['macro_recall']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"\nPatient-level majority-vote accuracy: "
          f"{patient_metrics['correct_patients']}/{patient_metrics['total_patients']} "
          f"= {patient_metrics['patient_accuracy']:.4f}")

    print("\nPer-class performance:")
    for cls, r in metrics["per_class"].items():
        print(f"  {cls:22s} | support={r['support']:4d} | "
              f"P={r['precision']:.4f} | R={r['recall']:.4f} | F1={r['f1']:.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"  {'':22s}  " + "  ".join(f"{c[:8]:>8}" for c in DISEASE_CLASSES))
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {DISEASE_CLASSES[i]:22s}  " + "  ".join(f"{v:8d}" for v in row))

    # Save report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run_evaluation()
