#!/usr/bin/env python3
"""
Acoustic Epidemiology: Sliding-Window Multi-Scale Inference Engine
Task: Full-Recording Respiratory Sound Analysis using Sliding-Window Inference
Baseline Model: ml/experiments/acoustic_baseline_v1/sprsound_sound_cnn_baseline_fold0.keras

Investigates whether fixed center-cropping causes adventitious respiratory sounds
(wheezes/CAS, crackles/DAS) to be clipped and missed, and evaluates multi-window
aggregation across full recordings without retraining.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple
import numpy as np

# Set environment variables for headless execution and temporary cache
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["KERAS_HOME"] = "/tmp/.keras"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_MODEL_PATH = PROJECT_ROOT / "ml" / "experiments" / "acoustic_baseline_v1" / "sprsound_sound_cnn_baseline_fold0.keras"
RECORDS_DIR = PROJECT_ROOT / "ml" / "features" / "extracted" / "records"
MANIFEST_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "dataset_split_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "experiments" / "sliding_window_v1"

CLASSES = ["Normal", "CAS", "DAS", "CAS & DAS"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def extract_windows(spectrogram: np.ndarray, window_size: int = 128, stride: int = 64) -> List[np.ndarray]:
    """
    Extracts sliding windows along the temporal axis (axis 1) of a (128, T) log mel-spectrogram.
    Guarantees full temporal coverage from frame 0 to frame T-1.
    """
    num_freqs, T = spectrogram.shape
    if num_freqs != 128:
        raise ValueError(f"Expected 128 mel frequency bins, got {num_freqs}")

    if T < window_size:
        # Symmetrically pad if shorter than window_size
        pad_left = (window_size - T) // 2
        pad_right = window_size - T - pad_left
        padded = np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode="constant")
        return [padded[..., np.newaxis]]
    elif T == window_size:
        return [spectrogram[..., np.newaxis]]
    else:
        starts = list(range(0, T - window_size + 1, stride))
        if starts[-1] + window_size < T:
            starts.append(T - window_size)
        windows = []
        for s in starts:
            w = spectrogram[:, s : s + window_size]
            windows.append(w[..., np.newaxis])
        return windows


def aggregate_probabilities(
    probs: np.ndarray,
    method: str = "mean",
    top_k: int = 2
) -> np.ndarray:
    """
    Aggregates window-level prediction probabilities of shape (num_windows, num_classes).
    Returns (num_classes,) aggregated probability vector.
    """
    K, num_classes = probs.shape
    if K == 1:
        return probs[0]

    if method == "mean":
        agg = np.mean(probs, axis=0)
        return agg / np.sum(agg)

    elif method == "max":
        # Max probability across windows per class
        max_p = np.max(probs, axis=0)
        return max_p / np.sum(max_p)

    elif method == "top_k":
        # Mean of top-k probabilities for each class
        k_val = min(top_k, K)
        top_k_means = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            top_vals = np.partition(probs[:, c], -k_val)[-k_val:]
            top_k_means[c] = np.mean(top_vals)
        return top_k_means / np.sum(top_k_means)

    elif method == "abnormality_max":
        # Max abnormal confidence: if max abnormal class exceeds Normal, select max abnormal
        # Normal is class 0, abnormal are 1, 2, 3
        max_abnormal_p = np.max(probs[:, 1:], axis=0)
        mean_normal = np.mean(probs[:, 0])
        agg = np.zeros(num_classes, dtype=np.float32)
        agg[0] = mean_normal
        agg[1:] = max_abnormal_p
        return agg / np.sum(agg)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes confusion matrix, per-class metrics, accuracy, balanced accuracy, and F1 scores."""
    num_classes = len(CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    recalls = []
    f1s = []
    supports = []

    for c_idx, c_name in enumerate(CLASSES):
        tp = cm[c_idx, c_idx]
        fp = np.sum(cm[:, c_idx]) - tp
        fn = np.sum(cm[c_idx, :]) - tp
        support = np.sum(cm[c_idx, :])
        supports.append(support)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        f1s.append(f1)

        per_class[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": int(support)
        }

    total_samples = int(len(y_true))
    accuracy = float(np.sum(np.diag(cm)) / total_samples) if total_samples > 0 else 0.0
    balanced_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum([f1s[i] * supports[i] for i in range(num_classes)]) / total_samples) if total_samples > 0 else 0.0

    return {
        "total_samples": total_samples,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class
    }


def load_dataset(partition: str, val_fold: int = 0):
    """
    Loads samples from RECORDS_DIR matching partition.
    Returns list of dicts: {"record_id", "patient_id", "label_idx", "spectrogram", "center_crop_tensor", "raw_duration"}
    """
    manifest_info = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest_info[row["record_id"]] = row

    samples = []
    if partition == "val_fold0":
        for npz_file in sorted(RECORDS_DIR.glob("train_cv_*_record.npz")):
            data = np.load(npz_file)
            rec_id = str(data["record_id"])
            raw_label = str(data["label"])
            pid = str(data["patient_id"])

            if raw_label not in CLASS_TO_IDX:
                continue

            meta = manifest_info.get(rec_id, {})
            if int(meta.get("cv_fold", -1)) != val_fold:
                continue

            samples.append({
                "record_id": rec_id,
                "patient_id": pid,
                "label_name": raw_label,
                "label_idx": CLASS_TO_IDX[raw_label],
                "spectrogram": data["spectrogram"],
                "center_crop_tensor": data["tensor"],
                "raw_duration": float(data.get("raw_duration_seconds", 0.0))
            })

    elif partition == "held_out_inter_test":
        for npz_file in sorted(RECORDS_DIR.glob("held_out_inter_test_*_record.npz")):
            data = np.load(npz_file)
            rec_id = str(data["record_id"])
            raw_label = str(data["label"])
            pid = str(data["patient_id"])

            if raw_label not in CLASS_TO_IDX:
                continue

            samples.append({
                "record_id": rec_id,
                "patient_id": pid,
                "label_name": raw_label,
                "label_idx": CLASS_TO_IDX[raw_label],
                "spectrogram": data["spectrogram"],
                "center_crop_tensor": data["tensor"],
                "raw_duration": float(data.get("raw_duration_seconds", 0.0))
            })

    else:
        raise ValueError(f"Unknown partition: {partition}")

    return samples


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading baseline model from:", BASELINE_MODEL_PATH)
    import tensorflow as tf
    model = tf.keras.models.load_model(BASELINE_MODEL_PATH, compile=False)
    print("Model loaded successfully. Input shape:", model.input_shape, "Output shape:", model.output_shape)

    # 1. Evaluate on Validation Fold 0
    print("\n==========================================")
    print("STEP 1: Evaluating on Validation Fold 0 (N=354)")
    print("==========================================")
    val_samples = load_dataset("val_fold0", val_fold=0)
    print(f"Loaded {len(val_samples)} validation samples.")

    # Measure Center-Crop inference
    t0 = time.time()
    val_cc_preds = []
    val_y_true = [s["label_idx"] for s in val_samples]
    val_cc_tensors = np.vstack([s["center_crop_tensor"] for s in val_samples])
    val_cc_probs = model.predict(val_cc_tensors, batch_size=32, verbose=0)
    val_cc_preds = np.argmax(val_cc_probs, axis=1)
    val_cc_time = time.time() - t0
    val_cc_metrics = compute_metrics(np.array(val_y_true), val_cc_preds)

    print(f"Center-Crop Validation Time: {val_cc_time:.2f}s ({val_cc_time*1000/len(val_samples):.1f} ms/sample)")
    print(f"  Accuracy: {val_cc_metrics['accuracy']:.4f}, Balanced Acc: {val_cc_metrics['balanced_accuracy']:.4f}, Macro F1: {val_cc_metrics['macro_f1']:.4f}")

    # Pre-extract all sliding windows (stride=64, W=128)
    strides = [64, 32]
    val_results_by_config = {}

    for stride in strides:
        print(f"\n--- Testing Stride = {stride} (overlap = {100*(128-stride)/128:.1f}%) on Val Fold 0 ---")
        window_counts = []
        all_windows_flat = []
        sample_window_slices = []

        curr_idx = 0
        for s in val_samples:
            wins = extract_windows(s["spectrogram"], window_size=128, stride=stride)
            window_counts.append(len(wins))
            all_windows_flat.extend(wins)
            sample_window_slices.append((curr_idx, curr_idx + len(wins)))
            curr_idx += len(wins)

        print(f"Total windows extracted for {len(val_samples)} samples: {len(all_windows_flat)} (avg {np.mean(window_counts):.2f} windows/sample)")

        # Run model inference in batches
        X_all_windows = np.array(all_windows_flat, dtype=np.float32)
        t_infer_start = time.time()
        all_window_probs = model.predict(X_all_windows, batch_size=64, verbose=0)
        infer_time = time.time() - t_infer_start
        print(f"Sliding-window inference time: {infer_time:.2f}s ({infer_time*1000/len(val_samples):.1f} ms/sample, {infer_time*1000/len(all_windows_flat):.1f} ms/window)")

        # Evaluate aggregation methods
        methods = ["mean", "max", "top_k", "abnormality_max"]
        for m in methods:
            agg_preds = []
            for start_idx, end_idx in sample_window_slices:
                sample_probs = all_window_probs[start_idx:end_idx]
                agg_p = aggregate_probabilities(sample_probs, method=m, top_k=2)
                agg_preds.append(np.argmax(agg_p))

            metrics = compute_metrics(np.array(val_y_true), np.array(agg_preds))
            cfg_name = f"stride_{stride}_{m}"
            metrics["window_count_stats"] = {
                "total_windows": len(all_windows_flat),
                "mean_windows_per_sample": float(np.mean(window_counts)),
                "min_windows": int(np.min(window_counts)),
                "max_windows": int(np.max(window_counts))
            }
            metrics["inference_time_sec"] = round(infer_time, 3)
            metrics["ms_per_sample"] = round(infer_time * 1000 / len(val_samples), 2)
            val_results_by_config[cfg_name] = metrics

            print(f"[{cfg_name}] Acc: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Recalls: Norm={metrics['per_class_metrics']['Normal']['recall']:.3f}, CAS={metrics['per_class_metrics']['CAS']['recall']:.3f}, DAS={metrics['per_class_metrics']['DAS']['recall']:.3f}, CAS&DAS={metrics['per_class_metrics']['CAS & DAS']['recall']:.3f}")

    # Select best validation method based strictly on Balanced Accuracy
    best_cfg = max(val_results_by_config.keys(), key=lambda k: val_results_by_config[k]["balanced_accuracy"])
    print(f"\n>>> BEST CONFIG ON VALIDATION: {best_cfg} with Balanced Acc = {val_results_by_config[best_cfg]['balanced_accuracy']:.4f} <<<")

    # Save validation evaluation json
    val_eval_out = {
        "baseline_center_crop": val_cc_metrics,
        "sliding_window_configs": val_results_by_config,
        "selected_best_config": best_cfg,
        "selected_best_metrics": val_results_by_config[best_cfg]
    }
    with open(OUTPUT_DIR / "eval_val_fold0.json", "w", encoding="utf-8") as f:
        json.dump(val_eval_out, f, indent=2)

    # Save CM CSV for best validation config
    best_cm_val = val_results_by_config[best_cfg]["confusion_matrix"]
    with open(OUTPUT_DIR / "cm_val_fold0.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CLASSES)
        for i, row in enumerate(best_cm_val):
            writer.writerow([CLASSES[i]] + row)

    # 2. Evaluate on Independent Held-Out Test Set
    print("\n==========================================")
    print("STEP 2: Evaluating on Held-Out Test Set (N=347)")
    print("==========================================")
    test_samples = load_dataset("held_out_inter_test")
    print(f"Loaded {len(test_samples)} held-out test samples.")

    # Measure Center-Crop inference on Test
    t0 = time.time()
    test_y_true = [s["label_idx"] for s in test_samples]
    test_cc_tensors = np.vstack([s["center_crop_tensor"] for s in test_samples])
    test_cc_probs = model.predict(test_cc_tensors, batch_size=32, verbose=0)
    test_cc_preds = np.argmax(test_cc_probs, axis=1)
    test_cc_time = time.time() - t0
    test_cc_metrics = compute_metrics(np.array(test_y_true), test_cc_preds)

    print(f"Center-Crop Test Time: {test_cc_time:.2f}s ({test_cc_time*1000/len(test_samples):.1f} ms/sample)")
    print(f"  Accuracy: {test_cc_metrics['accuracy']:.4f}, Balanced Acc: {test_cc_metrics['balanced_accuracy']:.4f}, Macro F1: {test_cc_metrics['macro_f1']:.4f}")

    # Evaluate all configurations on Test for scientific completeness, but highlight the locked best validation config
    test_results_by_config = {}
    for stride in strides:
        test_window_counts = []
        test_all_windows_flat = []
        test_sample_slices = []

        curr_idx = 0
        for s in test_samples:
            wins = extract_windows(s["spectrogram"], window_size=128, stride=stride)
            test_window_counts.append(len(wins))
            test_all_windows_flat.extend(wins)
            test_sample_slices.append((curr_idx, curr_idx + len(wins)))
            curr_idx += len(wins)

        X_test_windows = np.array(test_all_windows_flat, dtype=np.float32)
        t_infer_start = time.time()
        test_window_probs = model.predict(X_test_windows, batch_size=64, verbose=0)
        infer_time = time.time() - t_infer_start

        for m in methods:
            agg_preds = []
            for start_idx, end_idx in test_sample_slices:
                sample_probs = test_window_probs[start_idx:end_idx]
                agg_p = aggregate_probabilities(sample_probs, method=m, top_k=2)
                agg_preds.append(np.argmax(agg_p))

            metrics = compute_metrics(np.array(test_y_true), np.array(agg_preds))
            cfg_name = f"stride_{stride}_{m}"
            metrics["window_count_stats"] = {
                "total_windows": len(test_all_windows_flat),
                "mean_windows_per_sample": float(np.mean(test_window_counts)),
                "min_windows": int(np.min(test_window_counts)),
                "max_windows": int(np.max(test_window_counts))
            }
            metrics["inference_time_sec"] = round(infer_time, 3)
            metrics["ms_per_sample"] = round(infer_time * 1000 / len(test_samples), 2)
            test_results_by_config[cfg_name] = metrics

            print(f"[{cfg_name}] Acc: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Recalls: Norm={metrics['per_class_metrics']['Normal']['recall']:.3f}, CAS={metrics['per_class_metrics']['CAS']['recall']:.3f}, DAS={metrics['per_class_metrics']['DAS']['recall']:.3f}, CAS&DAS={metrics['per_class_metrics']['CAS & DAS']['recall']:.3f}")

    # Save test evaluation json
    test_eval_out = {
        "baseline_center_crop": test_cc_metrics,
        "selected_config_from_val": best_cfg,
        "selected_config_test_metrics": test_results_by_config[best_cfg],
        "all_test_configs": test_results_by_config
    }
    with open(OUTPUT_DIR / "eval_held_out_test.json", "w", encoding="utf-8") as f:
        json.dump(test_eval_out, f, indent=2)

    # Save CM CSV for locked validation config on test set
    best_cm_test = test_results_by_config[best_cfg]["confusion_matrix"]
    with open(OUTPUT_DIR / "cm_held_out_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CLASSES)
        for i, row in enumerate(best_cm_test):
            writer.writerow([CLASSES[i]] + row)

    # Save config.json
    parts = best_cfg.split("_")
    best_stride = int(parts[1])
    best_method = "_".join(parts[2:])
    config_data = {
        "experiment_id": "SLIDING_WINDOW_INFERENCE_V1",
        "task": "FULL_RECORDING_RESPIRATORY_SOUND_ANALYSIS",
        "baseline_model_artifact": "ml/experiments/acoustic_baseline_v1/sprsound_sound_cnn_baseline_fold0.keras",
        "input_spectrogram_shape": [128, "T"],
        "window_parameters": {
            "window_size_frames": 128,
            "window_duration_seconds": 8.192,
            "stride_frames": best_stride,
            "stride_seconds": best_stride * 0.064,
            "overlap_percentage": float((128 - best_stride) / 128 * 100)
        },
        "selected_aggregation_strategy": best_method,
        "aggregation_selection_criterion": "Validation Fold 0 Balanced Accuracy",
        "classes": CLASSES,
        "benchmarks": {
            "val_fold0_samples": len(val_samples),
            "held_out_test_samples": len(test_samples)
        }
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print("\nExperiment completed successfully. Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    run_experiment()
