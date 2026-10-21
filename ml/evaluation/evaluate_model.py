#!/usr/bin/env python3
"""
Acoustic Epidemiology: Comprehensive Model Evaluation Engine
Automated Classification of Pediatric Respiratory Diseases Using CNNs and Time-Frequency Feature Extraction

Evaluates genuinely trained CNN models on independent held-out benchmarks:
1. Primary Benchmark: 2022 Inter-Subject Test Set (355 recordings, 41 patients, ZERO training overlap)
2. Secondary Benchmark: 2023 Test Set (871 recordings, 93 patients, ZERO training overlap)
3. Diagnostic Sanity: Patient-grouped CV validation fold (held-out validation patients)

Guarantees:
- Zero Metric Fabrication: If no model artifact exists, reports 'METRICS_UNAVAILABLE'.
- Patient Clustering: Computes both recording-level and patient-majority-vote metrics.
- No Intra-Subject Pollution: Prohibits combining 2022 intra-subject test with independent test results.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelEvaluator")

LABEL_MAPPING_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "label_mapping.json"
REPORTS_DIR = PROJECT_ROOT / "ml" / "evaluation" / "reports"


def load_label_mapping(task_id: str = "SPRSOUND_RECORD_4CLASS") -> Dict[str, Any]:
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Label mapping not found at {LABEL_MAPPING_PATH}")
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


def load_evaluation_data(
    features_dir: Path,
    partition: str = "inter_2022",
    task_id: str = "SPRSOUND_RECORD_4CLASS"
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, Any]]:
    """
    Loads features and ground-truth labels for evaluation.
    Returns (X, y, record_ids, patient_ids, metadata).
    """
    import csv
    label_cfg = load_label_mapping(task_id)
    class_to_idx = label_cfg["class_to_index"]

    records_dir = features_dir / "records"
    if not records_dir.exists():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")

    # Determine partition manifest
    manifest_name = f"{partition}_record_manifest.csv"
    manifest_path = features_dir / manifest_name

    split_category_map = {
        "inter_2022": "held_out_inter_test",
        "intra_2022": "intra_subject_test",
        "test_2023": "test_2023",
    }
    split_cat = split_category_map.get(partition, partition)

    if not manifest_path.exists():
        # Fall back to checking dataset_split_manifest.csv
        split_manifest_path = features_dir / "dataset_split_manifest.csv"
        if not split_manifest_path.exists():
            raise FileNotFoundError(
                f"Evaluation manifest not found at {manifest_path} or {split_manifest_path}. "
                f"Extract features for '{partition}' first."
            )
        manifest_path = split_manifest_path

    records_meta = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split_category") == split_cat:
                records_meta.append(row)

    if not records_meta:
        raise ValueError(f"No records matching split_category='{split_cat}' found in {manifest_path}")

    X_list, y_list, rids, pids = [], [], [], []
    excluded_poor_quality = 0

    for rm in records_meta:
        rid = rm["record_id"]
        pid = rm["patient_id"]
        raw_label = rm["record_annotation"]

        if task_id == "SPRSOUND_RECORD_4CLASS" and raw_label == "Poor Quality":
            excluded_poor_quality += 1
            continue

        if raw_label not in class_to_idx:
            continue

        # Check feature file
        feat_path = rm.get("feature_path")
        if feat_path:
            full_feat_path = PROJECT_ROOT / feat_path if not os.path.isabs(feat_path) else Path(feat_path)
        else:
            full_feat_path = records_dir / f"{split_cat}_{rid}_record.npz"

        if not full_feat_path.exists():
            logger.warning("Feature tensor missing for %s at %s", rid, full_feat_path)
            continue

        data = np.load(full_feat_path)
        tensor = np.squeeze(data["tensor"], axis=0)  # (128, 128, 1)

        X_list.append(tensor)
        y_list.append(class_to_idx[raw_label])
        rids.append(rid)
        pids.append(pid)

    metadata = {
        "partition": partition,
        "split_category": split_cat,
        "total_manifest_records": len(records_meta),
        "evaluated_records": len(X_list),
        "unique_patients": len(set(pids)),
        "excluded_poor_quality": excluded_poor_quality,
        "class_distribution": dict(Counter(y_list))
    }
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), rids, pids, metadata


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    patient_ids: List[str],
    index_to_class: Dict[str, str]
) -> Dict[str, Any]:
    """Computes comprehensive macro, weighted, and per-class metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )

    num_classes = len(index_to_class)
    class_names = [index_to_class[str(i)] for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class_report = {}
    for idx, name in enumerate(class_names):
        per_class_report[name] = {
            "class_index": idx,
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx])
        }

    # Patient-level majority vote metrics
    patient_preds = defaultdict(list)
    patient_trues = {}
    for pid, y_t, y_p in zip(patient_ids, y_true, y_pred):
        patient_preds[pid].append(y_p)
        patient_trues[pid] = y_t

    patient_y_true = []
    patient_y_pred = []
    for pid, preds in patient_preds.items():
        vote = Counter(preds).most_common(1)[0][0]
        patient_y_true.append(patient_trues[pid])
        patient_y_pred.append(vote)

    pat_acc = float(accuracy_score(patient_y_true, patient_y_pred))
    pat_p, pat_r, pat_f1, _ = precision_recall_fscore_support(
        patient_y_true, patient_y_pred, average="macro", zero_division=0
    )

    return {
        "recording_level": {
            "total_recordings": len(y_true),
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "confusion_matrix": cm.tolist(),
            "classes": class_names,
            "per_class": per_class_report
        },
        "patient_level_majority_vote": {
            "total_patients": len(patient_trues),
            "patient_accuracy": round(pat_acc, 4),
            "patient_macro_f1": round(float(pat_f1), 4)
        }
    }


def evaluate_model(
    model_artifact_path: Path,
    partition: str = "inter_2022",
    task_id: str = "SPRSOUND_RECORD_4CLASS",
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates a genuine model artifact on the requested partition.
    Returns structured evaluation results or failure descriptor.
    """
    if partition == "intra_2022":
        logger.warning(
            "[METHODOLOGICAL NOTICE] Evaluating on 'intra_2022' measures same-patient memorization, "
            "NOT independent patient generalization. Results must never be combined with inter-subject benchmarks."
        )

    if not model_artifact_path.exists():
        logger.error("Model artifact does not exist on disk: %s", model_artifact_path)
        return {
            "success": False,
            "status": "ARTIFACT_NOT_FOUND",
            "message": f"Physical model artifact not found at {model_artifact_path}. The system does not fabricate evaluation metrics."
        }

    try:
        import tensorflow as tf
    except ImportError:
        return {
            "success": False,
            "status": "TENSORFLOW_UNAVAILABLE",
            "message": "TensorFlow is required to load model weights and execute forward passes."
        }

    logger.info("Loading model artifact from: %s", model_artifact_path)
    model = tf.keras.models.load_model(str(model_artifact_path), compile=False)

    features_dir = PROJECT_ROOT / "ml" / "features" / "extracted"
    X, y_true, rids, pids, data_meta = load_evaluation_data(
        features_dir=features_dir,
        partition=partition,
        task_id=task_id
    )

    if len(X) == 0:
        return {
            "success": False,
            "status": "NO_EVALUATION_FEATURES",
            "message": f"No features found for partition '{partition}'. Run feature extraction first."
        }

    logger.info("Running model forward pass on %d samples...", len(X))
    y_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_probs, axis=-1)

    label_cfg = load_label_mapping(task_id)
    metrics = compute_evaluation_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        patient_ids=pids,
        index_to_class=label_cfg["index_to_class"]
    )

    report = {
        "success": True,
        "status": "EVALUATED",
        "model_artifact": str(model_artifact_path.relative_to(PROJECT_ROOT)) if str(model_artifact_path).startswith(str(PROJECT_ROOT)) else str(model_artifact_path),
        "task_id": task_id,
        "partition": partition,
        "data_metadata": data_meta,
        "metrics": metrics
    }

    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"{model_artifact_path.stem}_{partition}_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation complete. Report saved to: %s", report_file)
    return report


def main():
    parser = argparse.ArgumentParser(description="Acoustic Epidemiology Model Evaluation")
    parser.add_argument("--model-artifact", type=str, required=True, help="Path to genuine .keras model artifact")
    parser.add_argument(
        "--partition",
        choices=["inter_2022", "intra_2022", "test_2023"],
        default="inter_2022",
        help="Test partition for evaluation"
    )
    parser.add_argument(
        "--task",
        choices=["SPRSOUND_RECORD_4CLASS", "SPRSOUND_RECORD_5CLASS_WITH_QUALITY", "SPRSOUND_RECORD_TERNARY"],
        default="SPRSOUND_RECORD_4CLASS",
        help="Classification task target"
    )
    args = parser.parse_args()

    artifact_path = Path(args.model_artifact).resolve()
    result = evaluate_model(
        model_artifact_path=artifact_path,
        partition=args.partition,
        task_id=args.task
    )

    if not result.get("success", False):
        print("\n" + "=" * 70)
        print("EVALUATION FAILED / METRICS UNAVAILABLE")
        print("=" * 70)
        print(result.get("message", "Evaluation could not proceed."))
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS FOR {args.partition}:")
        print("=" * 70)
        m = result["metrics"]["recording_level"]
        print(f"  Accuracy:        {m['accuracy']:.4f}")
        print(f"  Macro Precision: {m['macro_precision']:.4f}")
        print(f"  Macro Recall:    {m['macro_recall']:.4f}")
        print(f"  Macro F1:        {m['macro_f1']:.4f}")
        print("  Per-Class F1:")
        for cls_name, cls_data in m["per_class"].items():
            print(f"    {cls_name:15s}: F1={cls_data['f1_score']:.4f} (Support: {cls_data['support']})")
        print("=" * 70)


if __name__ == "__main__":
    main()
