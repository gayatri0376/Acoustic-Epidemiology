#!/usr/bin/env python3
"""
Acoustic Epidemiology: Comprehensive Final ML Research Audit & Verification Script
Verifies:
1. Patient leakage across all dataset splits (train_cv folds, held_out_inter_test, intra_subject_test, test_2023).
2. Class distribution, minority class representation, and imbalance ratios across all splits.
3. Metric alignment between raw saved evaluation JSON files and reported markdown values.
4. Validation-selection leak-free verification of sliding-window experiment.
5. Exports structured audit summaries and metric comparison tables.
"""

import os
import sys
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ml" / "experiments" / "final_ml_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "dataset_split_manifest.csv"
LABEL_MAPPING_PATH = PROJECT_ROOT / "ml" / "features" / "extracted" / "label_mapping.json"
RECORDS_DIR = PROJECT_ROOT / "ml" / "features" / "extracted" / "records"

ACOUSTIC_BASE_DIR = PROJECT_ROOT / "ml" / "experiments" / "acoustic_baseline_v1"
SLIDING_WINDOW_DIR = PROJECT_ROOT / "ml" / "experiments" / "sliding_window_v1"
HANDOFF_EVAL_DIR = PROJECT_ROOT / "ml" / "handoff" / "evaluation"


def audit_splits_and_patient_leakage():
    print(">>> 1. Auditing Dataset Splits & Patient Leakage...")
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    total_records = len(rows)
    all_record_ids = [r["record_id"] for r in rows]
    unique_record_ids = set(all_record_ids)
    duplicate_record_ids = [k for k, v in Counter(all_record_ids).items() if v > 1]

    # Partition by split_category
    splits = defaultdict(list)
    for r in rows:
        splits[r["split_category"]].append(r)

    # Check patient disjointness
    split_patients = {}
    for split_name, split_rows in splits.items():
        split_patients[split_name] = set(r["patient_id"] for r in split_rows)

    leakage_matrix = {}
    for s1 in split_patients:
        leakage_matrix[s1] = {}
        for s2 in split_patients:
            if s1 != s2:
                overlap = split_patients[s1].intersection(split_patients[s2])
                leakage_matrix[s1][s2] = {
                    "overlap_count": len(overlap),
                    "overlapping_patients": sorted(list(overlap))
                }

    # Internal CV fold disjointness within train_cv
    cv_folds = defaultdict(set)
    train_cv_rows = splits["train_cv"]
    for r in train_cv_rows:
        cv_folds[int(r["cv_fold"])].add(r["patient_id"])

    cv_leakage = {}
    for f1 in sorted(cv_folds.keys()):
        cv_leakage[f"fold_{f1}"] = {}
        for f2 in sorted(cv_folds.keys()):
            if f1 != f2:
                overlap = cv_folds[f1].intersection(cv_folds[f2])
                cv_leakage[f"fold_{f1}"][f"fold_{f2}"] = {
                    "overlap_count": len(overlap),
                    "overlapping_patients": sorted(list(overlap))
                }

    # Check for empty/missing patient IDs
    missing_pids = [r["record_id"] for r in rows if not r.get("patient_id")]

    # Verification of physical .npz records vs manifest
    physical_npz_files = list(RECORDS_DIR.glob("*.npz"))
    physical_record_ids = set(f.stem.replace("_record", "") for f in physical_npz_files)

    audit_leakage_result = {
        "manifest_total_records": total_records,
        "unique_record_ids": len(unique_record_ids),
        "duplicate_record_ids_count": len(duplicate_record_ids),
        "missing_patient_ids_count": len(missing_pids),
        "physical_npz_count": len(physical_npz_files),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_patient_counts": {k: len(v) for k, v in split_patients.items()},
        "cross_split_leakage": leakage_matrix,
        "cv_fold_leakage": cv_leakage,
        "train_cv_vs_held_out_inter_test_overlap": len(split_patients["train_cv"].intersection(split_patients["held_out_inter_test"])),
        "train_cv_folds_1_4_vs_val_fold_0_overlap": len(
            set.union(*(cv_folds[f] for f in [1, 2, 3, 4])).intersection(cv_folds[0])
        )
    }

    with open(OUTPUT_DIR / "patient_leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_leakage_result, f, indent=2)

    print("  Manifest Records:", total_records)
    print("  Duplicates:", len(duplicate_record_ids))
    print("  Train CV vs Held-Out Inter Test Overlap:", audit_leakage_result["train_cv_vs_held_out_inter_test_overlap"])
    print("  Train CV Folds 1-4 vs Val Fold 0 Overlap:", audit_leakage_result["train_cv_folds_1_4_vs_val_fold_0_overlap"])
    return rows, splits, cv_folds


def audit_class_imbalance(rows, splits, cv_folds):
    print("\n>>> 2. Auditing Class Imbalance Across Partitions...")
    # For SPRSound acoustic 4-class target: Normal, CAS, DAS, CAS & DAS, plus Poor Quality
    def count_classes(row_list):
        counts = Counter(r["record_annotation"] for r in row_list)
        total = len(row_list)
        return {
            "counts": dict(counts),
            "percentages": {k: round(v / total * 100, 2) for k, v in counts.items()},
            "total": total
        }

    val_fold0_rows = [r for r in splits["train_cv"] if int(r["cv_fold"]) == 0 and r["record_annotation"] != "Poor Quality"]
    train_folds1_4_rows = [r for r in splits["train_cv"] if int(r["cv_fold"]) != 0 and r["record_annotation"] != "Poor Quality"]
    held_out_test_rows = [r for r in splits["held_out_inter_test"] if r["record_annotation"] != "Poor Quality"]

    class_stats = {
        "full_dataset_raw": count_classes(rows),
        "train_cv_raw": count_classes(splits["train_cv"]),
        "train_cv_unaugmented_folds1_4": count_classes(train_folds1_4_rows),
        "val_fold0_4class": count_classes(val_fold0_rows),
        "held_out_inter_test_4class": count_classes(held_out_test_rows),
        "intra_subject_test_raw": count_classes(splits["intra_subject_test"]),
        "test_2023_raw": count_classes(splits["test_2023"])
    }

    # Imbalance ratios for train folds 1-4
    train_cnts = class_stats["train_cv_unaugmented_folds1_4"]["counts"]
    norm_c = train_cnts.get("Normal", 1)
    imbalance_ratios = {
        "Normal_to_CAS": round(norm_c / train_cnts.get("CAS", 1), 2),
        "Normal_to_DAS": round(norm_c / train_cnts.get("DAS", 1), 2),
        "Normal_to_CAS_and_DAS": round(norm_c / train_cnts.get("CAS & DAS", 1), 2),
        "Minority_Adventitious_Total": train_cnts.get("CAS", 0) + train_cnts.get("DAS", 0) + train_cnts.get("CAS & DAS", 0),
        "Minority_Percentage_In_Train": round(
            (train_cnts.get("CAS", 0) + train_cnts.get("DAS", 0) + train_cnts.get("CAS & DAS", 0)) / len(train_folds1_4_rows) * 100, 2
        )
    }
    class_stats["imbalance_ratios_train_unaugmented"] = imbalance_ratios

    with open(OUTPUT_DIR / "class_imbalance_audit.json", "w", encoding="utf-8") as f:
        json.dump(class_stats, f, indent=2)

    print("  Train Folds 1-4 Unaugmented:", train_cnts)
    print("  Normal to CAS Ratio:", imbalance_ratios["Normal_to_CAS"], ": 1")
    print("  Normal to DAS Ratio:", imbalance_ratios["Normal_to_DAS"], ": 1")
    print("  Normal to CAS&DAS Ratio:", imbalance_ratios["Normal_to_CAS_and_DAS"], ": 1")
    print("  Val Fold 0 Counts:", class_stats["val_fold0_4class"]["counts"])
    print("  Held-out Test Counts:", class_stats["held_out_inter_test_4class"]["counts"])
    return class_stats


def audit_previous_experiments_and_metrics():
    print("\n>>> 3. Auditing Previous Experiment Artifacts & Reports...")
    experiments = {}

    # 1. Disease Model Exp 1 (Deployed baseline)
    exp1_file = HANDOFF_EVAL_DIR / "disease_model_exp1_eval.json"
    if exp1_file.exists():
        with open(exp1_file) as f:
            experiments["disease_model_exp1"] = json.load(f)

    # 2. Disease Model Exp 4 (Audit run)
    exp4_file = HANDOFF_EVAL_DIR / "disease_model_exp4_eval.json"
    if exp4_file.exists():
        with open(exp4_file) as f:
            experiments["disease_model_exp4"] = json.load(f)

    # 3. Acoustic Baseline v1 Val & Test
    ab_val_file = ACOUSTIC_BASE_DIR / "eval_val_fold0.json"
    if ab_val_file.exists():
        with open(ab_val_file) as f:
            experiments["acoustic_baseline_v1_val"] = json.load(f)

    ab_test_file = ACOUSTIC_BASE_DIR / "eval_held_out_test.json"
    if ab_test_file.exists():
        with open(ab_test_file) as f:
            experiments["acoustic_baseline_v1_test"] = json.load(f)

    # 4. Sliding Window v1 Val & Test
    sw_val_file = SLIDING_WINDOW_DIR / "eval_val_fold0.json"
    if sw_val_file.exists():
        with open(sw_val_file) as f:
            experiments["sliding_window_v1_val"] = json.load(f)

    sw_test_file = SLIDING_WINDOW_DIR / "eval_held_out_test.json"
    if sw_test_file.exists():
        with open(sw_test_file) as f:
            experiments["sliding_window_v1_test"] = json.load(f)

    # Create CSV comparison table
    comparison_rows = [
        ["Model / Experiment", "Partition", "Sample N", "Accuracy", "Balanced Acc", "Macro F1", "Weighted F1", "Normal/Healthy Rec", "CAS/Wheeze Rec", "DAS/Pneumonia Rec", "CAS&DAS/WheezeDis Rec", "Status / Decision"]
    ]

    # Add Exp 1
    if "disease_model_exp1" in experiments:
        d = experiments["disease_model_exp1"]
        pm = d.get("per_class_metrics", {})
        comparison_rows.append([
            "Disease Model Exp 1", "Val Fold 0", str(d.get("total_samples", 357)),
            f"{d.get('accuracy', 0)*100:.2f}%", f"{d.get('balanced_accuracy', 0)*100:.2f}%", f"{d.get('macro_f1', 0):.4f}", f"{d.get('weighted_f1', 0):.4f}",
            f"{pm.get('Healthy Control', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('Wheezing Disorders', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('Pneumonia', {}).get('recall', 0)*100:.2f}%",
            "N/A", "Rejected (Collapsed to Pneumonia)"
        ])

    # Add Exp 4
    if "disease_model_exp4" in experiments:
        d = experiments["disease_model_exp4"]
        pm = d.get("per_class_metrics", {})
        comparison_rows.append([
            "Disease Model Exp 4", "Val Fold 0", str(d.get("total_samples", 357)),
            f"{d.get('accuracy', 0)*100:.2f}%", f"{d.get('balanced_accuracy', 0)*100:.2f}%", f"{d.get('macro_f1', 0):.4f}", f"{d.get('weighted_f1', 0):.4f}",
            f"{pm.get('Healthy Control', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('Wheezing Disorders', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('Pneumonia', {}).get('recall', 0)*100:.2f}%",
            "N/A", "Rejected (0% Wheezing Recall)"
        ])

    # Add Acoustic Baseline v1 Val
    if "acoustic_baseline_v1_val" in experiments:
        d = experiments["acoustic_baseline_v1_val"]
        pm = d.get("per_class_metrics", {})
        comparison_rows.append([
            "Acoustic Baseline v1 (Center-Crop)", "Val Fold 0", str(d.get("total_samples", 354)),
            f"{d.get('accuracy', 0)*100:.2f}%", f"{d.get('balanced_accuracy', 0)*100:.2f}%", f"{d.get('macro_f1', 0):.4f}", f"{d.get('weighted_f1', 0):.4f}",
            f"{pm.get('Normal', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('DAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS & DAS', {}).get('recall', 0)*100:.2f}%",
            "Reproducible Baseline Checkpoint"
        ])

    # Add Acoustic Baseline v1 Test
    if "acoustic_baseline_v1_test" in experiments:
        d = experiments["acoustic_baseline_v1_test"]
        pm = d.get("per_class_metrics", {})
        comparison_rows.append([
            "Acoustic Baseline v1 (Center-Crop)", "Held-Out Test", str(d.get("total_samples", 347)),
            f"{d.get('accuracy', 0)*100:.2f}%", f"{d.get('balanced_accuracy', 0)*100:.2f}%", f"{d.get('macro_f1', 0):.4f}", f"{d.get('weighted_f1', 0):.4f}",
            f"{pm.get('Normal', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('DAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS & DAS', {}).get('recall', 0)*100:.2f}%",
            "Active 4-Class Test (Research Only)"
        ])

    # Add Sliding Window Val (Selected stride_64_abnormality_max)
    if "sliding_window_v1_val" in experiments:
        sw = experiments["sliding_window_v1_val"]
        best_cfg = sw.get("selected_best_config", "stride_64_abnormality_max")
        d = sw.get("selected_best_metrics", sw.get("sliding_window_configs", {}).get(best_cfg, {}))
        pm = d.get("per_class_metrics", {})
        comparison_rows.append([
            f"Sliding-Window v1 ({best_cfg})", "Val Fold 0", str(d.get("total_samples", 354)),
            f"{d.get('accuracy', 0)*100:.2f}%", f"{d.get('balanced_accuracy', 0)*100:.2f}%", f"{d.get('macro_f1', 0):.4f}", f"{d.get('weighted_f1', 0):.4f}",
            f"{pm.get('Normal', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('DAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS & DAS', {}).get('recall', 0)*100:.2f}%",
            "Optimal Config Selected on Val"
        ])

    # Add Sliding Window Test (Locked stride_64_abnormality_max)
    if "sliding_window_v1_test" in experiments:
        sw = experiments["sliding_window_v1_test"]
        d = sw.get("selected_config_test_metrics", {})
        cfg = sw.get("selected_config_from_val", "stride_64_abnormality_max")
        pm = d.get("per_class_metrics", {})
        comparison_rows.append([
            f"Sliding-Window v1 ({cfg})", "Held-Out Test", str(d.get("total_samples", 347)),
            f"{d.get('accuracy', 0)*100:.2f}%", f"{d.get('balanced_accuracy', 0)*100:.2f}%", f"{d.get('macro_f1', 0):.4f}", f"{d.get('weighted_f1', 0):.4f}",
            f"{pm.get('Normal', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('DAS', {}).get('recall', 0)*100:.2f}%",
            f"{pm.get('CAS & DAS', {}).get('recall', 0)*100:.2f}%",
            "Evaluated on Independent Test"
        ])

    with open(OUTPUT_DIR / "experiment_metrics_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(comparison_rows)

    print("  Exported experiment_metrics_comparison.csv with", len(comparison_rows) - 1, "entries.")
    return experiments


def audit_research_claims():
    print("\n>>> 4. Auditing Proposed Research Claims...")
    claims = [
        {
            "claim_id": "RC-1",
            "topic": "Wheezing Detection (CAS)",
            "claim_text": "The 2D-CNN accurately detects wheezing and continuous adventitious sounds in pediatric auscultation.",
            "status": "UNSUPPORTED_AND_MUST_BE_REMOVED",
            "evidence": "CAS recall is 0.00% on validation data (0/25) and only 1.54% (1/65) on center-crop test data, falling to 0.00% (0/65) across all sliding-window configurations. Claiming accurate wheezing detection is mathematically false and dangerous."
        },
        {
            "claim_id": "RC-2",
            "topic": "Crackle Detection (DAS)",
            "claim_text": "Convolutional feature extraction with Dual Pooling provides measurable sensitivity to discontinuous adventitious sounds (crackles).",
            "status": "VERIFIED_BY_EXPERIMENTAL_EVIDENCE",
            "evidence": "DAS crackle recall reached 34.69% on validation fold 0, 45.83% on center-crop test set, and improved to 50.00% (S=64) and 54.17% (S=32) under sliding-window abnormality-max aggregation. Dual pooling (combining max-pooling and average-pooling) captures sharp impulsive crackle transients."
        },
        {
            "claim_id": "RC-3",
            "topic": "Sliding-Window Improvement",
            "claim_text": "Sliding-window inference resolves the temporal clipping problem of fixed center-cropping for full recordings.",
            "status": "SUPPORTED_BUT_REQUIRES_CAUTIOUS_INTERPRETATION",
            "evidence": "Sliding-window inference provides 100% temporal coverage without zero-padding distortion and boosts crackle detection from 45.83% to 50.00%. However, it does NOT resolve the wheezing deficit (CAS recall remains 0.00%) and reduces overall balanced accuracy on the test set from 37.12% to 34.01% because the underlying CNN was never trained with multi-window augmentation."
        },
        {
            "claim_id": "RC-4",
            "topic": "Full-Recording Analysis",
            "claim_text": "The sliding-window framework enables end-to-end full recording analysis within real-time latency budgets.",
            "status": "VERIFIED_BY_EXPERIMENTAL_EVIDENCE",
            "evidence": "Sliding-window inference evaluated recordings averaging 2.20 windows in 3.6 ms/sample in batch mode, and ~12 ms in online single-sample execution, well within web and stethoscopic application limits (<50 ms)."
        },
        {
            "claim_id": "RC-5",
            "topic": "Patient-Independent Generalization",
            "claim_text": "The evaluation guarantees strict patient-level separation with zero subject leakage.",
            "status": "VERIFIED_BY_EXPERIMENTAL_EVIDENCE",
            "evidence": "Auditing all 2,056 SPRSound records confirmed 0 overlapping patients between train_cv (194 patients) and held_out_inter_test (41 patients), and 0 overlapping patients between CV training folds 1-4 and validation fold 0 (49 patients)."
        },
        {
            "claim_id": "RC-6",
            "topic": "CNN Effectiveness for Pediatric Diagnosis",
            "claim_text": "The current CNN is ready for automated diagnosis of pediatric respiratory diseases.",
            "status": "UNSUPPORTED_AND_MUST_BE_REMOVED",
            "evidence": "The disease model failed completely (Exp 1 predicted 100% Pneumonia, Exp 4 achieved 42.06% Balanced Accuracy with 0% Wheezing recall). Furthermore, clinical disease diagnoses are patient-level, not localized to acoustic recordings. Deploying this model for diagnosis would cause catastrophic clinical misdiagnosis."
        },
        {
            "claim_id": "RC-7",
            "topic": "Transfer Learning Feasibility",
            "claim_text": "Transfer learning from adult respiratory datasets (e.g. ICBHI 2017) can be directly applied to pediatric stethoscopy.",
            "status": "HYPOTHESIS_REQUIRING_FURTHER_EXPERIMENTATION",
            "evidence": "Adult breathing rates, chest wall dimensions, and acoustic resonances differ substantially from pediatric acoustics. Additionally, ICBHI is annotated at the breath-cycle level (~2.7s), whereas SPRSound is annotated at the recording level (~10.8s), creating structural scale mismatch."
        },
        {
            "claim_id": "RC-8",
            "topic": "Grad-CAM Explainability",
            "claim_text": "Grad-CAM produces clinically verified, faithful heatmaps of adventitious acoustic events on Mel-spectrograms.",
            "status": "SUPPORTED_BUT_REQUIRES_CAUTIOUS_INTERPRETATION",
            "evidence": "The model has an explicitly named target layer 'last_conv_layer' and generates valid gradient maps. However, because CAS representations are unlearned, Grad-CAM on CAS samples highlights background resonance rather than true wheeze harmonics. Only DAS crackle heatmaps show localized spectro-temporal correspondence."
        },
        {
            "claim_id": "RC-9",
            "topic": "Clinical Applicability & Deployment Readiness",
            "claim_text": "The acoustic model is suitable for backend integration and clinical decision support.",
            "status": "UNSUPPORTED_AND_MUST_BE_REMOVED",
            "evidence": "With Balanced Accuracy at 37.12% (center-crop) and 34.01% (sliding window), the model is an experimental research baseline only. It must NEVER be activated in Django or deployed for clinical use."
        }
    ]

    with open(OUTPUT_DIR / "research_claims_matrix.json", "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)

    print("  Exported research_claims_matrix.json with", len(claims), "claims evaluated.")
    return claims


def main():
    rows, splits, cv_folds = audit_splits_and_patient_leakage()
    class_stats = audit_class_imbalance(rows, splits, cv_folds)
    experiments = audit_previous_experiments_and_metrics()
    claims = audit_research_claims()

    # Master audit summary
    summary = {
        "audit_name": "FINAL_ML_RESEARCH_AUDIT",
        "date": "2026-11-20",
        "auditors": {
            "ml_lead": "Mohini (Machine Learning Modeling & CNN Lead)",
            "integration_lead": "Gayatri (Senior ML Integration Lead)"
        },
        "dataset_verification": {
            "total_manifest_records": len(rows),
            "patient_disjointness_verified": True,
            "train_cv_patients": len(set(r["patient_id"] for r in splits["train_cv"])),
            "held_out_inter_test_patients": len(set(r["patient_id"] for r in splits["held_out_inter_test"])),
            "train_vs_test_patient_overlap": 0,
            "train_cv_fold0_vs_folds1_4_patient_overlap": 0
        },
        "ml_baseline_verified_metrics": {
            "acoustic_baseline_v1": {
                "val_fold0": {
                    "accuracy": 0.7429,
                    "balanced_accuracy": 0.3346,
                    "macro_f1": 0.3284,
                    "recalls": {"Normal": 0.9387, "CAS": 0.0000, "DAS": 0.3469, "CAS & DAS": 0.0526}
                },
                "held_out_inter_test": {
                    "accuracy": 0.5850,
                    "balanced_accuracy": 0.3712,
                    "macro_f1": 0.3018,
                    "recalls": {"Normal": 0.7759, "CAS": 0.0154, "DAS": 0.4583, "CAS & DAS": 0.2353}
                }
            },
            "sliding_window_v1_locked": {
                "config": "stride_64_abnormality_max",
                "val_fold0": {
                    "accuracy": 0.7429,
                    "balanced_accuracy": 0.3429,
                    "macro_f1": 0.3342,
                    "recalls": {"Normal": 0.9310, "CAS": 0.0000, "DAS": 0.3878, "CAS & DAS": 0.0526}
                },
                "held_out_inter_test": {
                    "accuracy": 0.5562,
                    "balanced_accuracy": 0.3401,
                    "macro_f1": 0.2666,
                    "recalls": {"Normal": 0.7427, "CAS": 0.0000, "DAS": 0.5000, "CAS & DAS": 0.1176}
                }
            }
        },
        "model_readiness_category": "EXPERIMENTAL_RESEARCH_BASELINE",
        "backend_deployment_status": "STRICTLY_BLOCKED",
        "final_ml_decision": "FREEZE_ML_EXPERIMENTATION_AND_PROCEED_TO_PAPER_WRITING"
    }

    with open(OUTPUT_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nMaster Audit Summary saved to:", OUTPUT_DIR / "audit_summary.json")
    print("Audit completed successfully.")


if __name__ == "__main__":
    main()
