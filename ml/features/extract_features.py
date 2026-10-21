#!/usr/bin/env python3
"""
Acoustic Epidemiology: Reproducible Feature Extraction & Patient-Grouped Split Engine
Automated Classification of Pediatric Respiratory Diseases Using CNNs and Time-Frequency Feature Extraction

Extracts standardized Log Mel-Spectrogram tensors from the SPRSound dataset according to
the verified preprocessing contract and generates patient-grouped, zero-leakage cross-validation splits.

Usage:
    # Run smoke test on a small verified subset (records and events)
    python ml/features/extract_features.py --mode smoke_test

    # Generate patient-grouped fold assignments and verify disjointness
    python ml/features/extract_features.py --mode generate_splits

    # Run feature extraction on the smoke-tested subset
    python ml/features/extract_features.py --mode smoke_test --generate-splits
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.respiratory_preprocessor import (
    PreprocessingConfig,
    RespiratoryAudioPreprocessor
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureExtractor")

DATASET_ROOT = PROJECT_ROOT / "ml" / "dataset" / "SPRSound"
FEATURES_ROOT = PROJECT_ROOT / "ml" / "features"


class FeatureExtractionPipeline:
    """
    Orchestrates extraction of 4D Log Mel-Spectrogram tensors for whole recordings
    and segmented respiratory cycles, with strict patient grouping and zero leakage.
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        output_dir: Optional[Path] = None
    ):
        self.config = config or PreprocessingConfig()
        self.config.validate()
        self.preprocessor = RespiratoryAudioPreprocessor(self.config)
        self.output_dir = output_dir or (FEATURES_ROOT / "extracted")
        self.records_dir = self.output_dir / "records"
        self.events_dir = self.output_dir / "events"

        # Load clinical patient diagnoses
        self.patient_diagnoses = self._load_patient_diagnoses()

    def _load_patient_diagnoses(self) -> Dict[str, str]:
        """Loads clinical diagnoses from the official SPRSound_patient_summary.csv."""
        summary_path = DATASET_ROOT / "Patient Summary" / "SPRSound_patient_summary.csv"
        diagnoses = {}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("patient_num", "").strip()
                    if pid:
                        diagnoses[pid] = row.get("disease", "Unknown").strip()
        return diagnoses

    def parse_filename_metadata(self, stem: str) -> Dict[str, Any]:
        """
        Parses standard SPRSound filename format:
        {patient_num}_{age}_{gender}_{location}_{recording_number}
        """
        parts = stem.split("_")
        if len(parts) >= 5:
            pid = parts[0]
            try:
                age = float(parts[1])
            except ValueError:
                age = -1.0
            gender = parts[2]
            loc = parts[3]
            rec_num = parts[4]
        else:
            pid = parts[0] if parts else "UNKNOWN"
            age = -1.0
            gender = "UNKNOWN"
            loc = "UNKNOWN"
            rec_num = "UNKNOWN"

        clinical_dx = self.patient_diagnoses.get(pid, "Not_In_Summary")

        return {
            "record_id": stem,
            "patient_id": pid,
            "age_years": age,
            "gender": gender,
            "auscultation_location": loc,
            "recording_number": rec_num,
            "clinical_diagnosis": clinical_dx,
        }

    def generate_patient_grouped_splits(
        self,
        train_records: List[Dict[str, Any]],
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, int]:
        """
        Assigns each recording to a cross-validation fold (0 to n_splits-1)
        using StratifiedGroupKFold on patient_id and record_annotation.
        Guarantees that ALL recordings from a patient belong to the exact same fold.
        """
        record_ids = [r["record_id"] for r in train_records]
        patients = [r["patient_id"] for r in train_records]
        labels = [r["record_annotation"] for r in train_records]

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_assignments = {}

        for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(record_ids, labels, groups=patients)):
            for idx in val_indices:
                fold_assignments[record_ids[idx]] = fold_idx

        # Verification of patient disjointness across folds
        fold_patients = defaultdict(set)
        for r in train_records:
            rid = r["record_id"]
            fold = fold_assignments.get(rid, -1)
            fold_patients[fold].add(r["patient_id"])

        for f1 in range(n_splits):
            for f2 in range(f1 + 1, n_splits):
                overlap = fold_patients[f1].intersection(fold_patients[f2])
                if overlap:
                    raise ValueError(f"Patient leakage detected between fold {f1} and fold {f2}: {overlap}")

        logger.info("Generated %d patient-grouped folds. Patient overlap across folds: EXACTLY ZERO.", n_splits)
        return fold_assignments

    def select_smoke_test_samples(self) -> List[Tuple[Path, Path, str]]:
        """
        Selects a deterministic, small verified sample of genuine SPRSound recordings
        spanning each record annotation class (Normal, DAS, CAS, CAS & DAS, Poor Quality).
        Returns list of (wav_path, json_path, split_category).
        """
        train_wav_dir = DATASET_ROOT / "Classification" / "train_classification_wav"
        train_json_dir = DATASET_ROOT / "Classification" / "train_classification_json"

        # Organize by record_annotation
        by_class = defaultdict(list)
        for json_path in sorted(train_json_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec_ann = data.get("record_annotation", "Unknown")
            wav_path = train_wav_dir / f"{json_path.stem}.wav"
            if wav_path.exists():
                by_class[rec_ann].append((wav_path, json_path, "train_cv"))

        selected = []
        # Take 2 files from each class for a balanced 10-file smoke test
        for cls_name in ["Normal", "DAS", "CAS", "CAS & DAS", "Poor Quality"]:
            items = by_class.get(cls_name, [])
            selected.extend(items[:2])

        # Also select 2 inter-subject test files to verify held-out partition handling
        inter_json_dir = DATASET_ROOT / "Classification" / "valid_classification_json" / "2022" / "inter_test_json"
        inter_wav_dir = DATASET_ROOT / "Classification" / "valid_classification_wav" / "2022"
        if inter_json_dir.exists() and inter_wav_dir.exists():
            for jp in sorted(inter_json_dir.glob("*.json"))[:2]:
                wp = inter_wav_dir / f"{jp.stem}.wav"
                if wp.exists():
                    selected.append((wp, jp, "held_out_inter_test"))

        return selected

    def extract_record_features(
        self,
        wav_path: Path,
        json_path: Path,
        split_category: str,
        cv_fold: int = -1,
        overwrite: bool = False
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extracts both record-level and (where present) event-level features for a single file.
        Returns (record_metadata, list_of_event_metadata).
        Validates tensor shapes, non-finite values, and respects overwrite safeguards.
        """
        stem = wav_path.stem
        meta = self.parse_filename_metadata(stem)

        # 1. Read JSON annotation
        with open(json_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        record_ann = ann_data.get("record_annotation", "Unspecified")
        raw_events = ann_data.get("event_annotation", [])

        # Policy on Poor Quality:
        # Never silently discard. Extract with explicit quality flag.
        is_poor_quality = (record_ann == "Poor Quality")

        # Deterministic output path for record tensor
        record_npz_name = f"{split_category}_{stem}_record.npz"
        record_npz_path = self.records_dir / record_npz_name

        self.records_dir.mkdir(parents=True, exist_ok=True)

        # 2. Extract Whole-Record Mel Spectrogram
        if record_npz_path.exists() and not overwrite:
            existing_data = np.load(record_npz_path)
            record_tensor = existing_data["tensor"]
            record_feat_raw_dur = float(existing_data.get("raw_duration_seconds", 8.192))
            record_feat_time_frames = int(existing_data.get("time_frames_raw", 128))
        else:
            record_feat = self.preprocessor.extract_whole_record_features(str(wav_path), fixed_frames=128)
            record_tensor = record_feat["tensor"]
            record_feat_raw_dur = record_feat["raw_duration_seconds"]
            record_feat_time_frames = record_feat["time_frames_raw"]

            # Tensor shape and numerical validity guards
            if record_tensor.shape != (1, 128, 128, 1):
                raise ValueError(
                    f"Invalid record tensor shape for {stem}: {record_tensor.shape}, expected (1, 128, 128, 1)"
                )
            if np.isnan(record_tensor).any() or np.isinf(record_tensor).any():
                raise ValueError(f"Extracted record tensor for {stem} contains non-finite values (NaN or Inf).")

            np.savez_compressed(
                record_npz_path,
                tensor=record_tensor,
                spectrogram=record_feat["spectrogram"],
                record_id=stem,
                patient_id=meta["patient_id"],
                label=record_ann,
                raw_duration_seconds=record_feat_raw_dur,
                time_frames_raw=record_feat_time_frames
            )

        try:
            rel_feature_path = str(record_npz_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_feature_path = str(record_npz_path)

        record_meta = {
            **meta,
            "record_annotation": record_ann,
            "split_category": split_category,
            "cv_fold": cv_fold,
            "is_poor_quality": is_poor_quality,
            "num_events": len(raw_events),
            "audio_duration_seconds": round(record_feat_raw_dur, 3),
            "raw_time_frames": record_feat_time_frames,
            "tensor_shape": str(list(record_tensor.shape)),
            "feature_path": rel_feature_path,
            "val_min": round(float(record_tensor.min()), 4),
            "val_max": round(float(record_tensor.max()), 4),
            "val_mean": round(float(record_tensor.mean()), 4),
        }

        # 3. Extract Segmented Respiratory Events
        event_metadata_list = []
        if not is_poor_quality and raw_events:
            self.events_dir.mkdir(parents=True, exist_ok=True)
            audio = None
            sr = None

            for ev_idx, ev in enumerate(raw_events):
                ev_type = ev.get("type", "Unspecified")
                try:
                    s_ms = float(ev["start"])
                    e_ms = float(ev["end"])
                    if s_ms >= e_ms:
                        logger.warning("Skipping invalid event interval [%s, %s] in %s", s_ms, e_ms, stem)
                        continue

                    # Safe deterministic event filename
                    clean_type = ev_type.replace(" ", "_").replace("+", "_and_")
                    ev_npz_name = f"{split_category}_{stem}_ev{ev_idx:02d}_{clean_type}.npz"
                    ev_npz_path = self.events_dir / ev_npz_name

                    if ev_npz_path.exists() and not overwrite:
                        existing_ev = np.load(ev_npz_path)
                        ev_tensor = existing_ev["tensor"]
                    else:
                        if audio is None:
                            audio, sr = self.preprocessor.load_audio(str(wav_path))

                        # Target shape: (1, 128, 40, 1)
                        ev_feat = self.preprocessor.extract_event_features(
                            audio=audio,
                            sr=sr,
                            start_ms=s_ms,
                            end_ms=e_ms,
                            target_duration_s=self.config.event_duration_s
                        )
                        ev_tensor = ev_feat["tensor"]

                        if ev_tensor.shape != (1, 128, 40, 1):
                            raise ValueError(
                                f"Invalid event tensor shape for {stem}_ev{ev_idx}: {ev_tensor.shape}, expected (1, 128, 40, 1)"
                            )
                        if np.isnan(ev_tensor).any() or np.isinf(ev_tensor).any():
                            raise ValueError(f"Extracted event tensor for {stem}_ev{ev_idx} contains non-finite values.")

                        np.savez_compressed(
                            ev_npz_path,
                            tensor=ev_tensor,
                            spectrogram=ev_feat["spectrogram"],
                            record_id=stem,
                            patient_id=meta["patient_id"],
                            event_type=ev_type,
                            start_ms=s_ms,
                            end_ms=e_ms
                        )

                    try:
                        rel_ev_path = str(ev_npz_path.relative_to(PROJECT_ROOT))
                    except ValueError:
                        rel_ev_path = str(ev_npz_path)

                    event_metadata_list.append({
                        "event_id": f"{stem}_ev{ev_idx:02d}",
                        "record_id": stem,
                        "patient_id": meta["patient_id"],
                        "clinical_diagnosis": meta["clinical_diagnosis"],
                        "event_index": ev_idx,
                        "event_type": ev_type,
                        "start_ms": s_ms,
                        "end_ms": e_ms,
                        "duration_ms": round(e_ms - s_ms, 1),
                        "split_category": split_category,
                        "cv_fold": cv_fold,
                        "tensor_shape": str(list(ev_tensor.shape)),
                        "feature_path": rel_ev_path,
                        "val_min": round(float(ev_tensor.min()), 4),
                        "val_max": round(float(ev_tensor.max()), 4),
                        "val_mean": round(float(ev_tensor.mean()), 4),
                    })

                except Exception as exc:
                    logger.error("Failed to extract event %d from %s: %s", ev_idx, stem, exc)

        return record_meta, event_metadata_list

    def extract_partition(
        self,
        partition_name: str,
        wav_dir: Path,
        json_dir: Path,
        split_category: str,
        limit: Optional[int] = None,
        overwrite: bool = False,
        fold_map: Optional[Dict[str, int]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts features for a specific partition (e.g. train, inter_2022, intra_2022, test_2023).
        Guarantees WAV/JSON pairing, records manifests, and prevents accidental overwrite.
        """
        if not wav_dir.exists() or not json_dir.exists():
            raise FileNotFoundError(f"Partition directory missing: {wav_dir} or {json_dir}")

        json_files = sorted(json_dir.glob("*.json"))
        if limit is not None and limit > 0:
            json_files = json_files[:limit]

        logger.info(
            "Extracting partition '%s' (%d recordings) from %s ...",
            partition_name,
            len(json_files),
            json_dir
        )

        record_manifest = []
        event_manifest = []
        fold_map = fold_map or {}

        for idx, jp in enumerate(json_files, 1):
            stem = jp.stem
            wp = wav_dir / f"{stem}.wav"
            if not wp.exists():
                logger.error("WAV missing for annotation: %s", jp)
                continue

            cv_fold = fold_map.get(stem, -1)
            rec_meta, ev_metas = self.extract_record_features(
                wav_path=wp,
                json_path=jp,
                split_category=split_category,
                cv_fold=cv_fold,
                overwrite=overwrite
            )
            record_manifest.append(rec_meta)
            event_manifest.extend(ev_metas)

            if idx % 100 == 0 or idx == len(json_files):
                logger.info("Partition '%s' progress: %d/%d processed.", partition_name, idx, len(json_files))

        # Write partition manifests
        self._write_manifest(self.output_dir / f"{partition_name}_record_manifest.csv", record_manifest)
        if event_manifest:
            self._write_manifest(self.output_dir / f"{partition_name}_event_manifest.csv", event_manifest)

        # Save config
        config_path = self.output_dir / "preprocessing_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        return record_manifest, event_manifest

    def run_smoke_test(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes feature extraction on the small verified smoke-test subset.
        """
        logger.info("Initiating genuine dataset smoke test...")
        samples = self.select_smoke_test_samples()
        logger.info("Selected %d representative sample recordings for smoke test.", len(samples))

        record_manifest = []
        event_manifest = []

        train_samples_meta = [self.parse_filename_metadata(wp.stem) for wp, jp, sp in samples if sp == "train_cv"]
        for m, (wp, jp, sp) in zip(train_samples_meta, [s for s in samples if s[2] == "train_cv"]):
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
            m["record_annotation"] = d.get("record_annotation", "Unknown")

        min_cls = min(Counter(m["record_annotation"] for m in train_samples_meta).values()) if train_samples_meta else 1
        n_splits_smoke = min(2, min_cls)
        if n_splits_smoke > 1:
            splits_map = self.generate_patient_grouped_splits(train_samples_meta, n_splits=n_splits_smoke, random_state=42)
        else:
            splits_map = {m["record_id"]: 0 for m in train_samples_meta}

        for wav_path, json_path, split_cat in samples:
            cv_fold = splits_map.get(wav_path.stem, -1) if split_cat == "train_cv" else -1
            rec_meta, ev_metas = self.extract_record_features(
                wav_path, json_path, split_cat, cv_fold, overwrite=True
            )
            record_manifest.append(rec_meta)
            event_manifest.extend(ev_metas)
            logger.info(
                "Processed %s [%s] -> %d events | Record tensor: %s",
                wav_path.stem,
                rec_meta["record_annotation"],
                len(ev_metas),
                rec_meta["tensor_shape"]
            )

        # Write smoke test manifests
        self._write_manifest(self.output_dir / "smoke_record_manifest.csv", record_manifest)
        self._write_manifest(self.output_dir / "smoke_event_manifest.csv", event_manifest)

        # Save config
        config_path = self.output_dir / "preprocessing_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info("Smoke test complete: %d records, %d events extracted.", len(record_manifest), len(event_manifest))
        return record_manifest, event_manifest

    def generate_full_split_manifests(self) -> Dict[str, Any]:
        """
        Builds the complete patient-grouped 5-fold cross validation split metadata
        for all 1,949 training records and links the official 355 inter-subject test records.
        Does NOT extract audio features (metadata-only indexing).
        """
        train_json_dir = DATASET_ROOT / "Classification" / "train_classification_json"
        inter_json_dir = DATASET_ROOT / "Classification" / "valid_classification_json" / "2022" / "inter_test_json"
        intra_json_dir = DATASET_ROOT / "Classification" / "valid_classification_json" / "2022" / "intra_test_json"

        # 1. Parse all training records
        train_records = []
        for jp in sorted(train_json_dir.glob("*.json")):
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
            meta = self.parse_filename_metadata(jp.stem)
            meta["record_annotation"] = d.get("record_annotation", "Unknown")
            meta["num_events"] = len(d.get("event_annotation", []))
            meta["split_category"] = "train_cv"
            train_records.append(meta)

        # 2. Generate 5-fold StratifiedGroupKFold
        fold_assignments = self.generate_patient_grouped_splits(train_records, n_splits=5, random_state=42)
        for r in train_records:
            r["cv_fold"] = fold_assignments[r["record_id"]]

        # 3. Parse held-out inter-subject test records
        inter_records = []
        if inter_json_dir.exists():
            for jp in sorted(inter_json_dir.glob("*.json")):
                with open(jp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                meta = self.parse_filename_metadata(jp.stem)
                meta["record_annotation"] = d.get("record_annotation", "Unknown")
                meta["num_events"] = len(d.get("event_annotation", []))
                meta["split_category"] = "held_out_inter_test"
                meta["cv_fold"] = -1
                inter_records.append(meta)

        # 4. Parse intra-subject test records
        intra_records = []
        if intra_json_dir.exists():
            for jp in sorted(intra_json_dir.glob("*.json")):
                with open(jp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                meta = self.parse_filename_metadata(jp.stem)
                meta["record_annotation"] = d.get("record_annotation", "Unknown")
                meta["num_events"] = len(d.get("event_annotation", []))
                meta["split_category"] = "intra_subject_test"
                meta["cv_fold"] = -1
                intra_records.append(meta)

        # 5. Verify disjointness
        train_pids = set(r["patient_id"] for r in train_records)
        inter_pids = set(r["patient_id"] for r in inter_records)
        intra_pids = set(r["patient_id"] for r in intra_records)

        inter_leakage = train_pids.intersection(inter_pids)
        if inter_leakage:
            raise ValueError(f"CRITICAL: Patient leakage detected between train and inter-test: {inter_leakage}")

        # Write split metadata CSVs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_splits = train_records + inter_records + intra_records
        split_csv = self.output_dir / "dataset_split_manifest.csv"
        self._write_manifest(split_csv, all_splits)

        # Compute summary statistics
        fold_stats = {}
        for f_idx in range(5):
            fold_recs = [r for r in train_records if r["cv_fold"] == f_idx]
            fold_pids = set(r["patient_id"] for r in fold_recs)
            fold_classes = dict(Counter(r["record_annotation"] for r in fold_recs))
            fold_stats[f"fold_{f_idx}"] = {
                "record_count": len(fold_recs),
                "patient_count": len(fold_pids),
                "class_distribution": fold_classes
            }

        summary = {
            "total_train_records": len(train_records),
            "total_train_patients": len(train_pids),
            "held_out_inter_records": len(inter_records),
            "held_out_inter_patients": len(inter_pids),
            "intra_test_records": len(intra_records),
            "intra_test_patients": len(intra_pids),
            "train_inter_patient_overlap": len(inter_leakage),
            "folds": fold_stats,
            "split_manifest_path": str(split_csv.relative_to(PROJECT_ROOT))
        }

        summary_json = self.output_dir / "split_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Successfully wrote split manifest to %s", split_csv)
        return summary

    def run_train_extraction(self, limit: Optional[int] = None, overwrite: bool = False):
        """Extracts the training set partition (1,949 files)."""
        train_wav_dir = DATASET_ROOT / "Classification" / "train_classification_wav"
        train_json_dir = DATASET_ROOT / "Classification" / "train_classification_json"
        # Generate fold map to associate each recording with its patient-grouped fold
        split_manifest_path = self.output_dir / "dataset_split_manifest.csv"
        fold_map = {}
        if split_manifest_path.exists():
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split_category") == "train_cv":
                        fold_map[row["record_id"]] = int(row.get("cv_fold", -1))
        else:
            summary = self.generate_full_split_manifests()
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split_category") == "train_cv":
                        fold_map[row["record_id"]] = int(row.get("cv_fold", -1))

        return self.extract_partition(
            partition_name="train",
            wav_dir=train_wav_dir,
            json_dir=train_json_dir,
            split_category="train_cv",
            limit=limit,
            overwrite=overwrite,
            fold_map=fold_map
        )

    def run_inter_2022_extraction(self, limit: Optional[int] = None, overwrite: bool = False):
        """Extracts the official 2022 held-out inter-subject test partition (355 files)."""
        wav_dir = DATASET_ROOT / "Classification" / "valid_classification_wav" / "2022"
        json_dir = DATASET_ROOT / "Classification" / "valid_classification_json" / "2022" / "inter_test_json"
        return self.extract_partition(
            partition_name="inter_2022",
            wav_dir=wav_dir,
            json_dir=json_dir,
            split_category="held_out_inter_test",
            limit=limit,
            overwrite=overwrite
        )

    def run_intra_2022_extraction(self, limit: Optional[int] = None, overwrite: bool = False):
        """Extracts the 2022 intra-subject test partition (379 files). Note: Same-patient test only."""
        wav_dir = DATASET_ROOT / "Classification" / "valid_classification_wav" / "2022"
        json_dir = DATASET_ROOT / "Classification" / "valid_classification_json" / "2022" / "intra_test_json"
        return self.extract_partition(
            partition_name="intra_2022",
            wav_dir=wav_dir,
            json_dir=json_dir,
            split_category="intra_subject_test",
            limit=limit,
            overwrite=overwrite
        )

    def run_test_2023_extraction(self, limit: Optional[int] = None, overwrite: bool = False):
        """Extracts the 2023 test partition (871 files)."""
        wav_dir = DATASET_ROOT / "Classification" / "valid_classification_wav" / "2023"
        json_dir = DATASET_ROOT / "Classification" / "valid_classification_json" / "2023"
        return self.extract_partition(
            partition_name="test_2023",
            wav_dir=wav_dir,
            json_dir=json_dir,
            split_category="test_2023",
            limit=limit,
            overwrite=overwrite
        )

    def run_full_extraction(self, confirm_full: bool = False, overwrite: bool = False):
        """
        Extracts all 4 dataset partitions (3,554 recordings).
        Requires explicit confirm_full=True to prevent unintended mass execution.
        """
        if not confirm_full:
            raise ValueError(
                "Full dataset extraction requires explicit confirmation flag: '--confirm-full'. "
                "Processing 3,554 recordings consumes substantial CPU and disk resources."
            )
        logger.info("Executing full extraction across all 4 SPRSound partitions (3,554 total files)...")
        self.generate_full_split_manifests()
        train_res = self.run_train_extraction(overwrite=overwrite)
        inter_res = self.run_inter_2022_extraction(overwrite=overwrite)
        intra_res = self.run_intra_2022_extraction(overwrite=overwrite)
        test23_res = self.run_test_2023_extraction(overwrite=overwrite)
        return {
            "train": train_res,
            "inter_2022": inter_res,
            "intra_2022": intra_res,
            "test_2023": test23_res
        }

    @staticmethod
    def _write_manifest(csv_path: Path, rows: List[Dict[str, Any]]):
        """Writes a list of dictionaries to a CSV file."""
        if not rows:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="SPRSound Feature Extraction & Split Engine for Acoustic Epidemiology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast smoke test on 12 representative files:
  python ml/features/extract_features.py --mode smoke_test

  # Generate 5-fold patient-grouped split metadata:
  python ml/features/extract_features.py --mode generate_splits

  # Extract training partition (first 10 files for validation):
  python ml/features/extract_features.py --mode train --limit 10

  # Extract official 2022 held-out inter-subject benchmark:
  python ml/features/extract_features.py --mode inter_2022

  # Extract 2023 test partition:
  python ml/features/extract_features.py --mode test_2023

  # Full extraction (requires explicit --confirm-full):
  python ml/features/extract_features.py --mode full --confirm-full
        """
    )
    parser.add_argument(
        "--mode",
        choices=["smoke_test", "generate_splits", "train", "inter_2022", "intra_2022", "test_2023", "full"],
        default="smoke_test",
        help="Feature extraction or partition mode"
    )
    parser.add_argument(
        "--generate-splits",
        action="store_true",
        help="Also generate full 5-fold split manifest"
    )
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="Explicit confirmation required to execute full 3,554-file extraction"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files instead of reusing existing extracts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of recordings processed (useful for pipeline verification)"
    )
    args = parser.parse_args()

    pipeline = FeatureExtractionPipeline()

    if args.mode == "smoke_test":
        rec_meta, ev_meta = pipeline.run_smoke_test()
        print("\n" + "=" * 60)
        print("SMOKE TEST SUMMARY:")
        print(f"  Records processed: {len(rec_meta)}")
        print(f"  Events processed:  {len(ev_meta)}")
        if rec_meta:
            print(f"  Sample Record Shape: {rec_meta[0]['tensor_shape']}, range: [{rec_meta[0]['val_min']}, {rec_meta[0]['val_max']}]")
        if ev_meta:
            print(f"  Sample Event Shape:  {ev_meta[0]['tensor_shape']}, range: [{ev_meta[0]['val_min']}, {ev_meta[0]['val_max']}]")
        print("=" * 60)

    elif args.mode == "generate_splits":
        summary = pipeline.generate_full_split_manifests()
        print("\n" + "=" * 60)
        print("SPLIT GENERATION SUMMARY:")
        print(f"  Train Records:  {summary['total_train_records']} across {summary['total_train_patients']} patients")
        print(f"  Held-out Inter: {summary['held_out_inter_records']} across {summary['held_out_inter_patients']} patients")
        print(f"  Patient Overlap (Train vs Inter-Test): {summary['train_inter_patient_overlap']}")
        print("  Fold distribution:")
        for f_name, f_data in summary["folds"].items():
            print(f"    {f_name}: {f_data['record_count']} recs, {f_data['patient_count']} patients | {f_data['class_distribution']}")
        print("=" * 60)

    elif args.mode == "train":
        if args.generate_splits:
            pipeline.generate_full_split_manifests()
        rec_meta, ev_meta = pipeline.run_train_extraction(limit=args.limit, overwrite=args.overwrite)
        print(f"\n[OK] Training partition extraction complete: {len(rec_meta)} records, {len(ev_meta)} events.")

    elif args.mode == "inter_2022":
        rec_meta, ev_meta = pipeline.run_inter_2022_extraction(limit=args.limit, overwrite=args.overwrite)
        print(f"\n[OK] 2022 Inter-subject test extraction complete: {len(rec_meta)} records, {len(ev_meta)} events.")

    elif args.mode == "intra_2022":
        rec_meta, ev_meta = pipeline.run_intra_2022_extraction(limit=args.limit, overwrite=args.overwrite)
        print(f"\n[OK] 2022 Intra-subject test extraction complete: {len(rec_meta)} records, {len(ev_meta)} events.")

    elif args.mode == "test_2023":
        rec_meta, ev_meta = pipeline.run_test_2023_extraction(limit=args.limit, overwrite=args.overwrite)
        print(f"\n[OK] 2023 test extraction complete: {len(rec_meta)} records, {len(ev_meta)} events.")

    elif args.mode == "full":
        if not args.confirm_full:
            print("\n[ERROR] --mode full requires explicit '--confirm-full' flag to prevent unintended mass extraction.")
            sys.exit(1)
        res = pipeline.run_full_extraction(confirm_full=True, overwrite=args.overwrite)
        print("\n[OK] Full extraction across all 4 partitions complete.")

    if args.generate_splits and args.mode not in ["generate_splits", "train"]:
        pipeline.generate_full_split_manifests()


if __name__ == "__main__":
    main()

