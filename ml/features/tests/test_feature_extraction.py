"""
Unit & Integration Tests for Feature Extraction & Patient Leakage Prevention
Acoustic Epidemiology Project

Tests:
1. Frame counts and tensor dimensions for record-level and event-level features
2. Deterministic extraction reproducibility (bitwise identical tensors)
3. Metadata integrity and schema completeness
4. Safe handling of empty event lists (Poor Quality recordings)
5. Robust handling of invalid/malformed annotation intervals
6. Zero patient leakage across StratifiedGroupKFold cross-validation folds
7. Disjointness between training partitions and the held-out inter-subject test set
"""

import os
import json
import tempfile
import unittest
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

from ml.preprocessing.respiratory_preprocessor import (
    PreprocessingConfig,
    RespiratoryAudioPreprocessor
)
from ml.features.extract_features import FeatureExtractionPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "ml" / "dataset" / "SPRSound"


class TestFeatureExtractionIntegrity(unittest.TestCase):
    """Tests feature extraction logic, tensor dimensions, and deterministic reproducibility."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "extracted"
        self.pipeline = FeatureExtractionPipeline(output_dir=self.output_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_pair(self, stem, duration_s=9.2, record_label="Normal", events=None):
        """Creates a synthetic audio WAV and matching JSON annotation."""
        wav_path = Path(self.temp_dir.name) / f"{stem}.wav"
        json_path = Path(self.temp_dir.name) / f"{stem}.json"

        fs = 8000
        n_samples = int(duration_s * fs)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(str(wav_path), audio, fs)

        ann = {
            "record_annotation": record_label,
            "event_annotation": events if events is not None else []
        }
        json_path.write_text(json.dumps(ann))
        return wav_path, json_path

    def test_record_tensor_dimensions_and_numeric_range(self):
        wav_path, json_path = self._create_synthetic_pair(
            "40111111_5.0_0_p1_101",
            duration_s=9.216,
            record_label="Normal",
            events=[{"start": "1000", "end": "2500", "type": "Normal"}]
        )
        rec_meta, ev_metas = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 0)

        self.assertEqual(rec_meta["tensor_shape"], "[1, 128, 128, 1]")
        self.assertTrue(0.0 <= rec_meta["val_min"] <= rec_meta["val_max"] <= 1.0)
        self.assertEqual(len(ev_metas), 1)
        self.assertEqual(ev_metas[0]["tensor_shape"], "[1, 128, 40, 1]")
        self.assertTrue(0.0 <= ev_metas[0]["val_min"] <= ev_metas[0]["val_max"] <= 1.0)

    def test_deterministic_extraction_reproducibility(self):
        wav_path, json_path = self._create_synthetic_pair(
            "40222222_3.5_1_p2_202",
            duration_s=8.5,
            record_label="DAS",
            events=[{"start": "500", "end": "2000", "type": "Fine Crackle"}]
        )
        # Extract twice
        meta1, ev1 = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 0)
        feat_file1 = Path(meta1["feature_path"])
        if not feat_file1.is_absolute():
            feat_file1 = PROJECT_ROOT / feat_file1
        data1 = np.load(feat_file1)["tensor"]

        meta2, ev2 = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 0)
        feat_file2 = Path(meta2["feature_path"])
        if not feat_file2.is_absolute():
            feat_file2 = PROJECT_ROOT / feat_file2
        data2 = np.load(feat_file2)["tensor"]

        np.testing.assert_array_equal(data1, data2)

    def test_empty_event_list_handling(self):
        # Poor quality recording with zero events
        wav_path, json_path = self._create_synthetic_pair(
            "40333333_12.0_0_p3_303",
            duration_s=9.0,
            record_label="Poor Quality",
            events=[]
        )
        rec_meta, ev_metas = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 1)

        self.assertTrue(rec_meta["is_poor_quality"])
        self.assertEqual(rec_meta["num_events"], 0)
        self.assertEqual(len(ev_metas), 0)
        # Record tensor must still be extracted with shape (1, 128, 128, 1)
        self.assertEqual(rec_meta["tensor_shape"], "[1, 128, 128, 1]")

    def test_malformed_annotation_intervals_skipped_safely(self):
        # Event with start >= end
        wav_path, json_path = self._create_synthetic_pair(
            "40444444_2.1_1_p4_404",
            duration_s=8.0,
            record_label="CAS",
            events=[
                {"start": "3000", "end": "2000", "type": "Wheeze"},  # Inverted
                {"start": "1000", "end": "2500", "type": "Wheeze"},  # Valid
            ]
        )
        rec_meta, ev_metas = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 2)

        # Inverted event is rejected; valid event is processed
        self.assertEqual(len(ev_metas), 1)
        self.assertEqual(ev_metas[0]["event_index"], 1)
        self.assertEqual(ev_metas[0]["event_type"], "Wheeze")

    def test_overwrite_safeguard(self):
        """Verifies that existing feature tensors are not overwritten unless overwrite=True."""
        wav_path, json_path = self._create_synthetic_pair(
            "40999999_4.0_0_p1_999",
            duration_s=8.5,
            record_label="Normal",
            events=[]
        )
        rec_meta1, _ = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 0, overwrite=False)
        feat_path = self.output_dir / "records" / "train_cv_40999999_4.0_0_p1_999_record.npz"
        self.assertTrue(feat_path.exists())
        mtime1 = os.path.getmtime(feat_path)

        # Second extraction with overwrite=False
        rec_meta2, _ = self.pipeline.extract_record_features(wav_path, json_path, "train_cv", 0, overwrite=False)
        mtime2 = os.path.getmtime(feat_path)
        self.assertEqual(mtime1, mtime2)

    def test_full_extraction_requires_confirmation(self):
        """Verifies that full extraction raises ValueError without explicit confirm_full."""
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.run_full_extraction(confirm_full=False)
        self.assertIn("requires explicit confirmation flag", str(ctx.exception))


class TestPatientLeakagePrevention(unittest.TestCase):
    """Tests patient-grouped cross-validation and disjointness verification."""

    def setUp(self):
        self.pipeline = FeatureExtractionPipeline()

    def test_stratified_group_kfold_zero_patient_leakage(self):
        # Create a synthetic dataset with 20 patients having multiple recordings each
        synthetic_records = []
        classes = ["Normal", "DAS", "CAS", "CAS & DAS", "Poor Quality"]
        for p_idx in range(20):
            pid = f"PATIENT_{p_idx:03d}"
            # Assign each patient a fixed class and 3 recordings at locations p1, p2, p3
            cls_name = classes[p_idx % len(classes)]
            for loc in ["p1", "p2", "p3"]:
                synthetic_records.append({
                    "record_id": f"{pid}_5.0_0_{loc}_001",
                    "patient_id": pid,
                    "record_annotation": cls_name
                })

        fold_map = self.pipeline.generate_patient_grouped_splits(synthetic_records, n_splits=5, random_state=42)

        # Group patients by assigned fold
        fold_patients = defaultdict(set)
        for r in synthetic_records:
            fold = fold_map[r["record_id"]]
            fold_patients[fold].add(r["patient_id"])

        # Check every pair of folds has disjoint patient sets
        for f1 in range(5):
            self.assertTrue(len(fold_patients[f1]) > 0, f"Fold {f1} has 0 patients")
            for f2 in range(f1 + 1, 5):
                overlap = fold_patients[f1].intersection(fold_patients[f2])
                self.assertEqual(
                    len(overlap), 0,
                    f"Patient leakage detected between Fold {f1} and Fold {f2}: {overlap}"
                )

    def test_full_dataset_split_disjointness_verification(self):
        """Verifies that the generated full split manifest has zero patient leakage."""
        manifest_path = PROJECT_ROOT / "ml" / "features" / "extracted" / "dataset_split_manifest.csv"
        if not manifest_path.exists():
            # If not yet generated, generate it
            self.pipeline.generate_full_split_manifests()

        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        train_patients = set()
        inter_patients = set()
        fold_patients = defaultdict(set)

        for r in rows:
            pid = r["patient_id"]
            split = r["split_category"]
            fold = int(r["cv_fold"])
            if split == "train_cv":
                train_patients.add(pid)
                fold_patients[fold].add(pid)
            elif split == "held_out_inter_test":
                inter_patients.add(pid)

        # 1. Zero leakage between train and held-out inter-subject test
        overlap = train_patients.intersection(inter_patients)
        self.assertEqual(len(overlap), 0, f"Leakage detected between train and inter-test: {overlap}")

        # 2. Zero leakage across all 5 folds
        for f1 in range(5):
            for f2 in range(f1 + 1, 5):
                fold_overlap = fold_patients[f1].intersection(fold_patients[f2])
                self.assertEqual(len(fold_overlap), 0, f"Leakage between fold {f1} and fold {f2}: {fold_overlap}")



class TestDatasetPartitionReconciliation(unittest.TestCase):
    """
    Regression test suite verifying official SPRSound dataset partition reconciliation:
    1. Reconciles 1,605 files in valid_classification into 734 (2022 test) + 871 (2023 test).
    2. Verifies 1-to-1 correspondence between audio WAVs and JSON annotations.
    3. Confirms zero patient leakage between train and 2022 inter-subject test set.
    4. Confirms zero patient leakage between train and 2023 test set.
    """

    def setUp(self):
        self.classification_dir = DATASET_ROOT / "Classification"

    def test_partition_counts_and_reconciliation(self):
        train_wav = set(p.stem for p in (self.classification_dir / "train_classification_wav").glob("*.wav"))
        train_json = set(p.stem for p in (self.classification_dir / "train_classification_json").glob("*.json"))
        self.assertEqual(len(train_wav), 1949)
        self.assertEqual(len(train_json), 1949)
        self.assertEqual(train_wav, train_json)

        # 2022 test partition
        val_2022_wav = set(p.stem for p in (self.classification_dir / "valid_classification_wav" / "2022").glob("*.wav"))
        inter_2022_json = set(p.stem for p in (self.classification_dir / "valid_classification_json" / "2022" / "inter_test_json").glob("*.json"))
        intra_2022_json = set(p.stem for p in (self.classification_dir / "valid_classification_json" / "2022" / "intra_test_json").glob("*.json"))

        self.assertEqual(len(inter_2022_json), 355)
        self.assertEqual(len(intra_2022_json), 379)
        self.assertEqual(len(inter_2022_json.intersection(intra_2022_json)), 0)
        self.assertEqual(len(val_2022_wav), 734)
        self.assertEqual(inter_2022_json.union(intra_2022_json), val_2022_wav)

        # 2023 test partition
        val_2023_wav = set(p.stem for p in (self.classification_dir / "valid_classification_wav" / "2023").glob("*.wav"))
        val_2023_json = set(p.stem for p in (self.classification_dir / "valid_classification_json" / "2023").glob("*.json"))
        self.assertEqual(len(val_2023_wav), 871)
        self.assertEqual(len(val_2023_json), 871)
        self.assertEqual(val_2023_wav, val_2023_json)

        # Total reconciliation: 734 + 871 = 1605
        total_val_wavs = len(val_2022_wav) + len(val_2023_wav)
        total_val_jsons = len(inter_2022_json) + len(intra_2022_json) + len(val_2023_json)
        self.assertEqual(total_val_wavs, 1605)
        self.assertEqual(total_val_jsons, 1605)

    def test_patient_isolation_across_partitions(self):
        pipeline = FeatureExtractionPipeline()

        train_stems = [p.stem for p in (self.classification_dir / "train_classification_json").glob("*.json")]
        train_pids = set(pipeline.parse_filename_metadata(s)["patient_id"] for s in train_stems)
        self.assertEqual(len(train_pids), 251)

        inter_2022_stems = [p.stem for p in (self.classification_dir / "valid_classification_json" / "2022" / "inter_test_json").glob("*.json")]
        inter_2022_pids = set(pipeline.parse_filename_metadata(s)["patient_id"] for s in inter_2022_stems)
        self.assertEqual(len(inter_2022_pids), 41)
        # Zero leakage between train and 2022 inter-subject test
        self.assertEqual(len(train_pids.intersection(inter_2022_pids)), 0)

        # Intra-subject test deliberately overlaps with train
        intra_2022_stems = [p.stem for p in (self.classification_dir / "valid_classification_json" / "2022" / "intra_test_json").glob("*.json")]
        intra_2022_pids = set(pipeline.parse_filename_metadata(s)["patient_id"] for s in intra_2022_stems)
        self.assertEqual(len(intra_2022_pids), 162)
        self.assertEqual(len(train_pids.intersection(intra_2022_pids)), 162)

        # 2023 test set patients (ignoring empty PID prefix from withheld patient)
        val_2023_stems = [p.stem for p in (self.classification_dir / "valid_classification_json" / "2023").glob("*.json")]
        val_2023_pids = set(pipeline.parse_filename_metadata(s)["patient_id"] for s in val_2023_stems if pipeline.parse_filename_metadata(s)["patient_id"])
        # Zero leakage between train and 2023 test patients
        self.assertEqual(len(train_pids.intersection(val_2023_pids)), 0)
        # Zero leakage between 2022 inter-test and 2023 test patients
        self.assertEqual(len(inter_2022_pids.intersection(val_2023_pids)), 0)


if __name__ == "__main__":
    unittest.main()

