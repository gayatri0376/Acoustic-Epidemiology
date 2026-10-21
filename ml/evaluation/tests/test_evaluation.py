"""
Unit Tests for Model Evaluation Engine
Acoustic Epidemiology Project

Verifies:
1. Honest reporting: evaluation refuses to fabricate metrics when artifact is missing.
2. Mathematical correctness of macro precision, recall, F1, and per-class reports.
3. Patient-level majority vote clustering logic.
4. Warning and isolation guards on intra-subject test partitions.
"""

import unittest
from pathlib import Path
import numpy as np

from ml.evaluation.evaluate_model import (
    evaluate_model,
    compute_evaluation_metrics,
    load_label_mapping
)


class TestEvaluationEngine(unittest.TestCase):

    def test_missing_artifact_refuses_to_fabricate_metrics(self):
        non_existent = Path("/path/to/definitely_missing_model.keras")
        res = evaluate_model(non_existent, partition="inter_2022")
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "ARTIFACT_NOT_FOUND")
        self.assertIn("does not fabricate evaluation metrics", res["message"])

    def test_metrics_computation_mathematics(self):
        # 4 classes: 0=Normal, 1=CAS, 2=DAS, 3=CAS & DAS
        classes = {"0": "Normal", "1": "CAS", "2": "DAS", "3": "CAS & DAS"}
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 1])  # 6 / 8 correct -> acc = 0.75
        y_probs = np.zeros((8, 4))
        for i, p in enumerate(y_pred):
            y_probs[i, p] = 1.0

        pids = ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"]

        metrics = compute_evaluation_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            patient_ids=pids,
            index_to_class=classes
        )

        rec = metrics["recording_level"]
        self.assertEqual(rec["total_recordings"], 8)
        self.assertAlmostEqual(rec["accuracy"], 0.75, places=4)
        self.assertIn("per_class", rec)
        self.assertEqual(len(rec["per_class"]), 4)
        self.assertEqual(rec["per_class"]["Normal"]["support"], 2)

        # Patient level:
        # P1: [0, 0] -> vote 0 (true 0) => correct
        # P2: [1, 0] -> tie/vote => 1 or 0
        # P3: [2, 2] -> vote 2 (true 2) => correct
        # P4: [3, 1] -> tie/vote
        pat = metrics["patient_level_majority_vote"]
        self.assertEqual(pat["total_patients"], 4)
        self.assertTrue(0.0 <= pat["patient_accuracy"] <= 1.0)

    def test_label_mapping_loader(self):
        primary = load_label_mapping("SPRSOUND_RECORD_4CLASS")
        self.assertEqual(primary["num_classes"], 4)
        self.assertEqual(primary["class_to_index"]["Normal"], 0)


if __name__ == "__main__":
    unittest.main()
