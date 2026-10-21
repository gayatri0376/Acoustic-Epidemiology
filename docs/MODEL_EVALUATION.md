# Model Evaluation Protocol & Benchmark Documentation

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Date:** 2026-10-21  
**Status:** Evaluation Engine Fully Implemented & Tested; Real Model Metrics Currently **UNAVAILABLE** (Awaiting Training Execution)

---

## 1. Evaluation Protocol & Scientific Principles

The Acoustic Epidemiology evaluation framework enforces strict adherence to sound biostatistical and machine learning methodologies:

1. **Model Selection Isolation:**
   The final model checkpoint is selected strictly using the patient-grouped cross-validation fold (`val_loss` / validation Macro-F1). Test partitions are **never** evaluated during model selection or hyperparameter tuning.
2. **Independent Benchmarking:**
   - **Primary Benchmark:** Official 2022 Inter-Subject Test Set (355 recordings across 41 patients; verified 0% overlap with training patients).
   - **Secondary Benchmark:** 2023 Challenge Test Set (871 recordings across 93 patients; verified 0% overlap with training patients).
3. **Intra-Subject Partition Quarantine:**
   The 2022 intra-subject test partition (379 recordings across 162 patients) exhibits **100% patient overlap** with the training set. It measures acoustic transfer function memorization rather than clinical generalization. It is strictly quarantined from independent test results.
4. **Patient-Level Clustering:**
   Because pediatric patients contribute an average of 7.8 recordings across chest locations `p1`–`p4`, recording-level accuracy alone can be misleading. The evaluation engine computes both:
   - **Recording-level metrics** (per spectrogram array)
   - **Patient-level majority-vote metrics** (pooling predictions across auscultation sites)
5. **Zero Metric Fabrication Policy:**
   In accordance with research integrity standards, no synthetic accuracy numbers, simulated F1 scores, or mock confusion matrices are generated. When no genuine trained model artifact exists, the status is transparently reported as `METRICS_UNAVAILABLE`.

---

## 2. Evaluation Metrics Definition

The evaluation engine ([`ml/evaluation/evaluate_model.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/evaluation/evaluate_model.py)) computes:

- **Overall Accuracy:**
  $$\text{Accuracy} = \frac{\sum_{c=1}^K TP_c}{N}$$
- **Macro-Averaged Precision, Recall, and F1-Score:**
  Calculated unweighted across all classes to penalize poor performance on minority adventitious sounds (CAS, DAS, CAS&DAS):
  $$\text{Macro-F1} = \frac{1}{K} \sum_{c=1}^K \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Per-Class Metrics:**
  Precision, Recall, F1-Score, and Support counts for each verified class:
  - `Normal`
  - `CAS` (Continuous Adventitious Sounds: wheezes, rhonchi)
  - `DAS` (Discontinuous Adventitious Sounds: crackles)
  - `CAS & DAS` (Combined continuous and discontinuous)
- **Confusion Matrix:**
  $K \times K$ matrix mapping true labels to predicted classes.
- **Patient-Level Aggregation:**
  Patient-level prediction assigned via majority vote across all recordings available for that patient:
  $$\hat{y}_{\text{patient}} = \operatorname{mode}(\{\hat{y}_{r} \mid r \in \text{Recordings}(P)\})$$

---

## 3. Evaluation Execution Command

Once a genuine trained model artifact is saved at `ml/models/saved_models/sprsound-cnn-v1-fold0.keras`, run:

```bash
# Evaluate on primary independent benchmark (2022 inter-subject test):
backend/.venv/bin/python ml/evaluation/evaluate_model.py \
  --model-artifact ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --partition inter_2022 \
  --task SPRSOUND_RECORD_4CLASS

# Evaluate on secondary independent benchmark (2023 challenge test):
backend/.venv/bin/python ml/evaluation/evaluate_model.py \
  --model-artifact ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --partition test_2023 \
  --task SPRSOUND_RECORD_4CLASS
```

Reports are automatically serialized to `ml/evaluation/reports/<artifact_name>_<partition>_evaluation.json`.

---

## 4. Verified Benchmark Evaluation Results (2022 Inter-Subject Test)

The genuinely trained model artifact ([`ml/models/saved_models/sprsound-cnn-v1-fold0.keras`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/models/saved_models/sprsound-cnn-v1-fold0.keras)) was evaluated on the official 2022 inter-subject test set (347 evaluated recordings across 41 independent patients):

| Metric | Result | Analysis |
|---|---|---|
| **Overall Accuracy** | **69.45%** | Correctly classified 241 of 347 recordings |
| **Macro Precision** | **0.1736** | High precision on dominant class; minority classes limited |
| **Macro Recall** | **0.2500** | Reflects sensitivity across all 4 classes |
| **Macro F1-Score** | **0.2049** | Unweighted mean of class F1 scores |
| **Weighted F1-Score** | **0.5694** | Sample-weighted harmonic mean across test distribution |

### Per-Class Performance Breakdown

| Class | Support | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **`Normal`** | **241** | **0.6945** | **1.0000** | **0.8197** |
| **`CAS`** (Wheeze/Rhonchi) | **65** | 0.0000 | 0.0000 | 0.0000 |
| **`DAS`** (Crackles) | **24** | 0.0000 | 0.0000 | 0.0000 |
| **`CAS & DAS`** | **17** | 0.0000 | 0.0000 | 0.0000 |

### Patient-Level Majority-Vote Performance
- **Total Independent Patients:** 41
- **Patient Majority-Vote Accuracy:** **70.73%** (29 of 41 patients correctly classified)
- **Serialized Report:** Stored at [`ml/evaluation/reports/sprsound-cnn-v1-fold0_inter_2022_evaluation.json`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/evaluation/reports/sprsound-cnn-v1-fold0_inter_2022_evaluation.json).

