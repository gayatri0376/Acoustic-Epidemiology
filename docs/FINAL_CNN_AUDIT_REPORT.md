# FINAL CNN AUDIT, VALIDATION & RESEARCH HANDOFF REPORT
**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Role:** Mohini (Machine Learning Modeling & CNN Lead)  
**Date:** 19 November 2026  
**Final Project Deadline:** 18 November 2026 (Day +1 Final Audit & Verification)  
**Status:** COMPLETE — RIGOROUS AUDIT CONCLUDED — INTEGRATION DECISIONS DOCUMENTED  

---

## 1. Experiment 4 Verification

### 1.1 Complete Confusion Matrix (Fold 0 Validation Set, $N=357$)

```text
Actual \ Predicted               Healthy Control       Pneumonia    Wheezing Disorders        Total
---------------------------------------------------------------------------------------------------
Healthy Control                               24               9                     0           33
Pneumonia                                    115             132                     0          247
Wheezing Disorders                            52              25                     0           77
---------------------------------------------------------------------------------------------------
Total Predicted                              191             166                     0          357
```

### 1.2 Class-wise Prediction Distribution

| Class Name | Ground Truth Count | Ground Truth % | Predicted Count | Predicted % | Net Bias |
|---|---:|---:|---:|---:|---|
| **Healthy Control** | 33 | 9.24% | 191 | 53.50% | Over-predicted by $+44.26\%$ |
| **Pneumonia** | 247 | 69.19% | 166 | 46.50% | Under-predicted by $-22.69\%$ |
| **Wheezing Disorders** | 77 | 21.57% | 0 | 0.00% | **Zero predictions ($-21.57\%$)** |
| **Total** | 357 | 100.00% | 357 | 100.00% | — |

### 1.3 Class-wise Performance Metrics

| Class Name | Precision | Recall | F1-Score | Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Healthy Control** | 12.57% | 72.73% | 0.2143 | 33 | 24 | 167 | 9 |
| **Pneumonia** | 79.52% | 53.44% | 0.6392 | 247 | 132 | 34 | 115 |
| **Wheezing Disorders** | 0.00% | 0.00% | 0.0000 | 77 | 0 | 0 | 77 |

### 1.4 Summary Performance Metrics
- **Overall Accuracy:** $43.70\%$ ($156 / 357$)
- **Balanced Accuracy (Macro Average Recall):** $42.06\%$ ($\frac{0.7273 + 0.5344 + 0.0000}{3}$)
- **Macro F1-Score:** $0.2845$ ($\frac{0.2143 + 0.6392 + 0.0000}{3}$)
- **Weighted F1-Score:** $0.4619$

### 1.5 Why Wheezing Disorders Recall is 0.00%
The model **never predicted Wheezing Disorders for any validation sample** (0 out of 357 predictions). The mathematical and clinical causes are:
1. **Compound Over-Weighting of Healthy Control:**  
   In Experiment 4, Healthy Control was augmented $4\times$ (from 83 to 415 samples) and was simultaneously assigned an unadjusted focal loss weight of $\alpha_0 = 5.1285$. This gave Healthy Control an effective gradient multiplication factor of $4 \times 5.1285 = 20.51$. In contrast, Wheezing Disorders was augmented $2\times$ with $\alpha_2 = 2.3134$, yielding an effective factor of $4.63$. Healthy Control received $4.43\times$ higher gradient leverage than Wheezing Disorders.
2. **Acoustic Subsumption by Normal Sounds:**  
   In the Wheezing Disorders validation subset ($N=77$), **61 recordings (79.2%) contain normal vesicular breath sounds** without wheezing or rhonchi. When evaluated, the network's high Healthy Control sensitivity caused 52 of these 77 samples to be classified as Healthy Control, and the remaining 25 were classified as Pneumonia.
3. **Logit Margin Deficiency:**  
   At no point during forward inference did the softmax logit for Wheezing Disorders exceed both the Healthy Control threshold and the Pneumonia baseline.

---

## 2. Model Artifact Verification

| Parameter | Sound Classification Model | Disease Model (Exp 1 - Deployed) | Disease Model (Exp 4 - Verification) |
|---|---|---|---|
| **Artifact Filename** | `sprsound-cnn-v1-fold0.keras` | `sprsound-disease-cnn-v1-fold0.keras` | In-memory session checkpoint (Epoch 7) |
| **Filesystem Location** | `ml/models/saved_models/` | `Cnn model/saved_models/` | `ml/handoff/evaluation/disease_model_exp4_eval.json` |
| **File Size (bytes)** | 5,172,706 | 1,406,343 | N/A (Session evaluation record) |
| **Modification Timestamp** | 2026-10-21 12:14:18 | 2026-11-19 20:46:12 | 2026-11-19 21:35:08 |
| **Architecture Name** | 4-Block `PediatricRespiratorySoundCNN` | 3-Block `PediatricThroatDiseaseCNN` | 3-Block `PediatricThroatDiseaseCNN` |
| **Parameter Count** | 1,289,860 | 110,211 | 110,211 |
| **Input Shape** | `(None, 128, 128, 1)` | `(None, 128, 128, 1)` | `(None, 128, 128, 1)` |
| **Output Shape** | `(None, 4)` | `(None, 3)` | `(None, 3)` |
| **Class Order** | `['Normal', 'CAS', 'DAS', 'CAS & DAS']` | `['Healthy Control', 'Pneumonia', 'Wheezing Disorders']` | `['Healthy Control', 'Pneumonia', 'Wheezing Disorders']` |
| **Loss Function** | Categorical Cross-Entropy | Focal Loss ($\alpha=[1.5, 0.6, 1.0], \gamma=2.0$) | Focal Loss ($\alpha=[5.13, 0.42, 2.31], \gamma=1.5$) |
| **Augmentation** | None | SpecAugment ($1\times$ per sample) | Targeted SpecAugment ($4\times$ Healthy, $2\times$ Wheezing) |
| **Provenance Status** | **VERIFIED ON DISK** | **VERIFIED ON DISK** | **SESSION EVALUATION RECORD (UNSERIALIZED)** |

*Explicit Provenance Statement:* The physical `.keras` file in `Cnn model/saved_models/` corresponds strictly to **Experiment 1**. Experiment 4 was an empirical verification run whose metrics were logged and verified. In accordance with safety rules, existing baseline artifacts on disk were not overwritten.

---

## 3. Checkpoint Verification

### 3.1 Checkpoint Integrity Audit
- **Selected Epoch:** Epoch 7 (out of 25 completed epochs).
- **Selection Metric:** `Balanced Accuracy` ($\frac{\text{Recall}_0 + \text{Recall}_1 + \text{Recall}_2}{3}$).
- **Genuinely Trained Status:** **VERIFIED.** Epoch 7 is an actively trained intermediate state, not an untrained initial checkpoint. At Epoch 1, Balanced Accuracy was 33.6%; it peaked at Epoch 7 at 42.06% with active dual-class discrimination (72.7% Healthy, 53.4% Pneumonia), before subsequent epochs overfitted to Healthy Control.

### 3.2 Full Epoch-by-Epoch Training & Validation Progression (Experiment 4)

| Epoch | Train Loss | Val Balanced Accuracy | Val Macro F1 | Healthy Recall | Pneumonia Recall | Wheezing Recall | Notes / Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| **1** | 0.9412 | 33.6% | 0.0784 | 97.0% | 0.0% | 3.9% | Initial learning |
| **2** | 0.7120 | 33.3% | 0.0564 | 100.0% | 0.0% | 0.0% | Healthy bias onset |
| **3** | 0.6241 | 33.3% | 0.0564 | 100.0% | 0.0% | 0.0% | Healthy plateau |
| **4** | 0.5518 | 33.3% | 0.0564 | 100.0% | 0.0% | 0.0% | Feature adaptation |
| **5** | 0.4890 | 37.2% | 0.1948 | 84.8% | 26.7% | 0.0% | Pneumonia boundary forms |
| **6** | 0.4312 | 38.8% | 0.2964 | 51.5% | 64.8% | 0.0% | Pneumonia strengthens |
| **7** | **0.3985** | **42.1%** | **0.2845** | **72.7%** | **53.4%** | **0.0%** | **BEST BALANCED ACCURACY (RESTORED)** |
| **8** | 0.3621 | 40.0% | 0.2294 | 84.8% | 35.2% | 0.0% | Reverting to Healthy bias |
| **9** | 0.3340 | 36.8% | 0.1378 | 97.0% | 13.4% | 0.0% | Overfitting |
| **10** | 0.3012 | 34.4% | 0.0785 | 100.0% | 3.2% | 0.0% | Overfitting |
| **11–25** | $\le 0.28$ | $33.3\% - 33.9\%$ | $\le 0.0676$ | $100.0\%$ | $\le 1.6\%$ | $0.0\%$ | Complete collapse to Healthy Control |

---

## 4. Patient Leakage Audit

### 4.1 Cross-Validation Split Summary

| Partition / Fold | Recording Count | Patient Count | Patients per Class (H / P / W) | Inter-Fold Overlap |
|---|---:|---:|---|---|
| **Disease Fold 0 (Validation)** | 357 | 45 | 4 / 32 / 9 | **0 patients** |
| **Disease Fold 1** | 326 | 45 | 3 / 34 / 8 | **0 patients** |
| **Disease Fold 2** | 279 | 46 | 4 / 29 / 13 | **0 patients** |
| **Disease Fold 3** | 330 | 45 | 3 / 36 / 6 | **0 patients** |
| **Disease Fold 4** | 342 | 45 | 3 / 33 / 9 | **0 patients** |
| **Total Disease Dataset** | **1,634** | **226** | **17 / 164 / 45** | **Zero leakage across all pairs** |

### 4.2 Leakage Vector Audit
- **Duplicate Recordings:** Checked via record UUIDs and filenames. All 1,634 entries in `disease_split_manifest.csv` correspond to distinct audio files.
- **Preprocessing Leakage:** Feature extraction was applied per-recording independently. No global dataset-wide normalization parameters were fit across train and validation sets.
- **Augmentation Leakage:** SpecAugment was executed dynamically inside the training batch loop on training samples only. Validation samples were never augmented.
- **Split Reproducibility:** Verified. Generated via stratified group k-fold on patient ID with deterministic seed.

---

## 5. Scientific Validity of Disease Classification

### 5.1 The Core Scientific Problem: Patient Diagnosis vs. Localized Sound
Assigning systemic, patient-level hospital admission diagnoses directly to individual 8-second auscultation recordings is **scientifically invalid for single-recording inference on this dataset**:

1. **Auscultation Physics:**  
   Pneumonia produces localized consolidation (typically crackles in lower lobes). Auscultating an anterior upper quadrant (`p1` or `p2`) of a pneumonia patient frequently captures **completely healthy, clear vesicular breathing**.
2. **Empirical Evidence of Label Noise:**  
   - In Pneumonia: **852 out of 1,257 recordings (67.8%) are acoustically Normal.**
   - In Wheezing Disorders: **206 out of 261 recordings (78.9%) are acoustically Normal.**
   - In Healthy Control: **116 out of 116 recordings (100.0%) are acoustically Normal.**
3. **The Normal Confounder:**  
   Out of 1,634 recordings, **1,174 (71.8%) share the exact same acoustic feature set (normal vesicular breathing)** while bearing three opposing labels.
4. **Conclusion on Single-Recording Disease Prediction:**  
   A CNN cannot extract acoustic signatures of pneumonia from an audio segment that contains no acoustic abnormalities. Asking a neural network to classify disease from single normal breath sounds forces it to memorize sample IDs or collapse to class priors.

---

## 6. Four-Class Acoustic Sound Classification Evaluation

The 4-class acoustic classification model (`sprsound-cnn-v1-fold0.keras`) was evaluated on both Validation Fold 0 and the independent held-out test set (`held_out_inter_test`).

### 6.1 Validation Fold 0 Evaluation ($N=354$ recordings, 49 patients)

```text
Actual \ Predicted         Normal       CAS       DAS    CAS & DAS     Total
----------------------------------------------------------------------------
Normal                        261         0         0            0       261
CAS                            25         0         0            0        25
DAS                            49         0         0            0        49
CAS & DAS                      19         0         0            0        19
----------------------------------------------------------------------------
Total Predicted               354         0         0            0       354
```

- **Overall Accuracy:** $73.73\%$
- **Balanced Accuracy:** $25.00\%$
- **Macro F1-Score:** $0.2122$
- **Weighted F1-Score:** $0.6258$
- **Per-Class Metrics:**
  - `Normal`: Precision = 73.73%, Recall = 100.00%, F1 = 0.8488 (Support = 261)
  - `CAS`: Precision = 0.00%, Recall = 0.00%, F1 = 0.0000 (Support = 25)
  - `DAS`: Precision = 0.00%, Recall = 0.00%, F1 = 0.0000 (Support = 49)
  - `CAS & DAS`: Precision = 0.00%, Recall = 0.00%, F1 = 0.0000 (Support = 19)

### 6.2 Independent Held-Out Test Set Evaluation ($N=347$ recordings)

```text
Actual \ Predicted         Normal       CAS       DAS    CAS & DAS     Total
----------------------------------------------------------------------------
Normal                        241         0         0            0       241
CAS                            65         0         0            0        65
DAS                            24         0         0            0        24
CAS & DAS                      17         0         0            0        17
----------------------------------------------------------------------------
Total Predicted               347         0         0            0       347
```

- **Overall Accuracy:** $69.45\%$
- **Balanced Accuracy:** $25.00\%$
- **Macro F1-Score:** $0.2049$
- **Weighted F1-Score:** $0.5693$
- **Adventitious Sound Recall:** **0.00% across all adventitious classes (CAS, DAS, CAS & DAS)**.

### 6.3 Sound Model Checkpoint Root Cause
Inspection of `sprsound-cnn-v1-fold0_history.json` revealed:
- `val_loss` was $0.9992$ at Epoch 0, increasing monotonically to $1.6308$ at Epoch 6.
- Because `ModelCheckpoint` monitored `val_loss` with `save_best_only=True`, the saved artifact is an **Epoch 0 model**. It learned only the majority class prior ($73.7\%$ Normal) and has **zero discriminative capacity for adventitious sounds**.

---

## 7. Multi-Site Patient-Level Aggregation (Experimental Research Direction)

Multi-site aggregation is an **experimental research hypothesis**, not an established clinical rule:

### 7.1 Proposed Aggregation Scheme
Instead of evaluating recordings independently, a patient's auscultations across all recorded quadrants ($p_1, p_2, p_3, p_4$) are pooled:
$$\text{Patient Diagnosis} = \begin{cases} 
\text{Wheezing Disorder} & \text{if } \exists \, r \in \text{Recordings}(\text{Patient}) : \text{Predict}(r) = \text{CAS} \\
\text{Pneumonia} & \text{if } \exists \, r \in \text{Recordings}(\text{Patient}) : \text{Predict}(r) = \text{DAS} \\
\text{Healthy Control} & \text{if } \forall \, r \in \text{Recordings}(\text{Patient}) : \text{Predict}(r) = \text{Normal}
\end{cases}$$

### 7.2 Strict Experimental Caveats
1. **Unvalidated:** This rule has **not** undergone prospective clinical validation.
2. **Co-morbidity Ambiguity:** A child with viral pneumonia can present with both wheezing and crackles. Heuristic rules cannot replace multi-modal clinical judgment.
3. **Sound Model Dependency:** This scheme strictly depends on an acoustic classifier that accurately detects CAS and DAS. Because the current sound model has 0% adventitious recall, aggregation currently yields 100% Healthy Control diagnoses.

---

## 8. Final Artifacts & Handoff Directory

All configurations, mappings, and evaluation outputs have been compiled into `ml/handoff/`:

- `ml/handoff/README.md`: Reproduction documentation and provenance report.
- `ml/handoff/configs/sound_class_mapping.json`: 4-class acoustic mapping.
- `ml/handoff/configs/sound_preprocessing_config.json`: Verified 8kHz preprocessing spec.
- `ml/handoff/configs/disease_class_mapping.json`: 3-class disease mapping marked experimental.
- `ml/handoff/configs/disease_preprocessing_config.json`: Disease preprocessing spec.
- `ml/handoff/evaluation/sound_model_val_fold0_eval.json`: Validation evaluation metrics.
- `ml/handoff/evaluation/sound_model_val_fold0_cm.csv`: Validation confusion matrix.
- `ml/handoff/evaluation/sound_model_held_out_test_eval.json`: Held-out test evaluation metrics.
- `ml/handoff/evaluation/sound_model_held_out_test_cm.csv`: Held-out test confusion matrix.
- `ml/handoff/evaluation/disease_model_exp1_eval.json`: Deployed disease model evaluation (Pneumonia collapse).
- `ml/handoff/evaluation/disease_model_v2_eval.json`: Disease model v2 evaluation (Healthy collapse).
- `ml/handoff/evaluation/disease_model_exp4_eval.json`: Experiment 4 verified metrics.
- `ml/handoff/evaluation/disease_model_exp4_cm.csv`: Experiment 4 confusion matrix.

---

## 9. Backend Safety & Non-Interference Statement

- **No backend files were modified.**
- **No database records were altered, overwritten, or created.**
- **No active models were registered, replaced, or activated in Django.**
- **No Git commands were executed.**
- **No dependencies or packages were modified.**
- **All existing backend tests continue to pass without modification.**

---

## 10. Final Report & Four Independent Conclusions

### Conclusion A: Four-Class Acoustic Sound Classification
**Status: REJECTED FOR PRODUCTION / RESEARCH BENCHMARK ONLY**  
The active model artifact (`sprsound-cnn-v1-fold0.keras`) collapsed to the majority class (Normal) due to Epoch 0 checkpoint restoration under monotonic validation loss increase. It achieves 73.73% accuracy on validation and 69.45% on independent held-out testing solely by predicting `Normal` for 100% of samples. It detects 0% of Continuous Adventitious Sounds (CAS), 0% of Discontinuous Adventitious Sounds (DAS), and 0% of mixed sounds. It is **not ready for clinical or deployment use**.

### Conclusion B: Three-Class Disease Classification
**Status: SCIENTIFICALLY UNSUPPORTED ON SINGLE RECORDINGS**  
Single-recording pediatric disease classification is not supported by the SPRSound dataset in its current formulation. Because 71.8% of recordings from pneumonia and wheezing patients are acoustically normal vesicular sounds, single recordings do not contain localized acoustic features corresponding to systemic diagnoses. The models inevitably collapse to predicting class priors.

### Conclusion C: Multi-Site Patient-Level Aggregation
**Status: EXPERIMENTAL RESEARCH DIRECTION ONLY**  
Pooling multiple chest quadrant recordings per patient to infer disease status is clinically rational but remains an unvalidated experimental concept. It must not be deployed as an automated diagnostic rule without prospective clinical trial data and an accurate underlying acoustic sound classifier.

### Conclusion D: Backend Integration Readiness
**Status: BACKEND INTEGRATION IS STRICTLY BLOCKED**  
Neither the sound classification model nor the disease classification model possesses the discriminative capability required for deployment. Integrating either model into production would result in deceptive behavior (the sound model predicting 100% Normal; the disease model predicting 100% Pneumonia or 100% Healthy Control).

---

## Summary of Next Steps for Project Leadership

1. **Final Task Status:** **AUDIT & HANDOFF COMPLETE.**
2. **Files Created:**
   - `docs/FINAL_CNN_AUDIT_REPORT.md`
   - `docs/MASTER_CNN_DEBUGGING_REPORT.md`
   - `ml/handoff/README.md`
   - All evaluation JSONs and CSVs in `ml/handoff/evaluation/`
   - All configuration specs in `ml/handoff/configs/`
3. **Missing Evidence:** Pre-trained acoustic weights from an external dataset (e.g., ICBHI 2017) to initialize sound feature representations prior to fine-tuning on SPRSound.
4. **Exact Recommendation:**
   - **Do not deploy CNN inference to production.**
   - Configure the backend to return an explicit educational research notice explaining that pediatric acoustic sound models are in developmental validation.
   - For future modeling, shift focus entirely to the **4-class acoustic sound classification task**, utilizing transfer learning from audio backbones (e.g., PANNs or Audio Spectrogram Transformer) and Balanced Accuracy checkpointing to learn genuine adventitious sound boundaries.
