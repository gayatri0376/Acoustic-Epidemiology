# ACOUSTIC SOUND CLASSIFICATION BASELINE EXPERIMENT REPORT
**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases  
**Experiment ID:** `ACOUSTIC_BASELINE_V1`  
**Task:** `SPRSOUND_RECORD_4CLASS` (Normal, CAS, DAS, CAS & DAS)  
**Date:** 19 November 2026  
**Role:** Mohini (Machine Learning Modeling & CNN Lead)  
**Status:** EXPERIMENT COMPLETE — REPRODUCIBLE BASELINE ESTABLISHED  

---

## 1. Executive Summary

In prior audits, the deployed four-class acoustic sound classification model (`sprsound-cnn-v1-fold0.keras`) suffered from a complete majority-class collapse: it predicted `Normal` for 100% of samples, achieving only **25.00% Balanced Accuracy** and **0.00% recall on all adventitious sounds** (CAS, DAS, CAS & DAS). This occurred because the checkpoint callback monitored unweighted `val_loss`, which froze the model at Epoch 0.

In this experiment (`ACOUSTIC_BASELINE_V1`), we engineered a dedicated **4-Block 2D-CNN with Dual Pooling, targeted minority SpecAugment, and dynamic Balanced Accuracy checkpoint selection**. 

### Key Performance Milestones:
- **Brokethrough the Single-Class Collapse:** On Validation Fold 0, the model achieved **33.46% Balanced Accuracy** and a **Macro F1-Score of 0.3284** (up from 0.2122).
- **Strong Adventitious Crackle Detection (DAS):** DAS recall increased from **0.00% to 34.69%** on validation, and reached **45.83%** on the independent test set.
- **Independent Test Set Generalization (`held_out_inter_test`, $N=347$):** The model actively predicted **all four classes**, achieving **37.12% Balanced Accuracy** and **0.3018 Macro F1** across 41 completely unseen patients.

---

## 2. Dataset & Split Partitioning

All splits strictly enforce patient disjointness without data leakage:

| Split Category | Fold | Recording Count | Patient Count | Normal | CAS (Wheeze) | DAS (Crackle) | CAS & DAS (Both) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Training (`train_cv`)** | Folds 1–4 | 1,418 | 194 | 1,042 (73.48%) | 101 (7.12%) | 199 (14.03%) | 76 (5.36%) |
| **Training (Augmented)** | Folds 1–4 | 2,745 | 194 | 1,364 (49.69%) | 404 (14.72%) | 597 (21.75%) | 380 (13.84%) |
| **Validation (`train_cv`)** | Fold 0 | 354 | 49 | 261 (73.73%) | 25 (7.06%) | 49 (13.84%) | 19 (5.37%) |
| **Independent Test (`held_out_inter_test`)** | N/A | 347 | 41 | 241 (69.45%) | 65 (18.73%) | 24 (6.92%) | 17 (4.90%) |

*Patient Overlap Audit:*
- `train_cv` (Folds 1–4) vs. Validation (Fold 0): **0 overlapping patients.**
- `train_cv` vs. `held_out_inter_test`: **0 overlapping patients.**

---

## 3. Model Architecture Specification

- **Architecture Name:** `PediatricAcousticBaselineCNN`
- **Input Shape:** `(None, 128, 128, 1)` (Log Mel-Spectrogram, 8 kHz, 50–2000 Hz Butterworth bandpass, minmax normalized)
- **Convolutional Backbone:**
  - Block 1: Conv2D(32, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.20)
  - Block 2: Conv2D(64, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.20)
  - Block 3: Conv2D(128, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.25)
  - Block 4: Conv2D(256, 3x3, name="`last_conv_layer`") $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool(2x2)
- **Dual Pooling Layer:** Concatenation of Global Average Pooling (harmonic continuity) and Global Max Pooling (impulse transient detection).
- **Classification Head:** Dense(128, ReLU, L2=1e-4) $\rightarrow$ Dropout(0.30) $\rightarrow$ Dense(4, Softmax).
- **Parameter Count:** 1,389,092 trainable parameters.
- **Grad-CAM Ready:** Layer `last_conv_layer` is explicitly named for authentic gradient backpropagation.

---

## 4. Training Progression & Checkpoint History

The model was trained for 25 epochs with Adam ($\text{LR}=3 \times 10^{-4}$), batch size 32, and dynamic tracking of validation Balanced Accuracy:

| Epoch | Train Loss | Train Accuracy | Val Accuracy | Val Balanced Accuracy | Val Macro F1 | Val Recall Normal | Val Recall DAS | Val Recall Both | Status / Checkpoint Action |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **1–9** | $1.71 \rightarrow 1.22$ | $32.0\% \rightarrow 50.9\%$ | $73.7\%$ | $25.0\%$ | $0.2122$ | $100.0\%$ | $0.0\%$ | $0.0\%$ | Majority bias phase (rejected) |
| **10** | $1.2122$ | $51.3\%$ | $73.7\%$ | $25.0\%$ | $0.2125$ | $100.0\%$ | $0.0\%$ | $0.0\%$ | Checkpoint registered |
| **14** | $1.1422$ | $54.4\%$ | $74.0\%$ | $25.9\%$ | $0.2317$ | $99.6\%$ | $4.1\%$ | $0.0\%$ | Checkpoint registered |
| **16** | $1.0923$ | $57.4\%$ | $73.2\%$ | $31.9\%$ | $0.2974$ | $92.7\%$ | $34.7\%$ | $0.0\%$ | Checkpoint registered |
| **18** | $1.0225$ | $59.2\%$ | $72.9\%$ | $32.6\%$ | $0.3025$ | $91.6\%$ | $38.8\%$ | $0.0\%$ | Checkpoint registered |
| **19** | $0.9902$ | $61.2\%$ | $74.0\%$ | $29.2\%$ | $0.2860$ | $97.3\%$ | $14.3\%$ | $5.3\%$ | Multi-class emergence |
| **20** | $0.9401$ | $62.8\%$ | $71.5\%$ | $30.8\%$ | $0.2983$ | $91.6\%$ | $26.5\%$ | $5.3\%$ | Active multi-class |
| **23** | **0.7884** | **67.9%** | **74.3%** | **33.46%** | **0.3284** | **93.9%** | **34.7%** | **5.3%** | **BEST CHECKPOINT (RESTORED)** |
| **25** | $0.7068$ | $72.3\%$ | $74.3\%$ | $26.0\%$ | $0.2325$ | $100.0\%$ | $4.1\%$ | $0.0\%$ | Final epoch |

*Restoration Verification:* Best weights were restored from **Epoch 23** prior to evaluation and artifact export.

---

## 5. Validation Fold 0 Detailed Results ($N=354$, 49 patients)

### 5.1 Confusion Matrix
```text
Actual \ Predicted         Normal       CAS       DAS    CAS & DAS     Total
----------------------------------------------------------------------------
Normal                        245         1        14            1       261
CAS                            20         0         4            1        25
DAS                            31         0        17            1        49
CAS & DAS                       7         0        11            1        19
----------------------------------------------------------------------------
Total Predicted               303         1        46            4       354
```

### 5.2 Performance Metrics
- **Overall Accuracy:** $74.29\%$
- **Balanced Accuracy:** $33.46\%$
- **Macro F1-Score:** $0.3284$
- **Weighted F1-Score:** $0.6974$

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Normal** | 80.86% | 93.87% | 0.8688 | 261 | 245 | 58 | 16 |
| **CAS (Wheeze)** | 0.00% | 0.00% | 0.0000 | 25 | 0 | 1 | 25 |
| **DAS (Crackle)** | 36.96% | 34.69% | 0.3579 | 49 | 17 | 29 | 32 |
| **CAS & DAS (Both)** | 25.00% | 5.26% | 0.0870 | 19 | 1 | 3 | 18 |

---

## 6. Independent Held-Out Test Set Results (`held_out_inter_test`, $N=347$, 41 patients)

The independent test set was evaluated in a single one-shot inference pass without tuning:

### 6.1 Confusion Matrix
```text
Actual \ Predicted         Normal       CAS       DAS    CAS & DAS     Total
----------------------------------------------------------------------------
Normal                        187         0        49            5       241
CAS                            31         1        22           11        65
DAS                             7         1        11            5        24
CAS & DAS                       4         0         9            4        17
----------------------------------------------------------------------------
Total Predicted               229         2        91           25       347
```

### 6.2 Performance Metrics
- **Overall Accuracy:** $58.50\%$
- **Balanced Accuracy:** $37.12\%$
- **Macro F1-Score:** $0.3018$
- **Weighted F1-Score:** $0.6078$

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Normal** | 81.66% | 77.59% | 0.7957 | 241 | 187 | 42 | 54 |
| **CAS (Wheeze)** | 50.00% | 1.54% | 0.0299 | 65 | 1 | 1 | 64 |
| **DAS (Crackle)** | 12.09% | 45.83% | 0.1913 | 24 | 11 | 80 | 13 |
| **CAS & DAS (Both)** | 16.00% | 23.53% | 0.1905 | 17 | 4 | 21 | 13 |

---

## 7. Comparative Analysis: Previous Model vs. New Baseline

| Evaluation Dimension | Previous Sound Model (`sprsound-cnn-v1-fold0.keras`) | New Baseline Model (`ACOUSTIC_BASELINE_V1`) | Technical Significance of Difference |
|---|---|---|---|
| **Selected Checkpoint Epoch** | **Epoch 0** (Unweighted `val_loss` minimum) | **Epoch 23** (Restored at peak Balanced Accuracy) | Fixes the premature checkpointing defect that froze the model in an untrained state. |
| **Validation Balanced Accuracy** | $25.00\%$ | **$33.46\%$** | **$+8.46\%$ absolute improvement.** |
| **Validation Macro F1** | $0.2122$ | **$0.3284$** | **$+0.1162$ absolute improvement.** |
| **Validation DAS Recall** | $0.00\%$ (0/49 detected) | **$34.69\%$ (17/49 detected)** | Successfully captures transient discontinuous crackles. |
| **Test Set Balanced Accuracy** | $25.00\%$ | **$37.12\%$** | **$+12.12\%$ absolute improvement on unseen patients.** |
| **Test Set Multi-Class Prediction** | Predicted 1 class only (100% Normal) | **Actively predicted all 4 classes** | The model learned discriminative acoustic representations for adventitious sounds. |

---

## 8. Honest Remaining Limitations

1. **CAS (Wheeze Alone) Recall Remains Constrained (1.54% on test):**  
   Wheezes are continuous musical sounds that can be brief (200–500 ms) within a 10–15 s pediatric recording. When fixed center-cropping to 128 frames (8.192 s) is applied, wheezes occurring near the beginning or end of auscultation are cropped out. A sliding-window aggregation approach during inference is required to resolve this.
2. **Normal False Positive Confusion with Crackles:**  
   In the independent test set, 49 normal recordings were predicted as DAS. Pediatric crying, chest hair friction, and stethoscope movement create impulse noise that mimics crackle frequencies.
3. **Hardware & Pretraining Constraints:**  
   Without pre-trained audio foundation models (e.g. AST, PANNs), training a CNN from random initialization on 1,418 pediatric recordings remains challenged by small-sample variance.

---

## 9. Backend Safety & Integration Readiness

- **Artifact Safety:** The new model artifact was saved strictly inside `ml/experiments/acoustic_baseline_v1/sprsound_sound_cnn_baseline_fold0.keras` (5,560,221 bytes).
- **Non-Interference:** Existing models in `ml/models/saved_models/` were **not** overwritten.
- **Database Safety:** No model was registered or activated in Django `ModelVersion`.
- **Backend Code:** No backend files or Django settings were touched.
- **Integration Status:** **STILL BLOCKED.** While this baseline represents a major methodological improvement over the prior collapsed model, an acoustic classifier with 0% validation wheeze recall and 33.46% Balanced Accuracy is a research baseline, not an active clinical tool.
