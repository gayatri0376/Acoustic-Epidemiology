# Acoustic Sound Baseline Experiment (v1)
**Project:** Acoustic Epidemiology – Automated Classification of Pediatric Respiratory Diseases  
**Experiment ID:** `ACOUSTIC_BASELINE_V1`  
**Task:** `SPRSOUND_RECORD_4CLASS` (Normal, CAS, DAS, CAS & DAS)  
**Date:** 19 November 2026  
**Engineer:** Mohini (Machine Learning Modeling & CNN Lead)  

---

## 1. Objective
Overcome the majority-class collapse observed in prior runs (which achieved only 25.00% Balanced Accuracy by predicting 100% `Normal` at Epoch 0) by developing a scientifically defensible, class-balanced 4-block 2D-CNN baseline with targeted SpecAugment and non-collapsed Balanced Accuracy checkpoint selection.

---

## 2. Model & Checkpoint Provenance
- **Artifact File:** `sprsound_sound_cnn_baseline_fold0.keras` (5,560,221 bytes)
- **Selected Epoch:** Epoch 23 (out of 25 completed epochs)
- **Selection Metric:** `Balanced Accuracy` ($\frac{1}{4} \sum_{c=0}^3 \text{Recall}_c$) with single-class collapse rejection
- **Model Architecture:** 4-Block 2D-CNN (`PediatricAcousticBaselineCNN`) with Dual Pooling (GAP + GMP)
- **Input Shape:** `(None, 128, 128, 1)`
- **Output Shape:** `(None, 4)` Softmax
- **Target Conv Layer for Grad-CAM:** `last_conv_layer`

---

## 3. Results Summary

### Validation Fold 0 ($N=354$, 49 patients, 0 patient leakage)
- **Accuracy:** 74.29%
- **Balanced Accuracy:** **33.46%** (up from 25.00% baseline collapse)
- **Macro F1:** **0.3284** (up from 0.2122 baseline collapse)
- **Per-Class Recall:**
  - `Normal`: **93.87%** (245 / 261)
  - `DAS` (Crackles): **34.69%** (17 / 49 true positives, Precision = 36.96%, F1 = 0.3579)
  - `CAS & DAS` (Co-occurring): **5.26%** (1 / 19 true positives)
  - `CAS` (Wheezes): **0.00%** (0 / 25)

### Independent Held-Out Test Set (`held_out_inter_test`, $N=347$, 41 patients, 0 patient leakage)
- **Accuracy:** 58.50%
- **Balanced Accuracy:** **37.12%** (up from 25.00%)
- **Macro F1:** **0.3018** (up from 0.2049)
- **Per-Class Recall (All 4 classes predicted!):**
  - `Normal`: **77.59%** (187 / 241)
  - `DAS` (Crackles): **45.83%** (11 / 24 true positives, Precision = 12.09%, F1 = 0.1913)
  - `CAS & DAS` (Co-occurring): **23.53%** (4 / 17 true positives, Precision = 16.00%, F1 = 0.1905)
  - `CAS` (Wheezes): **1.54%** (1 / 65 true positives, Precision = 50.00%, F1 = 0.0299)

---

## 4. How to Reproduce Evaluation

```bash
cd /Users/gayatri/Downloads/Django/Acoustic-Epidemiology
MPLCONFIGDIR=/tmp KERAS_HOME=/tmp/.keras backend/.venv/bin/python -c "
import json
with open('ml/experiments/acoustic_baseline_v1/eval_val_fold0.json') as f:
    print('Validation Fold 0:', json.dumps(json.load(f), indent=2))
with open('ml/experiments/acoustic_baseline_v1/eval_held_out_test.json') as f:
    print('Held-out Test:', json.dumps(json.load(f), indent=2))
"
```
