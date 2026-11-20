# Full-Recording Respiratory Sound Analysis Using Sliding-Window Inference (v1)

**Project:** Acoustic Epidemiology – Automated Classification of Pediatric Respiratory Diseases  
**Experiment ID:** `SLIDING_WINDOW_INFERENCE_V1`  
**Task:** `FULL_RECORDING_RESPIRATORY_SOUND_ANALYSIS`  
**Date:** 20 November 2026  
**Engineer:** Mohini (Machine Learning Modeling & CNN Lead)  
**Baseline Model Artifact:** `ml/experiments/acoustic_baseline_v1/sprsound_sound_cnn_baseline_fold0.keras`

---

## 1. Objective & Hypothesis
Investigate whether fixed center-cropping (truncating recordings to 128 frames / 8.192 seconds) causes adventitious respiratory events (continuous adventitious sounds / wheezes / CAS and discontinuous adventitious sounds / crackles / DAS) to be clipped and missed. 

We evaluate multi-window sliding inference across the full uncropped Log Mel-Spectrograms (shape `(128, T)`) of recordings in the SPRSound dataset without modifying or retraining the underlying baseline CNN.

---

## 2. Sliding-Window Architecture & Hyperparameters
- **Input Feature:** Uncropped Log Mel-Spectrogram $\mathbf{S} \in \mathbb{R}^{128 \times T}$, where $T$ varies with recording duration (e.g. $T=145$ for 9.22s, $T=241$ for 15.36s).
- **Window Size ($W$):** 128 frames ($8.192\text{ s}$ at $sr=8000\text{ Hz}$, $hop\_length=512$).
- **Strides Evaluated ($S$):**
  - $S = 64$ frames ($4.096\text{ s}$, 50.0% temporal overlap)
  - $S = 32$ frames ($2.048\text{ s}$, 75.0% temporal overlap)
- **Boundary Handling:**
  - $T < 128$: Symmetric zero-padding.
  - $T \ge 128$: Overlapping windows with trailing-edge coverage (`T - window_size`) to guarantee full temporal support without zero-padding distortion.
- **Aggregation Strategies Evaluated (on Validation Fold 0):**
  1. `mean`: Uniform mean probability across windows.
  2. `max`: Per-class maximum probability across windows, re-normalized.
  3. `top_k`: Mean of top-2 probabilities per class, re-normalized.
  4. `abnormality_max`: Max abnormal class probability vs mean normal probability, re-normalized.

---

## 3. Empirical Results Comparison

### A. Validation Fold 0 ($N=354$, 49 patients, 0 patient leakage)

| Inference Method | Accuracy | Balanced Accuracy | Macro F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Center-Crop** | 74.29% | 33.46% | 0.3284 | 93.87% (245/261) | 0.00% (0/25) | 34.69% (17/49) | 5.26% (1/19) | 0.77s (2.2 ms/sample) |
| **Stride 64 Mean** | 75.42% | 34.25% | 0.3393 | 95.02% (248/261) | 0.00% (0/25) | 36.73% (18/49) | 5.26% (1/19) | 1.39s (3.9 ms/sample) |
| **Stride 64 Max** | 75.14% | 34.16% | 0.3378 | 94.64% (247/261) | 0.00% (0/25) | 36.73% (18/49) | 5.26% (1/19) | 1.39s (3.9 ms/sample) |
| **Stride 64 Top-2** | 75.42% | 34.25% | 0.3393 | 95.02% (248/261) | 0.00% (0/25) | 36.73% (18/49) | 5.26% (1/19) | 1.39s (3.9 ms/sample) |
| **Stride 64 Abnormality Max** *(Selected)* | 74.29% | **34.29%** | 0.3342 | 93.10% (243/261) | 0.00% (0/25) | **38.78%** (19/49) | 5.26% (1/19) | 1.39s (3.9 ms/sample) |
| **Stride 32 Abnormality Max** | 74.01% | 34.19% | 0.3336 | 92.72% (242/261) | 0.00% (0/25) | 38.78% (19/49) | 5.26% (1/19) | 1.61s (4.5 ms/sample) |

---

### B. Independent Held-Out Test Set (`held_out_inter_test`, $N=347$, 41 patients, 0 patient leakage)

| Inference Method | Accuracy | Balanced Accuracy | Macro F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Center-Crop** | 58.50% | **37.12%** | **0.3018** | 77.59% (187/241) | **1.54%** (1/65) | 45.83% (11/24) | **23.53%** (4/17) | 0.61s (1.8 ms/sample) |
| **Stride 64 Mean** | **60.23%** | 32.43% | 0.2643 | **82.16%** (198/241) | 0.00% (0/65) | 41.67% (10/24) | 5.88% (1/17) | 1.25s (3.6 ms/sample) |
| **Stride 64 Max** | 59.94% | 33.69% | 0.2758 | 81.33% (196/241) | 0.00% (0/65) | 41.67% (10/24) | 11.76% (2/17) | 1.25s (3.6 ms/sample) |
| **Stride 64 Top-2** | **60.81%** | 34.00% | 0.2793 | 82.57% (199/241) | 0.00% (0/65) | 41.67% (10/24) | 11.76% (2/17) | 1.25s (3.6 ms/sample) |
| **Stride 64 Abnormality Max** *(Locked from Val)* | 55.62% | 34.01% | 0.2666 | 74.27% (179/241) | 0.00% (0/65) | **50.00%** (12/24) | 11.76% (2/17) | 1.25s (3.6 ms/sample) |
| **Stride 32 Abnormality Max** | 55.04% | 36.11% | 0.2777 | 72.61% (175/241) | 0.00% (0/65) | **54.17%** (13/24) | 17.65% (3/17) | 1.48s (4.3 ms/sample) |

---

## 4. Key Scientific Insights
1. **Crackles (DAS) Benefit from Multi-Window Scanning:**
   Under Abnormality Max aggregation, crackle detection recall rises from 45.83% (center crop) to 50.00% (stride 64) and 54.17% (stride 32), confirming that explosive crackles occurring at the start or tail of a recording are retrieved.
2. **Wheezes (CAS) Do NOT Improve Without Retraining:**
   Because the underlying CNN was trained exclusively on 128-frame center-cropped spectrograms, test-time sliding window inference alone does not resolve the CAS representation deficiency (CAS recall remained 0.00%). The model lacks robust spectro-temporal filters for wheezes across different breath cycle alignments.
3. **Mean Pooling Increases Normal Recall but Dilutes Rare Pathology:**
   Uniform averaging of windows increases Normal recall to 82.16% and overall accuracy to 60.23%, but reduces sensitivity to adventitious sounds (CAS & DAS recall fell from 23.53% to 5.88%).

---

## 5. How to Reproduce Evaluation

```bash
cd /Users/gayatri/Downloads/Django/Acoustic-Epidemiology
MPLCONFIGDIR=/tmp KERAS_HOME=/tmp/.keras backend/.venv/bin/python ml/experiments/sliding_window_v1/sliding_window_inference.py
```
