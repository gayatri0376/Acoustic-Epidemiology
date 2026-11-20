# Comprehensive Experiment Report: Full-Recording Respiratory Sound Analysis Using Sliding-Window Inference

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Experiment ID:** `SLIDING_WINDOW_INFERENCE_V1`  
**Date:** 20 November 2026  
**Author / Lead:** Mohini (Machine Learning Modeling & CNN Lead)  
**Auditor / Integration Engineer:** Gayatri (Senior Django Backend & ML Integration Lead)  
**Model Artifact Evaluated:** `ml/experiments/acoustic_baseline_v1/sprsound_sound_cnn_baseline_fold0.keras`  
**Status:** Evaluation Completed — Scientific Findings Documented — Deployment Recommendation Finalized

---

## Executive Summary

Fixed center-cropping in audio classification truncates variable-duration stethoscopic recordings to a fixed frame length ($W = 128$ frames, corresponding to $8.192\text{ s}$ at $sr=8000\text{ Hz}, \text{hop\_length}=512$). For a standard 15.36-second pediatric respiratory recording ($T = 241$ frames), fixed center-cropping completely discards **46.9% of the acoustic signal** (frames 0–56 and 184–240), creating a structural hypothesis that adventitious sounds occurring at recording margins are systematically missed.

In this experiment, we designed, implemented, and benchmarked a deterministic **Sliding-Window Multi-Scale Inference Engine** (`ml/experiments/sliding_window_v1/sliding_window_inference.py`) across the complete uncropped Log Mel-Spectrograms ($\mathbb{R}^{128 \times T}$) of the SPRSound pediatric dataset. We evaluated multiple stride configurations ($S=64$ [50% overlap] and $S=32$ [75% overlap]) and four aggregation strategies (`mean`, `max`, `top_k`, `abnormality_max`) on **Validation Fold 0** ($N=354$, 49 patients, 0 patient leakage) and benchmarked the locked optimal strategy against the **Independent Held-Out Test Set** (`held_out_inter_test`, $N=347$, 41 patients, 0 patient leakage).

### Key Empirical Findings:
1. **Crackles (DAS) Benefit from Multi-Window Scanning:**
   Under peak-sensitive `abnormality_max` aggregation, crackle recall increases from **45.83%** (baseline center-crop) to **50.00%** ($S=64$) and **54.17%** ($S=32$), demonstrating that episodic explosive crackles at recording boundaries are successfully recovered.
2. **Wheezes (CAS) Do Not Improve Without Retraining:**
   Because the underlying CNN was trained exclusively on 128-frame center-cropped slices, test-time sliding-window inference yields **0.00% recall** for wheezes across all aggregation methods on the independent test set (compared to 1.54% / 1 sample in center-crop). Sliding-window inference cannot compensate for an underlying lack of generalized spectral filters.
3. **Mean Probability Pooling Dilutes Pathological Signals:**
   Uniform averaging over windows boosts overall test accuracy from 58.50% to 60.23% by increasing Normal recall to 82.16%, but severely suppresses co-occurring adventitious sounds (`CAS & DAS` recall drops from 23.53% to 5.88%).
4. **Backend Integration Recommendation:**
   **BLOCKED for Clinical Production Deployment.** Neither center-crop nor sliding-window inference achieves clinically reliable diagnostic thresholds (Balanced Accuracy remains $\le 37.12\%$). While sliding-window inference is computationally viable (3.6 ms/sample in batch, 12 ms single-recording latency), it must NOT be deployed to production until an acoustic model is trained end-to-end with multi-window or variable-length augmentation.

---

## 1. Audit of Existing Preprocessing and Center-Cropping

### 1.1 Audio Feature Representation
The acoustic feature extraction pipeline produces Log Mel-Spectrograms from raw audio under the following parameters:
- **Audio Sample Rate ($f_s$):** $8,000\text{ Hz}$ mono PCM.
- **Short-Time Fourier Transform (STFT):**
  - FFT window length ($n\_fft$): $1,024$ samples ($128.0\text{ ms}$).
  - Hop length ($R$): $512$ samples ($64.0\text{ ms}$, 50% window overlap).
  - Window function: Periodic Hann window.
- **Mel Filterbank:**
  - Number of Mel bands ($F$): $128$ filters.
  - Frequency range: $[50\text{ Hz}, 2,000\text{ Hz}]$ (pediatric respiratory bandwidth capturing both vesicular sounds and high-pitched wheezes).
- **Decibel Scaling & Normalization:**
  - Log compression: $S_{\text{dB}} = 10 \cdot \log_{10}(\max(S, 10^{-10}))$.
  - Min-Max normalization per spectrogram: $[0.0, 1.0]$.

### 1.2 Fixed Center-Cropping Mechanics
The standard training tensor generation extracts a fixed temporal window:
$$\text{tensor} = \mathbf{S}[:, \text{start} : \text{start} + 128], \quad \text{where } \text{start} = \max\left(0, \left\lfloor \frac{T - 128}{2} \right\rfloor\right)$$

#### Truncation Impact across SPRSound Durations:
| Recording Duration | Raw Temporal Frames ($T$) | Preserved Frames ($W=128$) | Discarded Frames ($T-W$) | Discarded Duration | Signal Loss % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **8.192 seconds** | 128 | 128 | 0 | 0.00 s | **0.0%** |
| **9.216 seconds** | 145 | 128 | 17 | 1.09 s | **11.7%** |
| **14.720 seconds** | 231 | 128 | 103 | 6.59 s | **44.6%** |
| **15.360 seconds** | 241 | 128 | 113 | 7.23 s | **46.9%** |

In 15.36s recordings (the dominant duration in SPRSound), the first **3.58 seconds** (frames 0–55) and last **3.65 seconds** (frames 184–240) are completely lost.

---

## 2. Sliding-Window Architecture and Aggregation Methods

### 2.1 Window Extraction Algorithm
For any spectrogram $\mathbf{S} \in \mathbb{R}^{128 \times T}$:
1. **Short Recordings ($T < 128$):** Symmetric zero-padding along the temporal dimension:
   $$\text{pad\_left} = \lfloor(128 - T)/2\rfloor, \quad \text{pad\_right} = 128 - T - \text{pad\_left}$$
   Yields $K = 1$ window.
2. **Exact Recordings ($T = 128$):** Single window $\mathbf{S}$ ($K = 1$).
3. **Long Recordings ($T > 128$):** Sliding windows with stride $S$:
   $$\text{starts} = \{0, S, 2S, \dots, kS\} \quad \text{such that } kS \le T - 128$$
   If $\max(\text{starts}) + 128 < T$, the trailing slice $[T - 128 : T]$ is appended to the window set. This guarantees **100% temporal coverage** of the entire recording without introducing zero-padding artifacts.

For $S = 64$ ($50\%$ overlap), recordings have an average of **2.20 windows** ($\min = 1, \max = 3$).  
For $S = 32$ ($75\%$ overlap), recordings have an average of **2.61 windows** ($\min = 1, \max = 5$).

### 2.2 Mathematical Formulation of Aggregation Strategies

Let $P \in \mathbb{R}^{K \times 4}$ denote the model's Softmax output across $K$ windows for a recording, where $P_{k, c}$ is the probability of class $c$ in window $k$:

1. **Mean Probability Aggregation (`mean`):**
   $$\hat{P}_c = \frac{1}{K} \sum_{k=1}^K P_{k, c}, \quad \hat{y} = \arg\max_{c} \hat{P}_c$$
   *Property:* Smooths predictions, highly robust to random noise, but heavily penalizes transient adventitious sounds by averaging them with non-pathological respiratory phases.

2. **Max Probability Aggregation (`max`):**
   $$\hat{P}_c^* = \max_{k \in \{1,\dots,K\}} P_{k, c}, \quad \hat{P}_c = \frac{\hat{P}_c^*}{\sum_{j=0}^3 \hat{P}_j^*}, \quad \hat{y} = \arg\max_{c} \hat{P}_c$$
   *Property:* Highly sensitive to the single strongest event observed in any window.

3. **Top-$k$ Mean Aggregation (`top_k`, $m=2$):**
   $$\hat{P}_c^* = \frac{1}{\min(m, K)} \sum_{p \in \text{Top-}m(P_{:, c})} p, \quad \hat{P}_c = \frac{\hat{P}_c^*}{\sum_{j=0}^3 \hat{P}_j^*}$$
   *Property:* Balances peak sensitivity against single-window outlier hallucinations.

4. **Abnormality-Weighted Max Aggregation (`abnormality_max`):**
   $$\hat{P}_0^* = \frac{1}{K}\sum_{k=1}^K P_{k, 0} \quad (\text{Normal}), \qquad \hat{P}_c^* = \max_{k} P_{k, c} \quad (c \in \{1, 2, 3\} \text{ Abnormal})$$
   $$\hat{P}_c = \frac{\hat{P}_c^*}{\sum_{j=0}^3 \hat{P}_j^*}, \quad \hat{y} = \arg\max_{c} \hat{P}_c$$
   *Property:* Clinically motivated heuristic: adventitious sounds (crackles/wheezes) are transient local events, while normal vesicular breathing is continuous across all windows.

---

## 3. Empirical Results: Validation Fold 0 Benchmark

- **Sample Size:** $N = 354$ recordings (49 patients, 0 patient leakage).
- **Class Distribution:** Normal = 261, CAS = 25, DAS = 49, CAS & DAS = 19.

### Performance Comparison across All Configurations:

| Configuration | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Inference Time (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Center-Crop** | 74.29% | 33.46% | 0.3284 | 0.6948 | 93.87% (245) | 0.00% (0) | 34.69% (17) | 5.26% (1) | 2.2 ms |
| `stride_64_mean` | 75.42% | 34.25% | 0.3393 | 0.7061 | 95.02% (248) | 0.00% (0) | 36.73% (18) | 5.26% (1) | 3.9 ms |
| `stride_64_max` | 75.14% | 34.16% | 0.3378 | 0.7042 | 94.64% (247) | 0.00% (0) | 36.73% (18) | 5.26% (1) | 3.9 ms |
| `stride_64_top_k` | 75.42% | 34.25% | 0.3393 | 0.7061 | 95.02% (248) | 0.00% (0) | 36.73% (18) | 5.26% (1) | 3.9 ms |
| **`stride_64_abnormality_max`** *(Best Val)* | 74.29% | **34.29%** | 0.3342 | 0.7001 | 93.10% (243) | 0.00% (0) | **38.78%** (19) | 5.26% (1) | 3.9 ms |
| `stride_32_mean` | 75.14% | 34.16% | 0.3388 | 0.7045 | 94.64% (247) | 0.00% (0) | 36.73% (18) | 5.26% (1) | 4.5 ms |
| `stride_32_max` | 74.86% | 34.06% | 0.3373 | 0.7027 | 94.25% (246) | 0.00% (0) | 36.73% (18) | 5.26% (1) | 4.5 ms |
| `stride_32_top_k` | 75.14% | 34.16% | 0.3378 | 0.7043 | 94.64% (247) | 0.00% (0) | 36.73% (18) | 5.26% (1) | 4.5 ms |
| `stride_32_abnormality_max` | 74.01% | 34.19% | 0.3336 | 0.6989 | 92.72% (242) | 0.00% (0) | **38.78%** (19) | 5.26% (1) | 4.5 ms |

### Locked Optimal Configuration on Validation Data:
`stride_64_abnormality_max` achieved the highest Balanced Accuracy (**34.29%**, +0.83% over baseline center-crop) and improved DAS crackle detection from 17 to 19 true positives without excessive computational cost ($S=64$ requires only 780 window evaluations vs 924 for $S=32$).

---

## 4. Empirical Results: Independent Held-Out Benchmark (`held_out_inter_test`)

- **Sample Size:** $N = 347$ recordings (41 patients, 0 patient leakage).
- **Class Distribution:** Normal = 241, CAS = 65, DAS = 24, CAS & DAS = 17.

### Independent Benchmark Comparison:

| Configuration | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Inference Time (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Center-Crop** | 58.50% | **37.12%** | **0.3018** | 0.5808 | 77.59% (187) | **1.54%** (1) | 45.83% (11) | **23.53%** (4) | 1.8 ms |
| `stride_64_mean` | **60.23%** | 32.43% | 0.2643 | 0.5861 | **82.16%** (198) | 0.00% (0) | 41.67% (10) | 5.88% (1) | 3.6 ms |
| `stride_64_max` | 59.94% | 33.69% | 0.2758 | 0.5888 | 81.33% (196) | 0.00% (0) | 41.67% (10) | 11.76% (2) | 3.6 ms |
| `stride_64_top_k` | **60.81%** | 34.00% | 0.2793 | 0.5960 | **82.57%** (199) | 0.00% (0) | 41.67% (10) | 11.76% (2) | 3.6 ms |
| **`stride_64_abnormality_max`** *(Locked)* | 55.62% | 34.01% | 0.2666 | 0.5647 | 74.27% (179) | 0.00% (0) | **50.00%** (12) | 11.76% (2) | 3.6 ms |
| `stride_32_mean` | 60.52% | 33.90% | 0.2786 | 0.5925 | 82.16% (198) | 0.00% (0) | 41.67% (10) | 11.76% (2) | 4.3 ms |
| `stride_32_max` | 60.52% | 34.33% | 0.2849 | 0.5947 | 82.16% (198) | 0.00% (0) | 37.50% (9) | 17.65% (3) | 4.3 ms |
| `stride_32_top_k` | 60.81% | 35.37% | 0.2908 | 0.5995 | 82.16% (198) | 0.00% (0) | 41.67% (10) | 17.65% (3) | 4.3 ms |
| `stride_32_abnormality_max` | 55.04% | 36.11% | 0.2777 | 0.5627 | 72.61% (175) | 0.00% (0) | **54.17%** (13) | 17.65% (3) | 4.3 ms |

### Detailed Confusion Matrices (`cm_held_out_test.csv`):

#### 1. Baseline Center-Crop ($N=347$):
```text
               Predicted
True Class     Normal    CAS    DAS    CAS & DAS    Total
Normal          187        0     49        5          241
CAS              31        1     22       11           65
DAS               7        1     11        5           24
CAS & DAS         4        0      9        4           17
Total           229        2     91       25          347
```

#### 2. Locked Sliding-Window `stride_64_abnormality_max` ($N=347$):
```text
               Predicted
True Class     Normal    CAS    DAS    CAS & DAS    Total
Normal          179        0     57        5          241
CAS              31        0     25        9           65
DAS               7        0     12        5           24
CAS & DAS         4        0     11        2           17
Total           221        0    105       21          347
```

---

## 5. Detailed Scientific Comparison: Center-Crop vs. Sliding-Window

### 5.1 The Divergent Behavior of Crackles (DAS) vs. Wheezes (CAS)
A critical insight of this experiment is why DAS and CAS responded in completely opposite directions to multi-window scanning:
- **Discontinuous Adventitious Sounds (DAS / Crackles):**
  Crackles are explosive, brief acoustic transients ($\tau < 20\text{ ms}$) spanning wide frequency bands. When crackles occur in an early or late respiratory cycle of a 15-second recording, fixed center-cropping completely cuts them off. Under `abnormality_max`, sliding windows capture these peripheral crackle bursts, increasing true positive detection from 11 to 12 ($S=64$) and 13 ($S=32$).
- **Continuous Adventitious Sounds (CAS / Wheezes):**
  Wheezes are continuous musical tones ($\tau > 100\text{ ms}$) with distinct harmonic structures. However, in our dataset, the CNN was trained exclusively on 128-frame slices centered on the recording midpoint. When sliding windows are evaluated across non-centered breath phases, the model's feature activations for CAS fall below detection thresholds. Without exposure to diverse temporal offsets during training, multi-window inference failed to identify a single wheeze on the test set (0/65).

### 5.2 The Mean Aggregation Dilemma
Uniformly averaging window Softmax vectors acts as a low-pass filter over time. In a 15-second recording comprising 3 respiratory cycles, an abnormal finding (such as 2 crackles during inspiration in cycle 3) will produce high abnormal probability in window 3 ($P_{\text{DAS}} \approx 0.65$), but windows 1 and 2 will reflect normal vesicular sounds ($P_{\text{Normal}} \approx 0.85$). 
Averaging yields:
$$\bar{P}_{\text{Normal}} = \frac{0.85 + 0.85 + 0.25}{3} = 0.65, \quad \bar{P}_{\text{DAS}} = \frac{0.05 + 0.05 + 0.65}{3} = 0.25$$
Consequently, the model defaults to `Normal`. While this increases raw accuracy by eliminating false positives, it severely harms diagnostic utility for rare disorders.

---

## 6. Computational Latency and Resource Profile

Benchmarked on Apple M3 Max (`mps` backend / CPU fallback execution via Python 3.13):

| Metric | Fixed Center-Crop | Sliding Window ($S=64$) | Sliding Window ($S=32$) |
| :--- | :---: | :---: | :---: |
| **Number of Tensors Evaluated per Recording** | 1 tensor | 2.20 tensors (avg) | 2.61 tensors (avg) |
| **Batch Inference Latency (347 samples)** | 0.61 seconds | 1.25 seconds | 1.48 seconds |
| **Per-Sample Inference Time (Batch)** | 1.8 ms | 3.6 ms | 4.3 ms |
| **Simulated Single-Sample Online Latency** | 5.2 ms | 11.8 ms | 14.1 ms |
| **Memory Footprint per Recording** | 0.5 MB | 1.1 MB | 1.3 MB |
| **Backend Integration Complexity** | Low ($O(1)$ slice) | Moderate ($O(K)$ slice + agg) | Moderate ($O(K)$ slice + agg) |

Both approaches are well within real-time limits ($< 50\text{ ms}$ threshold for Django web requests). Computational latency is NOT a blocking factor.

---

## 7. Final Decisions and Direct Answers (Tasks A – I)

### A. Is fixed center-cropping responsible for the low CAS recall?
**Partially, but NOT solely.** While fixed center-cropping physically discards 46.9% of audio, test-time sliding window inference across the entire recording still resulted in 0.00% CAS recall on the independent test set. The primary bottleneck is the lack of discriminative wheeze representations in the baseline CNN weights, caused by severe training imbalance (101 CAS vs 1042 Normal) and training exclusively on center-cropped windows.

### B. Did sliding-window inference improve adventitious sound detection?
**Yes for crackles (DAS), No for wheezes (CAS).** Under `abnormality_max`, DAS recall improved from 45.83% to 50.00% ($S=64$) and 54.17% ($S=32$). However, CAS recall remained at 0.00%.

### C. Which aggregation method performed best?
On validation data, `abnormality_max` performed best (Balanced Accuracy 34.29%). For raw accuracy and Normal recall, `mean` performed best (60.23% accuracy, 82.16% Normal recall), but suppressed abnormal sensitivity.

### D. What stride is optimal?
$S = 64$ frames (50% overlap) provides the optimal trade-off between computational cost and temporal resolution. $S = 32$ offered marginal DAS gains (from 50% to 54.17%) but increased window count and computation by 18.5%.

### E. What is the impact on inference latency?
Inference time increased from 1.8 ms to 3.6 ms per sample in batch mode, and is estimated at ~12 ms for individual online HTTP uploads. This latency overhead is trivial and well within Django request timeouts.

### F. Does sliding-window inference resolve the baseline performance limitations?
**No.** Balanced Accuracy on the independent test set was 34.01% under sliding-window inference vs. 37.12% under center-crop. The baseline remains an experimental research model and does not meet clinical standards.

### G. Should sliding-window inference be deployed to the Django backend?
**NO.** Backend model activation remains strictly **BLOCKED**. Neither the center-crop baseline nor the sliding-window variant has achieved acceptable diagnostic reliability for pediatric auscultation.

### H. What are the prerequisites before any sliding-window deployment?
1. The CNN must be retrained using random temporal crops and multi-scale sliding-window augmentation so that convolutional kernels learn phase-invariant adventitious sound features.
2. Transfer learning from large-scale pre-trained respiratory audio encoders (e.g., PANNs, Audio Spectrogram Transformer, or ICBHI pretrained backbones) must be explored to provide rich frequency representations for wheezes.
3. The model must demonstrate $> 65\%$ Balanced Accuracy and $> 50\%$ recall across all four classes on independent test data before clinical deployment is considered.

### I. Exact Artifact Deliverables
1. `ml/experiments/sliding_window_v1/sliding_window_inference.py` (Fully functional inference engine)
2. `ml/experiments/sliding_window_v1/config.json` (Sliding-window experiment configuration)
3. `ml/experiments/sliding_window_v1/eval_val_fold0.json` (Complete validation metrics for all 8 configurations)
4. `ml/experiments/sliding_window_v1/eval_held_out_test.json` (Complete test metrics for all configurations)
5. `ml/experiments/sliding_window_v1/cm_val_fold0.csv` (Validation confusion matrix)
6. `ml/experiments/sliding_window_v1/cm_held_out_test.csv` (Independent test confusion matrix)
7. `ml/experiments/sliding_window_v1/README.md` (Reproduction guide and execution commands)
8. `docs/SLIDING_WINDOW_EXPERIMENT_REPORT.md` (This document)
