# Final Research Figure Checklist & Publication Specifications

**Project:** Acoustic Epidemiology: Patient-Independent Classification of Pediatric Respiratory Sounds Using CNN-Based Time-Frequency Analysis  
**Lead:** Mohini (Machine Learning Modeling & CNN Lead)  
**Reviewer:** Gayatri (Senior ML Integration Lead)  
**Date:** 20 November 2026  

---

## Overview

This document specifies the complete figure inventory, layout guidelines, data sources, and caption texts for the IEEE research paper draft. All visual artifacts are derived exclusively from verified experimental data and actual pipeline implementations.

---

## Figure Checklist & Specifications

### Figure 1: End-to-End System Architecture
- **Type:** System schematic flowchart.
- **Components:** Audio Acquisition $\rightarrow$ Digital Audio Validation $\rightarrow$ 50–2000 Hz Bandpass Filtering $\rightarrow$ Log Mel-Spectrogram Extraction $\rightarrow$ Dual-Branch Evaluation (Center-Crop Baseline vs. Multi-Scale Sliding-Window Engine) $\rightarrow$ Dual Pooling 2D-CNN $\rightarrow$ Softmax Class Distribution.
- **File Reference:** `docs/IEEE_RESEARCH_PAPER_DRAFT.md` (Section IV).
- **Caption:** *"Fig. 1. End-to-end Acoustic Epidemiology pipeline. Audio is digitized at 8,000 Hz, filtered via zero-phase Butterworth bandpass, transformed into 128-band Log Mel-Spectrograms, and evaluated via a 4-block 2D-CNN incorporating concatenated Dual Pooling (Global Average Pooling and Global Max Pooling)."*
- **Status:** **Ready for TikZ / Vector Illustrator rendering.**

---

### Figure 2: Time-Frequency Audio Preprocessing Pipeline
- **Type:** Multi-panel waveform and spectrogram representation.
- **Panels:**
  - Panel (a): Raw 8,000 Hz acoustic waveform from pediatric chest auscultation (15.36s).
  - Panel (b): Bandpass filtered signal ([50 Hz, 2000 Hz]) attenuating sub-50 Hz motion artifacts.
  - Panel (c): 128-band Log Mel-Spectrogram showing respiratory cycles and acoustic energy distributions.
  - Panel (d): Min-max normalized representation $[0.0, 1.0]$.
- **Data Source:** `ml/features/extracted/records/held_out_inter_test_9921_14.4_0_p1_1720_record.npz`.
- **Caption:** *"Fig. 2. Time-frequency feature conditioning on a 14.4-second pediatric recording. Butterworth filtering removes baseline drift, followed by STFT and 128-channel Mel filterbank decibel scaling."*
- **Status:** **Source numpy spectrogram data verified on disk.**

---

### Figure 3: Convolutional Neural Network with Dual Pooling
- **Type:** Detailed neural network architectural diagram.
- **Layer Progression:**
  - Input: `(128, 128, 1)`
  - Block 1: Conv2D(32, $3\times3$) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool($2\times2$) $\rightarrow$ Drop(0.20) $\rightarrow$ `(64, 64, 32)`
  - Block 2: Conv2D(64, $3\times3$) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool($2\times2$) $\rightarrow$ Drop(0.20) $\rightarrow$ `(32, 32, 64)`
  - Block 3: Conv2D(128, $3\times3$) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool($2\times2$) $\rightarrow$ Drop(0.25) $\rightarrow$ `(16, 16, 128)`
  - Block 4: Conv2D(256, $3\times3$, `last_conv_layer`) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool($2\times2$) $\rightarrow$ `(8, 8, 256)`
  - Dual Pooling Layer: Parallel Global Average Pooling `(256)` + Global Max Pooling `(256)` $\rightarrow$ Concat `(512)`
  - Head: Dense(128, ReLU, $L_2=10^{-4}$) $\rightarrow$ Drop(0.30) $\rightarrow$ Dense(4, Softmax).
- **Caption:** *"Fig. 3. Detailed architecture of PediatricAcousticBaselineCNN. The concatenation of Global Average Pooling (GAP) and Global Max Pooling (GMP) allows the network to preserve both sustained harmonic features (wheezes) and sharp impulsive transient peaks (crackles)."*
- **Status:** **Architecture verified from `ml/experiments/acoustic_baseline_v1/sprsound_sound_cnn_baseline_fold0.keras`.**

---

### Figure 4: Patient-Disjoint Dataset Partitioning Methodology
- **Type:** Stratified group partition diagram.
- **Elements:**
  - 2,683 Manifest Records $\rightarrow$ Exclude 70 Poor Quality recordings.
  - 2,613 Valid Auscultations across 356 Patients.
  - Stratified Group 5-Fold Partitioning on `patient_id`:
    - Pool: 251 Patients, 1,826 Records.
    - Folds 1–4: 202 Patients, 1,418 Records (Unaugmented Training).
    - Fold 0: 49 Patients, 354 Records (Validation Set).
  - Independent 2022 Inter-Subject Test Set: 41 Patients, 347 Records (**Strictly 0 Patient Overlap**).
- **Caption:** *"Fig. 4. Patient-independent partitioning scheme. Patients are grouped via StratifiedGroupKFold to guarantee zero subject leakage between training, validation, and independent test partitions."*
- **Status:** **Audit matrix verified in `ml/experiments/final_ml_audit/patient_leakage_audit.json`.**

---

### Figure 5: Confusion Matrices on Independent Held-Out Benchmark ($N=347$)
- **Type:** Dual side-by-side heatmaps.
- **Panel (a): Fixed Center-Crop Baseline ($N=347$):**
  - Normal: 187 TP / 241 (77.59% recall)
  - CAS: 1 TP / 65 (1.54% recall)
  - DAS: 11 TP / 24 (45.83% recall)
  - CAS & DAS: 4 TP / 17 (23.53% recall)
- **Panel (b): Sliding-Window Abnormality-Max ($S=64$, $N=347$):**
  - Normal: 179 TP / 241 (74.27% recall)
  - CAS: 0 TP / 65 (0.00% recall)
  - DAS: 12 TP / 24 (50.00% recall) — **+4.17% gain in crackle sensitivity**
  - CAS & DAS: 2 TP / 17 (11.76% recall)
- **Source Files:** `ml/experiments/acoustic_baseline_v1/cm_held_out_test.csv` and `ml/experiments/sliding_window_v1/cm_held_out_test.csv`.
- **Caption:** *"Fig. 5. Confusion matrices on the independent test set (41 unseen patients). Panel (a) shows the center-crop baseline. Panel (b) shows sliding-window inference with abnormality-weighted max aggregation, demonstrating increased true positive detection for crackles (DAS) alongside persistent wheeze (CAS) suppression."*
- **Status:** **CSV source files verified.**

---

### Figure 6: Sliding-Window Multi-Scale Slicing & Boundary Coverage
- **Type:** Temporal slicing and aggregation schematic.
- **Visuals:**
  - Full Mel-Spectrogram ($T = 241$ frames, 15.36s).
  - Center-Crop: Shows discarded margins (frames 0–55 and 184–240 shaded red).
  - Sliding Window ($W=128, S=64$): Window 1 [0:128], Window 2 [64:192], Trailing Window [113:241].
  - Aggregation: Softmax probability matrix $P \in \mathbb{R}^{3 \times 4} \rightarrow$ Abnormality-weighted aggregation vector.
- **Caption:** *"Fig. 6. Sliding-window multi-scale inference mechanics. Overlapping 128-frame windows with a trailing-edge slice ensure 100% temporal coverage of 15.36-second recordings without zero-padding distortion."*
- **Status:** **Algorithm verified in `ml/experiments/sliding_window_v1/sliding_window_inference.py`.**

---

### Figure 7: Performance Trade-Off: Crackle Gain vs. Wheeze Deficit
- **Type:** Clustered bar chart comparing Class Recall across models on Independent Test.
- **Metrics Plotted:**
  - Normal Recall: Center-Crop (77.59%) vs. SW Mean (82.16%) vs. SW AbnMax (74.27%).
  - DAS Crackle Recall: Center-Crop (45.83%) vs. SW Mean (41.67%) vs. SW AbnMax (50.00%) vs. SW-32 (54.17%).
  - CAS Wheeze Recall: Center-Crop (1.54%) vs. All SW Methods (0.00%).
  - CAS & DAS Recall: Center-Crop (23.53%) vs. SW Mean (5.88%) vs. SW AbnMax (11.76%).
- **Source File:** `ml/experiments/final_ml_audit/experiment_metrics_comparison.csv`.
- **Caption:** *"Fig. 7. Class-wise recall across inference configurations on independent test data. Crackle (DAS) detection improves under peak-sensitive aggregation, whereas uniform mean pooling increases Normal recall at the expense of adventitious sensitivity."*
- **Status:** **Data verified.**

---

### Figure 8: Grad-CAM Saliency Maps on Adventitious Sounds
- **Type:** Dual-spectrogram Grad-CAM overlay comparison.
- **Examples:**
  - Panel (a): True Positive DAS (Crackle) sample showing localized high-energy gradient attribution in the 200–800 Hz range corresponding to explosive acoustic transients.
  - Panel (b): False Negative CAS (Wheeze) sample showing diffuse, unlocalized gradients across background low frequencies, illustrating the network's failure to capture high-pitched musical harmonics.
- **Data Source:** `backend/media/explanations/*_gradcam.png`.
- **Target Layer:** Layer `last_conv_layer` (Block 4 Conv2D).
- **Caption:** *"Fig. 8. Grad-CAM saliency heatmaps generated from the last convolutional layer. The model successfully focuses on explosive crackle transients (a), but displays diffuse, unlocalized activations on wheezes (b), confirming the representation deficit for continuous adventitious sounds."*
- **Status:** **Source Grad-CAM artifacts verified on disk.**
