# Patient-Independent Classification of Pediatric Respiratory Sounds Using CNN-Based Time-Frequency Analysis: An Experimental Study on SPRSound

**Authors:**  
Mohini$^{1}$, Gayatri$^{2}$  
$^{1}$Department of Computer Science & Machine Learning Engineering  
$^{2}$Department of Software Systems & Backend Integration Engineering  
*Acoustic Epidemiology Research Group*  

---

## Abstract

Automated analysis of pediatric respiratory sounds holds significant potential for objective, non-invasive pulmonary assessment. However, existing deep-learning approaches frequently suffer from patient-level data leakage, evaluate adult cohorts that do not reflect pediatric chest wall mechanics, or attempt to predict systemic hospital discharge diagnoses directly from localized single-channel auscultation recordings. In this study, we present **Acoustic Epidemiology**, a patient-disjoint, four-class acoustic sound classification framework evaluated on the SJTU Paediatric Respiratory Sound Database (SPRSound). 

We first demonstrate that attempting to classify three clinical disease categories (*Healthy Control*, *Pneumonia*, and *Wheezing Disorders*) from localized stethoscopic recordings results in catastrophic model collapse (yielding 0.00% recall on wheezing disorders) because approximately 72% of thoracic recordings from diseased pediatric patients contain acoustically normal vesicular breath sounds. Reframing the task as four-class acoustic sound classification (*Normal*, *Continuous Adventitious Sounds [CAS / Wheezes]*, *Discontinuous Adventitious Sounds [DAS / Crackles]*, and *CAS & DAS*), we engineer a 4-block 2D Convolutional Neural Network (CNN) incorporating **Dual Pooling** (concatenated Global Average Pooling and Global Max Pooling) and targeted SpecAugment. Checkpoint selection was governed strictly by validation Balanced Accuracy.

Across an independent held-out benchmark of 347 recordings from 41 unseen patients (strictly 0 patient overlap), the baseline model achieved **58.50% Accuracy**, **37.12% Balanced Accuracy**, and **0.3018 Macro F1-Score**, with per-class recalls of 77.59% (*Normal*), 45.83% (*DAS*), 23.53% (*CAS & DAS*), and 1.54% (*CAS*). To investigate whether fixed center-cropping (8.192 seconds / 128 frames) discards critical adventitious events, we conducted a multi-scale sliding-window ablation across complete uncropped recordings. Locked leak-free on validation data, sliding-window inference with abnormality-weighted max aggregation increased crackle (*DAS*) recall from 45.83% to **50.00%** ($S=64$) and **54.17%** ($S=32$). However, wheeze (*CAS*) recall remained at 0.00%, resulting in an overall test Balanced Accuracy of 34.01%. Our findings demonstrate that while multi-window scanning recovers transient boundary crackles, post-hoc test-time inference cannot overcome representation deficits caused by severe training class imbalance (10.3:1) and weak recording-level annotations. The framework serves as an open, reproducible research baseline; we explicitly conclude that neither center-crop nor sliding-window models meet clinical diagnostic thresholds.

**Keywords:** Respiratory Sound Analysis, SPRSound, Convolutional Neural Networks, Mel Spectrogram, Pediatric Auscultation, Patient-Independent Evaluation, Weak Labels, Dual Pooling, Sliding-Window Inference.

---

## I. Introduction

Pediatric respiratory infections represent one of the primary causes of acute morbidity and mortality in infants and young children globally. In resource-constrained settings, delayed recognition of adventitious respiratory sounds—such as continuous wheezes in reactive airway disease and discontinuous crackles in lower respiratory infections—frequently leads to severe clinical deterioration. Conventional acoustic auscultation via acoustic stethoscopes is subjective, suffers from pronounced inter-observer variability, and requires specialized clinical training. Digital stethoscopes coupled with automated acoustic classification algorithms have emerged as a promising technology for objective respiratory triage.

Despite extensive literature on computer-aided auscultation, standard deep-learning pipelines exhibit major methodological deficiencies when applied to pediatric cohorts. First, numerous published studies partition auscultation audio randomly at the recording or slice level rather than grouping by patient identifier. Because multiple auscultation recordings from the same patient share thoracic anatomical transfer functions, ambient environmental room reverberations, and identical sensor placement mechanics, random partitioning causes profound patient-level data leakage, reporting inflated accuracies (often $>90\%$) that fail when evaluated on unseen subjects.

Second, the vast majority of respiratory sound benchmarks are derived from adult cohorts (such as the ICBHI 2017 scientific challenge dataset). Pediatric patients exhibit markedly different acoustic characteristics: higher baseline respiratory rates (30–60 breaths per minute in infants compared to 12–20 bpm in adults), significantly narrower airway calibers resulting in higher-frequency wheeze resonances, and thinner chest walls that attenuate high frequencies less than adult tissue.

Third, deep learning models frequently conflate systemic clinical diagnoses with local acoustic sound generation. A child clinically diagnosed with focal lobar pneumonia may present with dense crackles at the affected lung base, but produce clear, normal vesicular breath sounds at the contralateral apex. Forcing an acoustic network to predict systemic hospital discharge diagnoses from localized single-channel recordings introduces severe contradictory label noise.

This paper establishes an open, patient-independent acoustic classification baseline on the SPRSound database, evaluating time-frequency CNN feature extraction, Dual Pooling, and multi-scale sliding-window inference.

---

## II. Related Work

### A. Respiratory Sound Databases
The benchmark landscape in pulmonary acoustics has historically been dominated by adult databases. Rocha et al. [1] introduced the ICBHI 2017 database, consisting of 6,898 breathing cycles from 126 patients (mean age ~66 years). While ICBHI established standardized evaluation splits, its annotations are cycle-level segments, and its adult COPD predominance makes it poorly suited for pediatric modeling. To address this deficiency, Zhang et al. [2] released the SJTU Paediatric Respiratory Sound Database (SPRSound), comprising 2,683 recordings collected from 356 pediatric patients at Shanghai Children's Medical Center. SPRSound provides both recording-level adventitious classifications and multi-site thoracic auscultation metadata.

### B. Deep Learning on Time-Frequency Representations
Convolutional Neural Networks applied to Log Mel-Spectrograms have superseded traditional feature engineering (e.g., autoregressive coefficients, wavelet transforms, and mel-frequency cepstral coefficients) [3], [4]. Perna and Tagarelli [5] demonstrated that 2D-CNNs capture complex spectral textures associated with adventitious phenomena. However, standard architectures uniformly employ Global Average Pooling (GAP) prior to the classification head. While GAP enforces spatial invariance, it smooths out temporal impulse bursts, creating a theoretical blind spot for explosive, millisecond-duration crackles.

### C. Data Leakage in Healthcare Machine Learning
The critical review by Silva et al. [6] documented widespread data leakage in biomedical machine learning. Intra-subject correlation creates shortcut learning where networks identify patient-specific background noise rather than clinical pathology. Enforcing patient-grouped stratified cross-validation is a non-negotiable prerequisite for authentic generalizability.

---

## III. Research Gap and Motivation

While recent studies benchmark deep learning models on SPRSound, three fundamental scientific gaps remain unaddressed:
1. **The Diagnostic vs. Acoustic Granularity Gap:** Previous challenge entries frequently attempted to predict patient-level diseases directly from audio, without accounting for the fact that a large majority of recordings from sick children contain acoustically normal breath sounds. The exact mathematical and empirical failure mode of this mismatch has not been rigorously documented.
2. **The Temporal Truncation Dilemma:** Due to batching constraints in 2D-CNNs, existing pipelines truncate variable-length stethoscopic audio (8 to 15 seconds) to a fixed center-crop window (e.g., 8.192 seconds / 128 frames). In standard 15.36-second recordings, fixed center-cropping physically discards 46.9% of the signal. Whether adventitious events occurring in outer breath cycles can be recovered via post-hoc test-time sliding-window inference remains an open research question.
3. **The Crackle vs. Wheeze Detection Disparity:** Transient crackles (DAS) and continuous wheezes (CAS) exhibit fundamentally distinct time-frequency signatures. Standard pooling operations and uniform probability aggregation may interact differently with these two acoustic phenomena.

This study directly investigates these three gaps through leak-free empirical evaluation.

---

## IV. Dataset Description

We evaluate the official open-access SPRSound database [2], acquired with Yunting Model II electronic stethoscopes at four standardized chest positions ($p_1$: left posterior, $p_2$: left lateral, $p_3$: right posterior, $p_4$: right lateral) at 8,000 Hz (16-bit mono PCM).

### A. Four-Class Acoustic Target (`SPRSOUND_RECORD_4CLASS`)
- **Normal ($c=0$):** Clear vesicular breath sounds without adventitious acoustic phenomena.
- **Continuous Adventitious Sounds / CAS ($c=1$):** Musical, sustained tones ($\tau > 100\text{ ms}$, fundamental frequency typically 100–1000 Hz) corresponding to wheezes and rhonchi.
- **Discontinuous Adventitious Sounds / DAS ($c=2$):** Explosive, non-musical transients ($\tau < 20\text{ ms}$) corresponding to fine and coarse crackles.
- **CAS & DAS ($c=3$):** Simultaneous co-occurrence of both continuous and discontinuous adventitious sounds.
- **Poor Quality (Systematic Exclusion):** 70 recordings exhibiting severe motion artifacts, crying, or uninterpretable SNR were excluded, leaving 2,613 valid auscultations across 356 patients.

### B. Class Imbalance Profile
In the unaugmented training partition (Folds 1–4, $N=1,418$), class counts exhibit extreme skew:
- Normal: 1,042 recordings (73.48%)
- DAS: 199 recordings (14.03%) — **Ratio: 5.24 : 1**
- CAS: 101 recordings (7.12%) — **Ratio: 10.32 : 1**
- CAS & DAS: 76 recordings (5.36%) — **Ratio: 13.71 : 1**

---

## V. Proposed Methodology

```
+-------------------------------------------------------------------------------------------------------+
|                                    ACOUSTIC EPIDEMIOLOGY PIPELINE                                     |
+-------------------------------------------------------------------------------------------------------+
|  Raw Audio (8 kHz)  -->  Bandpass Filter (50-2000 Hz)  -->  Log Mel-Spectrogram (128 Mel Bins)        |
|                                                                        |                              |
|  +---------------------------------------------------------------------+                              |
|  |                                                                                                    |
|  v [Fixed Center-Crop Baseline]                         v [Sliding-Window Ablation Engine]             |
|  Slice (128x128x1, 8.192s)                              Multi-Window Slices (K x 128x128x1, Stride=64) |
|  4-Block 2D-CNN with Dual Pooling                       4-Block 2D-CNN with Dual Pooling              |
|  Softmax Probability Vector (1x4)                       Softmax Probability Matrix (Kx4)              |
|                                                         Abnormality-Weighted Max Aggregation          |
|                                                         Final Probability Vector (1x4)                |
+-------------------------------------------------------------------------------------------------------+
```

### A. Time-Frequency Feature Extraction Pipeline
1. **Signal Conditioning:** Raw audio is resampled to $f_s = 8,000\text{ Hz}$ mono PCM. A 4th-order zero-phase Butterworth bandpass filter ($[50\text{ Hz}, 2,000\text{ Hz}]$) attenuates low-frequency stethoscope friction ($<50\text{ Hz}$) and sensor noise above pediatric diagnostic ranges ($>2,000\text{ Hz}$).
2. **Short-Time Fourier Transform (STFT):** Computed with a periodic Hann window, FFT size $N_{\text{fft}} = 1,024$ ($128.0\text{ ms}$), and hop length $R = 512$ samples ($64.0\text{ ms}$, 50% temporal overlap).
3. **Log Mel-Filterbank:** 128 triangular Mel filters spanning $[50\text{ Hz}, 2,000\text{ Hz}]$ project linear frequencies onto the Mel scale. Logarithmic power compression is applied:
   $$S_{\text{dB}} = 10 \cdot \log_{10}(\max(S, 10^{-10}))$$
4. **Normalization:** Each spectrogram is min-max normalized to $[0.0, 1.0]$.
   *(See `docs/research_figures/fig2_preprocessing_pipeline.png` for visual waveform-to-spectrogram progression).*

### B. 2D-CNN with Dual Pooling Architecture
The network, `PediatricAcousticBaselineCNN` (1,389,092 parameters), comprises four convolutional blocks:
- **Conv Blocks 1–3:** Conv2D with 32, 64, and 128 filters ($3\times3$ kernels, $L_2 = 10^{-4}$), Batch Normalization, ReLU, MaxPool($2\times2$), and Dropout (0.20 to 0.25).
- **Conv Block 4:** Conv2D(256, $3\times3$, named `last_conv_layer`), Batch Normalization, ReLU, MaxPool($2\times2$). Explicitly named for Grad-CAM explainability.
- **Dual Pooling Layer:** A parallel concatenation of Global Average Pooling (GAP) and Global Max Pooling (GMP):
  $$\mathbf{h}_{\text{pooled}} = \left[ \frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W \mathbf{Z}_{i,j,:}, \;\; \max_{i,j} \mathbf{Z}_{i,j,:} \right] \in \mathbb{R}^{512}$$
  GAP integrates sustained harmonic energy across breathing cycles (essential for continuous wheezes), while GMP preserves localized impulse activation peaks (essential for transient crackles).
- **Dense Head:** Dense(128, ReLU, $L_2 = 10^{-4}$), Dropout(0.30), Dense(4, Softmax).

---

## VI. Experimental Setup

### A. Patient-Disjoint Partitioning Protocol
Patients are partitioned using `StratifiedGroupKFold` on `patient_id` *(see `docs/research_figures/fig4_patient_partitioning.png`)*:
- **Training Set (Folds 1–4):** 1,418 recordings across 202 patients. Expanded via minority SpecAugment to 2,745 training tensors.
- **Validation Set (Fold 0):** 354 recordings across 49 patients. Strictly 0 patient overlap with training folds.
- **Independent Held-Out Test Set (`held_out_inter_test`):** 347 recordings across 41 unseen patients. Strictly 0 patient overlap with training or validation cohorts.

### B. Training Hyperparameters & Collapse Rejection
- Optimizer: Adam ($\alpha = 3 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, batch size 32, 25 epochs).
- Loss Function: Sparse Categorical Cross-Entropy.
- Checkpoint Selection: Standard validation loss monitoring causes premature checkpointing at Epoch 0 where the model predicts 100% Normal. We tracked validation **Balanced Accuracy** ($\frac{1}{4}\sum_{c=0}^3 \text{Recall}_c$) with collapse rejection. The optimal checkpoint was restored from **Epoch 23**.
- Hardware Environment: Apple M3 Max, macOS 15, Python 3.13, TensorFlow/Keras headless execution.

---

## VII. Results

### A. Failure of the Disease Classification Target (Negative Finding)
In preliminary experiments, we trained a 3-class disease CNN (*Healthy Control*, *Pneumonia*, *Wheezing Disorders*):
- **Experiment 1 (Standard CE):** Collapsed to predicting 100% Pneumonia (Validation Accuracy 69.47%, Balanced Accuracy 33.77%, Wheezing Recall 1.30%).
- **Experiment 4 (Targeted SpecAugment + Focal Loss):** Healthy recall reached 72.73% and Pneumonia recall 53.44%, but **Wheezing Disorders recall remained 0.00%** (0 out of 77 validation samples).
- *Root Cause Analysis:* Of 77 validation recordings from wheezing patients, 61 (79.2%) contained clear vesicular breath sounds without wheezing or rhonchi. A recording-level model cannot learn systemic disease when the physical audio signal is acoustically normal.

### B. Acoustic Sound Baseline Performance
Evaluating `PediatricAcousticBaselineCNN` on the 4-class acoustic target broke through the majority-class collapse:

| Partition Group | Overall Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Normal Recall | CAS (Wheeze) Recall | DAS (Crackle) Recall | CAS & DAS Recall |
| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| **Validation Fold 0 ($N=354$)** | 74.29% | 33.46% | 0.3284 | 0.6948 | 93.87% (245/261) | 0.00% (0/25) | 34.69% (17/49) | 5.26% (1/19) |
| **Held-Out Test ($N=347$)** | 58.50% | **37.12%** | **0.3018** | 0.5808 | 77.59% (187/241) | **1.54%** (1/65) | **45.83%** (11/24) | **23.53%** (4/17) |

On the independent held-out test set, the model actively predicted all four classes across 41 completely unseen patients *(see `docs/research_figures/fig5_confusion_matrices.png`, Panel a)*.

---

## VIII. Sliding-Window Multi-Scale Ablation

To evaluate whether fixed center-cropping (8.192s) misses peripheral adventitious sounds, we evaluated sliding windows across the full uncropped Log Mel-Spectrograms ($\mathbb{R}^{128 \times T}$).

### A. Algorithm & Boundary Handling
Windows of $W = 128$ frames were extracted with stride $S = 64$ (50% overlap) and $S = 32$ (75% overlap). If $T - \text{starts}[-1] < 128$, the trailing slice $[T-128 : T]$ was appended, guaranteeing **100% temporal coverage without zero-padding distortion**.

### B. Aggregation Comparison on Validation Fold 0 ($N=354$)

| Configuration | Stride ($S$) | Aggregation Strategy | Accuracy | Balanced Accuracy | Macro F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall |
| :--- | :---:| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| **Center-Crop Baseline** | N/A | Center Crop | 74.29% | 33.46% | 0.3284 | 93.87% | 0.00% | 34.69% | 5.26% |
| `stride_64_mean` | 64 | Uniform Mean | 75.42% | 34.25% | 0.3393 | 95.02% | 0.00% | 36.73% | 5.26% |
| `stride_64_max` | 64 | Max Normalized | 75.14% | 34.16% | 0.3378 | 94.64% | 0.00% | 36.73% | 5.26% |
| `stride_64_top_k` | 64 | Top-2 Mean | 75.42% | 34.25% | 0.3393 | 95.02% | 0.00% | 36.73% | 5.26% |
| **`stride_64_abnormality_max`** | **64** | **Abnormality Max** | **74.29%** | **34.29%** | **0.3342** | **93.10%** | **0.00%** | **38.78%** | **5.26%** |

`stride_64_abnormality_max` was selected strictly on validation data due to achieving peak Balanced Accuracy (34.29%) and peak crackle sensitivity (38.78%).

### C. Independent Test Evaluation (`held_out_inter_test`, $N=347$)

| Configuration | Stride ($S$) | Aggregation Strategy | Accuracy | Balanced Accuracy | Macro F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Batch Latency |
| :--- | :---:| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| **Center-Crop Baseline** | N/A | Center Crop | 58.50% | **37.12%** | **0.3018** | 77.59% (187) | **1.54%** (1) | 45.83% (11) | **23.53%** (4) | 1.8 ms/sample |
| `stride_64_mean` | 64 | Uniform Mean | **60.23%** | 32.43% | 0.2643 | **82.16%** (198) | 0.00% (0) | 41.67% (10) | 5.88% (1) | 3.6 ms/sample |
| **`stride_64_abnormality_max`** *(Locked)* | **64** | **Abnormality Max** | **55.62%** | **34.01%** | **0.2666** | **74.27%** (179) | **0.00%** (0) | **50.00%** (12) | **11.76%** (2) | **3.6 ms/sample** |
| `stride_32_abnormality_max` *(Exploratory)* | 32 | Abnormality Max | 55.04% | 36.11% | 0.2777 | 72.61% (175) | 0.00% (0) | **54.17%** (13) | 17.65% (3) | 4.3 ms/sample |

*(See `docs/research_figures/fig5_confusion_matrices.png`, Panel b, and `docs/research_figures/fig7_performance_comparison.png`).*

---

## IX. Discussion

### A. Divergent Dynamics of Crackles (DAS) vs. Wheezes (CAS)
Under sliding-window inference, crackle (*DAS*) recall increased from 45.83% to **50.00%** ($S=64$) and **54.17%** ($S=32$). Crackles are explosive, discontinuous transients ($\tau < 20\text{ ms}$) spanning broad frequency bands. In 15-second recordings containing multiple breath cycles, crackles occurring near the margins are clipped out by center-cropping; multi-window scanning successfully retrieves these peripheral events.

Conversely, wheeze (*CAS*) recall remained at **0.00%** across all sliding-window configurations. Wheezes are continuous musical tones with distinct harmonic structures. Because the CNN was trained exclusively on center-cropped windows with a 10.3:1 class imbalance, its convolutional filters never learned robust, phase-invariant harmonic representations. Evaluating sliding windows at inference time cannot recover acoustic features that the underlying model never learned to extract.

### B. The Mean Probability Dilution Effect
Uniform mean pooling increased overall test accuracy from 58.50% to **60.23%** and Normal recall to **82.16%**, but decreased Balanced Accuracy to **32.43%** and suppressed co-occurring adventitious recall (`CAS & DAS`) from 23.53% to 5.88%. In a 15-second recording comprising three breath cycles, an adventitious sound occurring in only one cycle produces a high abnormal probability in window 3 ($P \approx 0.65$), but windows 1 and 2 reflect normal vesicular breathing ($P \approx 0.85$). Averaging across windows allows normal breath cycles to dilute and drown out the pathology. Peak-sensitive aggregation (`abnormality_max`) is essential to preserve sensitivity to rare acoustic events.

---

## X. Limitations

1. **Weak Supervision:** SPRSound annotates full recordings without breath-cycle bounding boxes. The network is penalized for classifying normal breath cycles within abnormal recordings as Normal.
2. **Extreme Class Imbalance:** Normal sounds constitute 73.5% of training samples, while wheezes constitute only 7.1%.
3. **Absence of Pre-trained Weights:** Due to offline execution constraints and domain mismatch with adult datasets (ICBHI), models were trained from scratch without large-scale audio foundation models (e.g., AudioSet or PANNs).
4. **Single-Center Cohort:** All data was collected at Shanghai Children's Medical Center with Yunting Model II electronic stethoscopes; multi-device generalization remains unverified.

---

## XI. Ethical and Clinical Considerations

The deployment of automated diagnostic systems in pediatric healthcare introduces profound safety and ethical implications:
1. **Risk of Misdiagnosis & False Reassurance:** A model with 0.00% wheeze recall would fail to detect acute pediatric asthma exacerbations or bronchiolitis, potentially delaying emergency bronchodilator therapy.
2. **Research Prototype Boundary:** This model is strictly an **experimental baseline** and is **NOT approved or safe for clinical triage, automated screening, or medical decision support**.
3. **Ethical Reporting:** We intentionally report negative findings (wheeze detection failure and disease-label mismatch) rather than selectively presenting favorable metrics.

---

## XII. Conclusion

This study establishes a scientifically rigorous, patient-independent acoustic classification baseline for pediatric respiratory sounds. We demonstrated that:
1. Predicting systemic pediatric diseases from single-channel stethoscopy induces severe label mismatch and model collapse.
2. A 4-block 2D-CNN with Dual Pooling achieves active 4-class prediction and 45.83% crackle recall on an independent held-out test set.
3. Multi-window sliding inference successfully recovers transient boundary crackles (improving DAS recall to 50.00%–54.17%), but post-hoc windowing cannot resolve wheezing representation deficits without end-to-end retraining.

---

## References

[1] B. M. Rocha et al., "An open access database for the evaluation of respiratory sound classification algorithms," *Physiological Measurement*, vol. 40, no. 3, p. 035001, Mar. 2019.  
[2] Q. Zhang et al., "SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database," *IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)*, vol. 16, no. 5, pp. 867–881, Oct. 2022.  
[3] L. Fraiwan, K. Khassawneh, and J. Alkhodari, "Automatic identification of respiratory diseases from stethoscopic lung sound signals using ensemble estimators and deep learning," *Biocybernetics and Biomedical Engineering*, vol. 41, no. 1, pp. 1–14, Jan. 2021.  
[4] S. Demir, K. Sengur, and V. Bajaj, "Convolutional neural networks for respiratory sound classification," *Artificial Intelligence in Medicine*, vol. 109, p. 101961, Sep. 2020.  
[5] D. Perna and A. Tagarelli, "Deep auscultation: Predicting respiratory anomalies from lung sounds using convolutional neural networks," in *Proc. IEEE 32nd International Symposium on Computer-Based Medical Systems (CBMS)*, 2019, pp. 20–25.  
[6] C. B. S. P. Silva et al., "Data leakage in machine learning for healthcare: A systematic review," *Journal of Biomedical Informatics*, vol. 142, p. 104388, Jun. 2023.
