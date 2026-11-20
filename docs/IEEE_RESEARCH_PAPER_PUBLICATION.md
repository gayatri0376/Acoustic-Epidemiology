# Acoustic Epidemiology: Patient-Independent Classification of Pediatric Respiratory Sounds Using Dual-Pooling CNNs and Multi-Scale Temporal Scanning

**Authors:**  
Gayatri$^1$, Mohini$^2$, Samruddhi$^3$, Shreya$^4$, Vishal$^5$  
$^1$Department of Software Systems & Backend Architecture  
$^2$Department of Machine Learning Modeling & Neural Computing  
$^3$Department of Biomedical Signal Processing & Data Engineering  
$^4$Department of Human-Computer Interaction & Clinical Visualization  
$^5$Department of Quality Assurance, Verification & Research Protocols  
*Acoustic Epidemiology Research Group*  
Email: {gayatri, mohini, samruddhi, shreya, vishal}@research.acousticepidemiology.org  

---

### Abstract
Pediatric respiratory infections are among the leading causes of acute morbidity and hospitalization in young children worldwide. Although digital stethoscopy paired with machine learning offers an objective pathway for automated pulmonary triage, existing literature frequently suffers from three systemic weaknesses: patient-level data leakage, reliance on adult datasets that fail to reflect pediatric thoracic acoustic properties, and the conflation of localized chest sounds with systemic hospital discharge diagnoses. 

In this work, we present an open, reproducible, patient-disjoint deep-learning framework for four-class pediatric respiratory sound classification evaluated on the SJTU Paediatric Respiratory Sound Database (SPRSound). We first investigate the diagnostic granularity gap, demonstrating that attempting to classify three clinical disease states (*Healthy Control*, *Pneumonia*, and *Wheezing Disorders*) directly from localized stethoscopic audio leads to severe model collapse (0.00% recall on wheezing disorders) because over 70% of chest recordings from diseased pediatric patients contain acoustically normal vesicular breath sounds. 

Reframing the objective around acoustic sound categories (*Normal*, *Continuous Adventitious Sounds [CAS / Wheezes]*, *Discontinuous Adventitious Sounds [DAS / Crackles]*, and *CAS & DAS*), we design a 4-block 2D Convolutional Neural Network (CNN) featuring a concatenated **Dual-Pooling** layer (Global Average Pooling and Global Max Pooling) and targeted time-frequency augmentation. On a strictly held-out, patient-disjoint test benchmark of 347 recordings from 41 unseen patients (zero patient overlap), our baseline model achieves **58.50% Accuracy**, **37.12% Balanced Accuracy**, and a **0.3018 Macro F1-Score**, actively identifying all four classes with per-class recalls of 77.59% (*Normal*), 45.83% (*DAS*), 23.53% (*CAS & DAS*), and 1.54% (*CAS*). 

To address temporal truncation caused by standard 8.192-second center-cropping, we develop a multi-scale sliding-window scanning protocol evaluated across complete, uncropped recordings. Using an abnormality-weighted maximum aggregation strategy locked strictly on validation data, sliding-window scanning increases crackle (*DAS*) recall from 45.83% to **50.00%** ($S=64$) and **54.17%** ($S=32$). However, wheeze (*CAS*) recall remains at 0.00%, illustrating that test-time windowing can recover localized transient anomalies but cannot compensate for representation deficits caused by severe training class imbalance (10.3:1) and weak recording-level labels. We provide our models, preprocessing pipeline, and clinical web workstation as an open baseline, explicitly concluding that current acoustic models serve as decision-support aids rather than stand-alone diagnostic tools.

**Keywords:** Pediatric auscultation, respiratory sound classification, SPRSound, convolutional neural networks, dual pooling, sliding-window inference, data leakage, Grad-CAM interpretability.

---

## I. Introduction

Acute respiratory tract infections—including pneumonia, bronchiolitis, and reactive airway diseases—account for over 700,000 deaths annually among children under five years of age, predominantly in low- and middle-income regions [1]. In pediatric practice, physical examination of the chest via auscultation remains the initial frontline screening tool. Clinicians listen for adventitious acoustic phenomena superimposed on baseline vesicular breathing: continuous, musical wheezes (indicative of small-airway constriction) and discontinuous, explosive crackles (indicative of alveolar fluid accumulation or reopening of atelectatic lung units) [2].

Despite its ubiquity, conventional acoustic stethoscopy is hindered by inherent subjectivity, low inter-examiner agreement (often with Cohen's kappa $\kappa < 0.40$ even among trained pediatricians), and ambient acoustic contamination in busy emergency wards [3]. These shortcomings have motivated the development of computerized respiratory sound analysis (CRSA) algorithms that ingest digitized stethoscopic audio and generate automated classifications [4].

However, translating machine learning algorithms from proof-of-concept demonstrations to credible clinical tools has encountered substantial methodological roadblocks. A critical review of published biomedical sound literature reveals widespread vulnerabilities:

1. **Patient-Level Data Leakage:** Many published studies divide audio into training, validation, and test sets through random sample-level or segment-level shuffling [5]. When multiple auscultation recordings from the same child appear in both training and test partitions, networks learn the patient's unique anatomical acoustic transfer function, sensor placement resonance, and background hospital room noise rather than generalized pathology. Such models frequently report inflated classification accuracies exceeding 90%, only to collapse when tested on genuinely unseen patients.
2. **The Adult-Pediatric Disparity:** Most available open benchmarks, such as the ICBHI 2017 challenge dataset [6], are predominantly composed of elderly adult patients diagnosed with chronic obstructive pulmonary disease (COPD). Pediatric chests possess fundamentally distinct acoustic properties: higher baseline respiratory rates (often 30–60 breaths per minute in infants compared to 12–20 bpm in adults), significantly narrower airway calibers that generate higher-frequency wheeze harmonics, and thinner chest walls that exhibit lower high-frequency sound attenuation.
3. **The Clinical Diagnosis vs. Acoustic Granularity Gap:** Several recent competitions have encouraged models to predict hospital discharge diagnoses directly from localized audio clips [7]. This ignores the clinical reality of auscultation: a child with right lower-lobe pneumonia will produce coarse crackles when auscultated over the right posterior base, but completely clear, normal breath sounds over the contralateral left apex. Assigning a global label of "Pneumonia" to every acoustic recording from that patient forces models to treat acoustically normal sounds as pathological, corrupting feature representations.

To resolve these challenges, this paper presents **Acoustic Epidemiology**, an open research framework that implements patient-independent four-class acoustic sound classification on the SJTU Paediatric Respiratory Sound Database (SPRSound) [7]. 

### Key Contributions
- **Empirical Demonstration of Diagnostic Failure:** We mathematically and empirically demonstrate why direct disease classification from single-site pediatric audio fails, proving that over 70% of thoracic recordings from diseased children contain normal vesicular sounds, inducing catastrophic model collapse.
- **Leak-Free Benchmark Architecture:** We introduce a 4-block 2D-CNN incorporating **Dual Pooling** (concatenated Global Average Pooling and Global Max Pooling) trained under strict patient-disjoint cross-validation with collapse-rejection checkpointing.
- **Multi-Scale Sliding-Window Ablation:** We evaluate multi-window scanning across complete uncropped audio files, demonstrating that abnormality-weighted max aggregation increases crackle sensitivity from 45.83% to 54.17%, while systematically analyzing why wheeze sensitivity cannot be recovered without end-to-end retraining.
- **Model Explainability & Clinical Safety:** We integrate Grad-CAM saliency heatmaps over Log-Mel Spectrograms to expose network decision-making regions, coupling the architecture with an auditable clinical web workstation designed around strict research disclaimers.

---

## II. Related Work

### A. Respiratory Sound Benchmarks
Public respiratory sound research gained momentum following the release of the ICBHI 2017 Respiratory Sound Database by Rocha et al. [6]. Consisting of 6,898 annotated respiratory cycles from 126 subjects, ICBHI catalyzed extensive neural network exploration. However, the median patient age in ICBHI is approximately 66 years, and recordings are dominated by chronic adult conditions. 

Recognizing the lack of pediatric data, Zhang et al. [7] released the SPRSound database in 2022. Collected at Shanghai Children's Medical Center using digital stethoscopes, SPRSound contains 2,683 recordings across 356 pediatric patients, providing both recording-level adventitious classifications and multi-site anatomical metadata.

### B. Time-Frequency Neural Architectures
Direct raw-waveform 1D-CNNs and recurrent networks have largely been superseded by 2D-CNNs operating on time-frequency representations [8], [9]. Log Mel-Spectrograms, which project linear frequencies onto the non-linear Mel scale modeling human auditory perception, have emerged as the standard representation. 

Standard 2D-CNN architectures (e.g., ResNet, VGG-variants) uniformly apply Global Average Pooling (GAP) following the final convolutional layer. While GAP provides spatial translation invariance and mitigates overfitting, it averages spatial activations across the entire time-frequency plane. In audio processing, this temporal smoothing acts as a low-pass filter on activation peaks, heavily penalizing brief, explosive acoustic events such as crackles [10].

### C. Evaluation Protocols and Integrity
Silva et al. [5] documented that data leakage in clinical machine learning occurs predominantly through intra-subject record splitting. When evaluating stethoscopic recordings, patient-grouped cross-validation is not an optional refinement—it is a baseline prerequisite. Models evaluated without patient grouping cannot substantiate claims of real-world clinical utility.

---

## III. Problem Formulation & The Granularity Dilemma

### A. The Diagnostic Label Mismatch
Let a patient $P_i$ have an established systemic clinical diagnosis $D_i \in \{\text{Healthy Control}, \text{Pneumonia}, \text{Wheezing Disorders}\}$. A clinical examination involves auscultating $M$ anatomical chest locations $L_{i,j}$ ($j=1, \dots, M$). Each recorded audio signal $x_{i,j}(t)$ generates an acoustic sound category $y_{i,j} \in \{\text{Normal}, \text{CAS}, \text{DAS}, \text{CAS \& DAS}\}$.

In standard practice, a patient-level disease diagnosis is assigned based on systemic findings: laboratory blood tests, chest radiography, and multi-site physical examination. Crucially:
$$\mathbb{P}(y_{i,j} = \text{Normal} \mid D_i \neq \text{Healthy}) \gg 0$$

In our audit of the SPRSound cohort, of the 77 validation recordings collected from patients diagnosed with Wheezing Disorders, **61 recordings (79.22%)** contained completely clear vesicular sounds without detectable wheezing. When a model is trained with cross-entropy loss to minimize error against $D_i$ using individual audio clips $x_{i,j}$, the optimization gradient forces identical acoustic features (vesicular airflow) to map simultaneously to *Healthy Control*, *Pneumonia*, and *Wheezing Disorders*. 

As documented in Section VII, this contradictory supervision leads directly to representation collapse, where the network minimizes risk by predicting the majority disease class across all inputs. Consequently, automated analysis of pediatric stethoscopy must be formulated strictly at the **acoustic sound level** ($y_{i,j}$).

---

## IV. Dataset & Partitioning Protocol

### A. SPRSound Cohort Specifications
We utilize the complete SPRSound database [7], recorded using Yunting Model II electronic stethoscopes at four anatomical chest locations:
- $p_1$: Left posterior lung field
- $p_2$: Left lateral chest wall
- $p_3$: Right posterior lung field
- $p_4$: Right lateral chest wall

Recordings are sampled at $f_s = 8,000\text{ Hz}$ with 16-bit PCM quantization. Durations range from 8.0 to 15.36 seconds, capturing multiple consecutive respiratory cycles.

### B. Exclusion Criteria
To maintain clinical data integrity, 70 recordings annotated as *Poor Quality* (severe microphone friction, child crying, or signal-to-noise ratio $< 0\text{ dB}$) were systematically excluded prior to partitioning. This left **2,613 valid auscultation recordings** across 356 unique patients.

### C. Target Sound Categories
Each valid recording is mapped to one of four mutually exclusive acoustic classes:
1. **Normal ($c=0$):** Regular vesicular breath sounds with no adventitious components.
2. **Continuous Adventitious Sounds / CAS ($c=1$):** Sustained musical tones ($\tau > 100\text{ ms}$, fundamental frequency typically 100–1000 Hz) including wheezes and rhonchi.
3. **Discontinuous Adventitious Sounds / DAS ($c=2$):** Explosive, non-musical transients ($\tau < 20\text{ ms}$) corresponding to fine or coarse crackles.
4. **CAS & DAS ($c=3$):** Simultaneous co-occurrence of both wheezes and crackles within the recording segment.

### D. Patient-Disjoint Partitioning
To guarantee zero patient leakage, the 2,613 recordings were partitioned using `StratifiedGroupKFold` grouped strictly by `patient_id`. Table I details the resulting dataset partitioning.

```
+-------------------------------------------------------------------------------------------------+
|                               TABLE I: DATASET PARTITIONING & PATIENT ISOLATION                 |
+-------------------------------------------------------------------------------------------------+
| Partition Name          | Split Identifier     | Total Audio | Excluded | Valid Audio | Patients|
+-------------------------+----------------------+-------------+----------+-------------+---------+
| Training (Folds 1–4)    | train_cv (Folds 1-4) | 1,457       | 39       | 1,418       | 202     |
| Validation (Fold 0)     | train_cv (Fold 0)    | 369         | 15       | 354         | 49      |
| Held-Out Test           | held_out_inter_test  | 355         | 8        | 347         | 41      |
| Intra-Subject Test      | intra_subject_test   | 141         | 5        | 136         | 13      |
| Secondary Benchmark     | test_2023            | 361         | 3        | 358         | 51      |
| Full Database Total     | All Partitions       | 2,683       | 70       | 2,613       | 356     |
+-------------------------------------------------------------------------------------------------+
```

Patients in the *Held-Out Test Set* ($N=347$, 41 patients) share zero overlap with either training or validation cohorts. Furthermore, within the cross-validation pool, Fold 0 ($N=354$, 49 patients) is strictly patient-disjoint from Folds 1–4.

### E. Class Imbalance Profile
The distribution across evaluated partitions exhibits acute clinical skew, summarized in Table II. In the training set, *Normal* breath sounds constitute 73.48% of samples, creating an imbalance ratio of **10.32 : 1** against wheezes (*CAS*) and **13.71 : 1** against co-occurring sounds (*CAS & DAS*).

```
+-------------------------------------------------------------------------------------------------+
|                        TABLE II: CLASS DISTRIBUTION ACROSS EVALUATED PARTITIONS                 |
+---------------------+-----------------------+-----------------------+---------------------------+
| Acoustic Sound Class| Training (Folds 1–4)  | Validation (Fold 0)   | Held-Out Test Set         |
+---------------------+-----------------------+-----------------------+---------------------------+
| Normal              | 1,042 (73.48%)        | 261 (73.73%)          | 241 (69.45%)              |
| CAS (Wheezes)       | 101 (7.12%)           | 25 (7.06%)            | 65 (18.73%)               |
| DAS (Crackles)      | 199 (14.03%)          | 49 (13.84%)           | 24 (6.92%)                |
| CAS & DAS (Mixed)   | 76 (5.36%)            | 19 (5.37%)            | 17 (4.90%)                |
| Total Audio Files   | 1,418 (100.0%)        | 354 (100.0%)          | 347 (100.0%)              |
+---------------------+-----------------------+-----------------------+---------------------------+
```

---

## V. Methodology

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

### A. Signal Conditioning & Preprocessing
Raw digital recordings $s(t)$ undergo a standardized three-stage conditioning pipeline:

1. **Bandpass Filtering:** To suppress stethoscope chest-piece friction and body movement artifacts ($<50\text{ Hz}$) as well as high-frequency ambient room acoustics above pediatric respiratory ranges ($>2,000\text{ Hz}$), we apply a 4th-order zero-phase Butterworth bandpass filter:
   $$H(f) = \frac{1}{\sqrt{1 + \left(\frac{f^2 - f_L f_H}{f(f_H - f_L)}\right)^{2n}}}$$
   where $f_L = 50\text{ Hz}$, $f_H = 2,000\text{ Hz}$, and $n=4$.
2. **Short-Time Fourier Transform (STFT):** Windowed complex spectra are extracted using a periodic Hann window:
   $$X(m, k) = \sum_{n=0}^{N_{\text{fft}}-1} s[n + mR] w[n] e^{-j 2\pi k n / N_{\text{fft}}}$$
   with frame length $N_{\text{fft}} = 1,024$ samples ($128.0\text{ ms}$) and hop size $R = 512$ samples ($64.0\text{ ms}$, 50% temporal overlap).
3. **Log Mel-Filterbank Projection:** The power spectrum $|X(m, k)|^2$ is filtered through $M = 128$ triangular filterbanks spaced evenly along the Mel frequency scale between $50\text{ Hz}$ and $2,000\text{ Hz}$:
   $$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
   Logarithmic decibel scaling is applied:
   $$S_{\text{dB}}(m, b) = 10 \log_{10}\left(\max\left(S(m, b), 10^{-10}\right)\right)$$
4. **Dynamic Range Normalization:** Each spectrogram is scaled to $[0.0, 1.0]$:
   $$\tilde{S}(m, b) = \frac{S_{\text{dB}}(m, b) - \min(S_{\text{dB}})}{\max(S_{\text{dB}}) - \min(S_{\text{dB}}) + \epsilon}$$

### B. CNN Architecture with Dual Pooling
The neural network, `PediatricAcousticBaselineCNN` (1,389,092 trainable parameters), is designed specifically to balance sensitivity between continuous harmonic signals and brief transient impulses.

```
Input Tensor: (128, 128, 1)  [128 Mel Bins x 128 Temporal Frames, 8.192s]
  │
  ├── Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.20)  [Output: 64x64x32]
  ├── Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.20)  [Output: 32x32x64]
  ├── Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.25) [Output: 16x16x128]
  └── Block 4: Conv2D(256, 3x3, name="last_conv_layer") -> BatchNorm -> ReLU -> MaxPool(2x2) [Output: 8x8x256]
        │
        ├── Parallel Dual Pooling:
        │     ├── Global Average Pooling (GAP) -> Vector (256)
        │     └── Global Max Pooling (GMP)     -> Vector (256)
        │
        └── Concatenation: h_pooled = [GAP, GMP] -> Vector (512)
              │
              ├── Dense(128, ReLU, L2=1e-4) -> Dropout(0.30)
              └── Dense(4, Softmax) -> Class Probabilities P(c|x)
```

#### Dual-Pooling Rationale
Standard convolutional classifiers discard spatial dimensions via Global Average Pooling:
$$\mathbf{h}_{\text{GAP}} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W \mathbf{Z}_{i,j,:}$$
While GAP effectively summarizes sustained harmonic energy across long windows (such as continuous wheezing), it severely attenuates transient crackles that may occupy only 1–2 time frames ($<20\text{ ms}$). To overcome this, we concatenate Global Max Pooling (GMP) in parallel:
$$\mathbf{h}_{\text{GMP}} = \max_{1 \le i \le H, 1 \le j \le W} \mathbf{Z}_{i,j,:}$$
$$\mathbf{h}_{\text{pooled}} = \left[ \mathbf{h}_{\text{GAP}} \,\|\, \mathbf{h}_{\text{GMP}} \right] \in \mathbb{R}^{512}$$
GMP preserves the highest feature response triggered by explosive crackle transients, while GAP maintains global harmonic context.

### C. Grad-CAM Interpretability Formulation
To verify that network decisions are driven by genuine acoustic phenomena rather than sensor artifacts, we implement Gradient-weighted Class Activation Mapping (Grad-CAM) [11] targeted at `last_conv_layer` ($\mathbf{A} \in \mathbb{R}^{8 \times 8 \times 256}$).

For a predicted class $c$, the neuron importance weights $\alpha_k^c$ are computed via global average pooling of the gradients:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^8 \sum_{j=1}^8 \frac{\partial y^c}{\partial A_{i,j}^k}$$
The class-discriminative saliency map $L_{\text{Grad-CAM}}^c$ is calculated as a rectified weighted combination of forward activation maps:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{256} \alpha_k^c \mathbf{A}^k\right)$$
Bilaterally interpolated back to the original input resolution $(128 \times 128)$, the resulting heatmap visually pinpoints the precise time-frequency coordinates that influenced classification.

---

## VI. Multi-Scale Sliding-Window Temporal Scanning

### A. The Fixed Center-Crop Limitation
In standard training protocols, audio inputs must be regularized to uniform dimensions $(128 \times 128 \times 1)$, corresponding to an 8.192-second window. In the SPRSound database, over 65% of recordings are 15.36 seconds long (240 temporal frames). Truncating recordings to an 8.192-second center-crop physically discards **46.88% of the available signal** (7.17 seconds). If an adventitious wheeze or crackle occurs exclusively during an early or late breath cycle, center-cropping forces a false negative.

### B. Sliding-Window Formulation
To analyze complete auscultations without temporal truncation, we define a sliding-window scanning operator over the full-length normalized spectrogram $\mathbf{S} \in \mathbb{R}^{128 \times T}$ ($T \ge 128$). Windows of fixed width $W=128$ are extracted with stride $S$:
$$t_k = k \cdot S, \quad k = 0, 1, \dots, K-1$$
$$\mathbf{W}_k = \mathbf{S}[:, t_k : t_k + W]$$

#### Trailing Boundary Preservation
If the remainder $T - t_{K-1} < W$, simply omitting the boundary loses terminal breaths, while zero-padding distorts the convolutional edge representations. Instead, we append a final boundary-anchored slice:
$$\mathbf{W}_{\text{final}} = \mathbf{S}[:, T - W : T]$$
This guarantees that **100% of the acoustic duration is scanned** without synthetic zero-padding.

Each slice $\mathbf{W}_k$ is passed through the network, yielding a sequence of probability vectors:
$$\mathbf{p}_k = f_{\theta}(\mathbf{W}_k) \in \Delta^4, \quad k=1, \dots, K$$

### C. Aggregation Functions
We compare four distinct temporal aggregation functions:

1. **Uniform Mean Aggregation:**
   $$\mathbf{p}_{\text{mean}} = \frac{1}{K} \sum_{k=1}^K \mathbf{p}_k$$
2. **Maximum Probability Aggregation:**
   $$\tilde{\mathbf{p}}_{\max} = \left[ \max_k p_{k,0}, \max_k p_{k,1}, \max_k p_{k,2}, \max_k p_{k,3} \right], \quad \mathbf{p}_{\max} = \frac{\tilde{\mathbf{p}}_{\max}}{\sum_{c=0}^3 \tilde{p}_{\max,c}}$$
3. **Top-$k$ Average Pooling ($k=2$):**
   $$p_{\text{top2},c} = \frac{1}{2} \sum_{j=1}^2 \text{sort\_desc}\left(\{p_{k,c}\}_{k=1}^K\right)_j$$
4. **Abnormality-Weighted Maximum Aggregation:**
   Recognizing that clinical pathology is localized while normal breathing is pervasive, we aggregate abnormal classes ($c \in \{1, 2, 3\}$) via their maximum window activation, while computing the normal class as the residual complement:
   $$\tilde{p}_c = \max_{1 \le k \le K} p_{k,c} \quad \text{for } c \in \{1, 2, 3\}$$
   $$p_{\text{abn}} = \max(\tilde{p}_1, \tilde{p}_2, \tilde{p}_3)$$
   $$\hat{p}_0 = \min_{1 \le k \le K} p_{k,0}$$
   Normalized across all four classes, this formulation ensures that a single high-confidence adventitious burst is not suppressed by multiple surrounding normal breath cycles.

---

## VII. Experimental Setup & Training

### A. Training Hyperparameters
- **Optimizer:** Adam ($\alpha = 3 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$).
- **Batch Size:** 32.
- **Epochs:** 25 with early stopping monitoring.
- **Loss Function:** Sparse Categorical Cross-Entropy:
  $$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid \mathbf{x}_i)$$
- **Regularization:** Weight decay $L_2 = 10^{-4}$ on all convolutional and dense kernels, coupled with spatial dropout ($p=0.20$ to $0.30$).
- **Data Augmentation:** Minority sound classes in the training partition were augmented using targeted SpecAugment (time masking $\le 16$ frames, frequency masking $\le 16$ Mel bins), expanding the training set from 1,418 to 2,745 tensors while strictly preserving validation isolation.

### B. Model Collapse Rejection Checkpointing
Under severe class imbalance (73.5% Normal), standard checkpoint selection based on minimum validation loss frequently selects models from Epoch 1 or 2 that have collapsed to predicting 100% Normal. We instituted strict **Collapse Rejection Checkpointing**: checkpoints are scored exclusively by validation **Balanced Accuracy**:
$$\text{BAcc} = \frac{1}{4} \sum_{c=0}^3 \text{Recall}_c = \frac{1}{4} \left( \frac{\text{TP}_0}{N_0} + \frac{\text{TP}_1}{N_1} + \frac{\text{TP}_2}{N_2} + \frac{\text{TP}_3}{N_3} \right)$$
Any epoch producing zero true positives across any class is rejected. The optimal checkpoint was restored from **Epoch 23**.

---

## VIII. Results

### A. Failure of Direct Disease Classification (Negative Result)
To document the diagnostic granularity gap, we trained a baseline 3-class disease CNN (*Healthy Control*, *Pneumonia*, *Wheezing Disorders*) on SPRSound:
- **Standard Cross-Entropy:** Collapsed into majority predicting Pneumonia (Validation Accuracy 69.47%, Balanced Accuracy 33.77%, Wheezing Recall 1.30%).
- **Class-Weighted Focal Loss + Augmentation:** Increased Healthy Control recall to 72.73% and Pneumonia recall to 53.44%, but **Wheezing Disorders recall remained exactly 0.00%** (0 out of 77 validation recordings detected).

Because 79.2% of auscultations from wheezing patients contain normal vesicular airflow, gradient updates for the wheezing class conflict directly with normal sounds, preventing the network from learning discriminative filters.

### B. Acoustic Baseline Classification Performance
Switching to four-class acoustic sound targets successfully broke through majority-class collapse. Table III summarizes baseline performance using 8.192-second center-cropping.

```
+-------------------------------------------------------------------------------------------------------+
|                               TABLE III: ACOUSTIC BASELINE CLASSIFICATION PERFORMANCE                 |
+----------------------+--------------------+--------------------+---------------+----------------------+
| Metric               | Validation Fold 0  | 95% Confidence Int.| Held-Out Test | 95% Confidence Int.  |
|                      | (N = 354)          |                    | (N = 347)     |                      |
+----------------------+--------------------+--------------------+---------------+----------------------+
| Overall Accuracy     | 74.29%             | [69.7%, 78.9%]     | 58.50%        | [53.3%, 63.7%]       |
| Balanced Accuracy    | 33.46%             | [29.2%, 37.7%]     | 37.12%        | [32.6%, 41.6%]       |
| Macro F1-Score       | 0.3284             | [0.281, 0.376]     | 0.3018        | [0.254, 0.350]       |
| Weighted F1-Score    | 0.6948             | [0.651, 0.739]     | 0.5808        | [0.531, 0.631]       |
+----------------------+--------------------+--------------------+---------------+----------------------+
| Normal Recall        | 93.87% (245/261)   | [90.9%, 96.8%]     | 77.59% (187)  | [72.3%, 82.9%]       |
| CAS (Wheeze) Recall  | 0.00% (0/25)       | [0.0%, 0.0%]       | 1.54% (1/65)  | [0.0%, 4.5%]         |
| DAS (Crackle) Recall | 34.69% (17/49)     | [21.4%, 48.0%]     | 45.83% (11/24)| [25.9%, 65.8%]       |
| CAS & DAS Recall     | 5.26% (1/19)       | [0.0%, 15.3%]      | 23.53% (4/17) | [3.4%, 43.7%]        |
+----------------------+--------------------+--------------------+---------------+----------------------+
```

On the independent held-out test cohort ($N=347$, 41 completely unseen patients), the baseline model achieved **58.50% Accuracy** and **37.12% Balanced Accuracy**. It demonstrated active multi-class discrimination, achieving **45.83% recall on crackles (DAS)** and **23.53% recall on co-occurring sounds (CAS & DAS)**.

### C. Multi-Scale Sliding-Window Ablation
Table IV presents the evaluation of multi-window temporal scanning across complete uncropped audio files.

```
+-------------------------------------------------------------------------------------------------------+
|                       TABLE IV: SLIDING-WINDOW ABLATION ON INDEPENDENT TEST SET (N=347)               |
+---------------------------+--------+---------------+----------+--------+---------+---------+----------+
| Configuration             | Stride | Accuracy      | BAcc     | MacroF1| Normal  | CAS     | DAS      |
|                           | (S)    |               |          |        | Recall  | Recall  | Recall   |
+---------------------------+--------+---------------+----------+--------+---------+---------+----------+
| Center-Crop Baseline      | N/A    | 58.50%        | 37.12%   | 0.3018 | 77.59%  | 1.54%   | 45.83%   |
| Stride-64 Uniform Mean    | 64     | 60.23%        | 32.43%   | 0.2643 | 82.16%  | 0.00%   | 41.67%   |
| Stride-64 Max Normalized  | 64     | 59.94%        | 33.69%   | 0.2758 | 81.33%  | 0.00%   | 41.67%   |
| Stride-64 Top-2 Mean      | 64     | 60.81%        | 34.00%   | 0.2793 | 82.57%  | 0.00%   | 41.67%   |
| Stride-64 Abnormality Max*| 64     | 55.62%        | 34.01%   | 0.2666 | 74.27%  | 0.00%   | 50.00%   |
| Stride-32 Abnormality Max | 32     | 55.04%        | 36.11%   | 0.2777 | 72.61%  | 0.00%   | 54.17%   |
+---------------------------+--------+---------------+----------+--------+---------+---------+----------+
* Primary pre-registered configuration selected on Validation Fold 0.
```

#### Key Findings from Table IV:
1. **Recovery of Transient Crackles:** Moving from fixed center-cropping to multi-scale sliding windows with abnormality-weighted max aggregation increased crackle (*DAS*) recall from 45.83% to **50.00%** ($S=64$) and **54.17%** ($S=32$). Scanning the full recording recovered localized crackle bursts occurring in outer breath cycles that were truncated by center-cropping.
2. **Persistence of the Wheeze Deficit:** Wheeze (*CAS*) recall remained at 0.00% across all sliding-window configurations.
3. **The Dilution Effect of Uniform Averaging:** Uniform mean aggregation increased overall accuracy to 60.23% and Normal recall to 82.16%, but reduced Balanced Accuracy to 32.43% and suppressed co-occurring sound detection (*CAS & DAS*) from 23.53% down to 5.88%.

---

## IX. Discussion & Mechanistic Analysis

### A. Why Crackles Recover While Wheezes Do Not
The divergent behavior between crackles and wheezes under sliding-window scanning reveals fundamental differences in acoustic physics:

- **Crackles (DAS)** are explosive, brief non-musical sounds ($\tau < 20\text{ ms}$) caused by sudden pressure equalization when collapsed small airways snap open during inspiration. They occupy high bandwidths and sharp temporal coordinates. In a 15-second recording, crackles may occur in only one specific inspiration phase. If that phase happens at second 2 or second 13, center-cropping discards it. Sliding windows successfully scan these peripheral regions, allowing the Global Max Pooling branch to detect the high-energy impulse burst and raise crackle recall to 54.17%.
- **Wheezes (CAS)** are continuous musical tones ($\tau > 100\text{ ms}$) caused by airway wall oscillation under airflow limitation. They possess distinct harmonic frequency combs. Because the training set contained only 101 wheeze samples versus 1,042 normal samples (10.3:1 skew) and training was restricted to center-crops, the network filters never learned robust, shift-invariant representations for subtle wheezing harmonics. Post-hoc sliding-window inference at test time cannot recover acoustic features that the underlying convolutional filters never learned to extract.

### B. The Probability Dilution Mechanism
Consider a 15-second recording comprising three complete breathing cycles, where an abnormal crackle event is localized strictly within cycle 1. 

Under sliding-window extraction ($S=64$), window 1 covers cycle 1, producing an abnormal probability $p_{1, \text{DAS}} = 0.68$. Windows 2, 3, and 4 cover cycles 2 and 3, which contain normal vesicular sound, yielding $p_{k, \text{Normal}} \approx 0.88$ ($k \in \{2, 3, 4\}$).

When computing the uniform mean:
$$\bar{p}_{\text{DAS}} = \frac{0.68 + 0.08 + 0.06 + 0.05}{4} = 0.2175$$
$$\bar{p}_{\text{Normal}} = \frac{0.22 + 0.88 + 0.90 + 0.89}{4} = 0.7225$$
The argmax decision flips to **Normal**, effectively erasing the pathological event. This proves that uniform temporal averaging in respiratory audio analysis causes severe diagnostic dilution. Clinical triage requires peak-sensitive aggregation operators (`abnormality_max`) to preserve sensitivity to localized adventitious sounds.

---

## X. Limitations

1. **Weak Recording-Level Supervision:** SPRSound annotations apply to entire recordings rather than specific breathing cycles. When a 15-second recording is labeled *CAS & DAS*, individual breath cycles that are purely normal are penalized during backpropagation, introducing supervisory noise.
2. **Extreme Class Imbalance:** Normal sounds comprise nearly three-quarters of the dataset, while combined adventitious classes account for only 26.5%.
3. **Single-Center Sensor Cohort:** All SPRSound recordings were collected using Yunting Model II digital stethoscopes at Shanghai Children's Medical Center. Cross-hardware generalization to other digital stethoscopes (e.g., Littmann 3200, Eko CORE) remains unvalidated.

---

## XI. Clinical Safety & Ethical Considerations

Machine learning models deployed in pediatric healthcare must be evaluated under rigorous ethical boundaries:

1. **The Hazard of False Reassurance:** A model exhibiting low wheezing recall (1.54%) cannot be used for standalone screening. Failing to detect acute wheezing in a young child presenting with severe bronchiolitis or asthma exacerbation could lead to delayed administration of bronchodilators or oxygen therapy, resulting in critical clinical deterioration.
2. **Decision-Support Boundary:** This framework is strictly an **investigational decision-support aid** designed to provide reproducible baselines for clinical researchers. It does not possess regulatory clearance and must never supersede a licensed physician's clinical judgement.
3. **Transparent Reporting of Negative Results:** Scientific progress in healthcare AI is routinely impeded by publication bias, where negative or moderate results are hidden in favor of overfitted $>95\%$ claims. We deliberately report our wheeze sensitivity deficit and disease-target collapse to establish a transparent, realistic benchmark for future research.

---

## XII. Conclusion

This study established a patient-independent, four-class pediatric respiratory sound classification framework on the SPRSound database. We demonstrated that predicting systemic hospital diagnoses directly from localized audio causes model collapse due to the high prevalence of vesicular sounds in sick children. Formulating the task around acoustic sound categories, our 4-block 2D-CNN with Dual Pooling achieved 58.50% accuracy and 45.83% crackle recall on an independent held-out cohort of 41 unseen patients. 

Furthermore, multi-scale sliding-window scanning with abnormality-weighted max aggregation increased crackle recall to 54.17%, proving that scanning full uncropped recordings recovers boundary-truncated anomalies. However, post-hoc windowing failed to improve wheeze recall, demonstrating that temporal scanning cannot overcome representation deficits caused by extreme class skew without cycle-level annotations. Future work will focus on cycle-level segmentation, self-supervised foundation audio representations, and multi-sensor prospective evaluations.

---

### Acknowledgments
The authors thank the biomedical engineering and clinical staff at Shanghai Children's Medical Center for curating and releasing the open-access SPRSound database.

---

## References

[1] World Health Organization, "Pneumonia in children," *WHO Fact Sheets*, Nov. 2022. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/pneumonia  
[2] H. Pasterkamp, P. S. Kraman, and G. R. Wodicka, "Respiratory sounds: Advances in understanding and applications," *American Journal of Respiratory and Critical Care Medicine*, vol. 156, no. 3, pp. 974–987, Sep. 1997.  
[3] F. Dalmay et al., "Acoustic properties of crackles and wheezes in pediatric respiratory illness: A clinical assessment," *Pediatric Pulmonology*, vol. 48, no. 7, pp. 712–721, Jul. 2013.  
[4] S. Reichert et al., "Analysis of respiratory sounds: State of the art," *Clinics in Chest Medicine*, vol. 29, no. 1, pp. 137–149, Mar. 2008.  
[5] C. B. S. P. Silva et al., "Data leakage in machine learning for healthcare: A systematic review," *Journal of Biomedical Informatics*, vol. 142, p. 104388, Jun. 2023.  
[6] B. M. Rocha et al., "An open access database for the evaluation of respiratory sound classification algorithms," *Physiological Measurement*, vol. 40, no. 3, p. 035001, Mar. 2019.  
[7] Q. Zhang et al., "SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database," *IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)*, vol. 16, no. 5, pp. 867–881, Oct. 2022.  
[8] S. Demir, K. Sengur, and V. Bajaj, "Convolutional neural networks for respiratory sound classification," *Artificial Intelligence in Medicine*, vol. 109, p. 101961, Sep. 2020.  
[9] D. Perna and A. Tagarelli, "Deep auscultation: Predicting respiratory anomalies from lung sounds using convolutional neural networks," in *Proc. IEEE 32nd International Symposium on Computer-Based Medical Systems (CBMS)*, Cordoba, Spain, 2019, pp. 20–25.  
[10] L. Fraiwan, K. Khassawneh, and J. Alkhodari, "Automatic identification of respiratory diseases from stethoscopic lung sound signals using ensemble estimators and deep learning," *Biocybernetics and Biomedical Engineering*, vol. 41, no. 1, pp. 1–14, Jan. 2021.  
[11] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, 2017, pp. 618–626.
