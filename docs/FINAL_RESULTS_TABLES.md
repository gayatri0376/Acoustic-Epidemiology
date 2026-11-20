# Final Research Results & Publication Tables

**Project:** Acoustic Epidemiology: Patient-Independent Classification of Pediatric Respiratory Sounds Using CNN-Based Time-Frequency Analysis  
**Lead:** Mohini (Machine Learning Modeling & CNN Lead)  
**Reviewer:** Gayatri (Senior ML Integration Lead)  
**Status:** ALL METRICS VERIFIED AGAINST MACHINE-READABLE EVALUATION ARTIFACTS  

---

## Table 1: SPRSound Dataset Partitioning & Patient Disjointness

| Partition Group | Split Identifier (`split_category`) | Total Recordings | Excluded Poor Quality | Evaluated Recordings | Patient Count | Patient Disjointness Status |
| :--- | :--- | :---:| :---:| :---:| :---:| :--- |
| **Training (Folds 1–4)** | `train_cv` (folds 1–4) | 1,457 | 39 | 1,418 | 202 | Strict patient isolation (Folds 1–4) |
| **Validation (Fold 0)** | `train_cv` (fold 0) | 369 | 15 | 354 | 49 | 0 overlap with Training Folds 1–4 |
| **Independent Held-Out Test** | `held_out_inter_test` | 355 | 8 | 347 | 41 | 0 overlap with Train/Val Pool |
| **Intra-Subject Test** | `intra_subject_test` | 141 | 5 | 136 | 13 | Segregated (shared patient test) |
| **Secondary Benchmark** | `test_2023` | 361 | 3 | 358 | 51 | 0 overlap with 2022 Partitions |
| **Complete SPRSound Manifest** | **All Categories** | **2,683** | **70** | **2,613** | **356** | **0 Duplicate Records Across Dataset** |

---

## Table 2: Class Distribution Across Evaluated Partitions (4-Class Acoustic Sound)

| Acoustic Sound Class | Training Folds 1–4 ($N=1,418$) | Validation Fold 0 ($N=354$) | Held-Out Test Set ($N=347$) | Full Valid Database ($N=2,613$) |
| :--- | :---:| :---:| :---:| :---:|
| **Normal** | 1,042 (73.48%) | 261 (73.73%) | 241 (69.45%) | 1,880 (71.95%) |
| **CAS (Wheezes)** | 101 (7.12%) | 25 (7.06%) | 65 (18.73%) | 252 (9.64%) |
| **DAS (Crackles)** | 199 (14.03%) | 49 (13.84%) | 24 (6.92%) | 339 (12.97%) |
| **CAS & DAS (Co-occurring)** | 76 (5.36%) | 19 (5.37%) | 17 (4.90%) | 142 (5.43%) |
| **Total** | **1,418 (100.0%)** | **354 (100.0%)** | **347 (100.0%)** | **2,613 (100.0%)** |

---

## Table 3: Acoustic Baseline Model Validation Metrics (Fold 0, $N=354$, Center-Crop)

- **Overall Accuracy:** 74.29%
- **Balanced Accuracy:** 33.46%
- **Macro F1-Score:** 0.3284
- **Weighted F1-Score:** 0.6948

| Class Name | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Support | Precision | Recall | F1-Score |
| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| **Normal** | 245 | 58 | 16 | 261 | 80.86% | **93.87%** | 0.8688 |
| **CAS (Wheezes)** | 0 | 1 | 25 | 25 | 0.00% | **0.00%** | 0.0000 |
| **DAS (Crackles)** | 17 | 29 | 32 | 49 | 36.96% | **34.69%** | 0.3579 |
| **CAS & DAS** | 1 | 3 | 18 | 19 | 25.00% | **5.26%** | 0.0870 |

---

## Table 4: Acoustic Baseline Model Independent Test Metrics (`held_out_inter_test`, $N=347$, Center-Crop)

- **Overall Accuracy:** 58.50%
- **Balanced Accuracy:** 37.12%
- **Macro F1-Score:** 0.3018
- **Weighted F1-Score:** 0.5808

| Class Name | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Support | Precision | Recall | F1-Score |
| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| **Normal** | 187 | 42 | 54 | 241 | 81.66% | **77.59%** | 0.7957 |
| **CAS (Wheezes)** | 1 | 1 | 64 | 65 | 50.00% | **1.54%** | 0.0299 |
| **DAS (Crackles)** | 11 | 80 | 13 | 24 | 12.09% | **45.83%** | 0.1913 |
| **CAS & DAS** | 4 | 21 | 13 | 17 | 16.00% | **23.53%** | 0.1905 |

---

## Table 5: Sliding-Window Aggregation Strategy Comparison (Validation Fold 0, $N=354$)

| Configuration Identifier | Stride ($S$) | Aggregation Method | Accuracy | Balanced Accuracy | Macro F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Selection Status |
| :--- | :---:| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :--- |
| **Center-Crop Baseline** | N/A | Center Crop ($W=128$) | 74.29% | 33.46% | 0.3284 | 93.87% | 0.00% | 34.69% | 5.26% | Baseline reference |
| `stride_64_mean` | 64 | Uniform Mean | 75.42% | 34.25% | 0.3393 | 95.02% | 0.00% | 36.73% | 5.26% | Candidate |
| `stride_64_max` | 64 | Max Normalized | 75.14% | 34.16% | 0.3378 | 94.64% | 0.00% | 36.73% | 5.26% | Candidate |
| `stride_64_top_k` | 64 | Top-2 Mean | 75.42% | 34.25% | 0.3393 | 95.02% | 0.00% | 36.73% | 5.26% | Candidate |
| **`stride_64_abnormality_max`** | **64** | **Abnormality Max** | **74.29%** | **34.29%** | **0.3342** | **93.10%** | **0.00%** | **38.78%** | **5.26%** | **SELECTED (Peak BAcc & DAS)** |
| `stride_32_mean` | 32 | Uniform Mean | 75.14% | 34.16% | 0.3388 | 94.64% | 0.00% | 36.73% | 5.26% | Candidate |
| `stride_32_max` | 32 | Max Normalized | 74.86% | 34.06% | 0.3373 | 94.25% | 0.00% | 36.73% | 5.26% | Candidate |
| `stride_32_top_k` | 32 | Top-2 Mean | 75.14% | 34.16% | 0.3378 | 94.64% | 0.00% | 36.73% | 5.26% | Candidate |
| `stride_32_abnormality_max` | 32 | Abnormality Max | 74.01% | 34.19% | 0.3336 | 92.72% | 0.00% | 38.78% | 5.26% | Candidate |

---

## Table 6: Sliding-Window Independent Benchmark (`held_out_inter_test`, $N=347$)

| Configuration Identifier | Stride ($S$) | Aggregation Method | Accuracy | Balanced Accuracy | Macro F1 | Normal Recall | CAS Recall | DAS Recall | CAS & DAS Recall | Protocol Role |
| :--- | :---:| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :--- |
| **Center-Crop Baseline** | N/A | Center Crop | 58.50% | **37.12%** | **0.3018** | 77.59% | **1.54%** | 45.83% | **23.53%** | Independent Baseline |
| `stride_64_mean` | 64 | Uniform Mean | **60.23%** | 32.43% | 0.2643 | **82.16%** | 0.00% | 41.67% | 5.88% | Ablation |
| `stride_64_max` | 64 | Max Normalized | 59.94% | 33.69% | 0.2758 | 81.33% | 0.00% | 41.67% | 11.76% | Ablation |
| `stride_64_top_k` | 64 | Top-2 Mean | **60.81%** | 34.00% | 0.2793 | **82.57%** | 0.00% | 41.67% | 11.76% | Ablation |
| **`stride_64_abnormality_max`** | **64** | **Abnormality Max** | **55.62%** | **34.01%** | **0.2666** | **74.27%** | **0.00%** | **50.00%** | **11.76%** | **PRIMARY LOCKED RESULT** |
| `stride_32_abnormality_max` | 32 | Abnormality Max | 55.04% | 36.11% | 0.2777 | 72.61% | 0.00% | **54.17%** | 17.65% | Exploratory Post-Hoc Analysis |

---

## Table 7: Class-Wise Performance Comparison on Independent Test Set ($N=347$)

| Class Name | Support | Center-Crop Precision | Center-Crop Recall | Center-Crop F1 | Stride-64 AbnMax Precision | Stride-64 AbnMax Recall | Stride-64 AbnMax F1 | Net Recall Shift |
| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---: |
| **Normal** | 241 | 81.66% | 77.59% | 0.7957 | 80.99% | 74.27% | 0.7749 | **-3.32%** |
| **CAS (Wheezes)** | 65 | 50.00% | 1.54% | 0.0299 | 0.00% | 0.00% | 0.0000 | **-1.54%** |
| **DAS (Crackles)** | 24 | 12.09% | 45.83% | 0.1913 | 11.43% | 50.00% | 0.1860 | **+4.17%** |
| **CAS & DAS** | 17 | 16.00% | 23.53% | 0.1905 | 9.52% | 11.76% | 0.1053 | **-11.77%** |

---

## Table 8: Computational Latency and Resource Footprint Profile

| Performance Metric | Fixed Center-Crop | Sliding-Window ($S=64$) | Sliding-Window ($S=32$) | Hardware & Environment Specifications |
| :--- | :---:| :---:| :---:| :--- |
| **Evaluated Tensors / Recording** | 1.00 tensor | 2.20 tensors (avg) | 2.61 tensors (avg) | Apple M3 Max (macOS 15), Python 3.13 |
| **Batch Evaluation Time ($N=347$)**| 0.61 seconds | 1.25 seconds | 1.48 seconds | TensorFlow/Keras headless execution |
| **Per-Sample Inference (Batch)** | **1.8 ms** | **3.6 ms** | **4.3 ms** | Batch size = 64 |
| **Online Single-Sample Latency** | **5.2 ms** | **11.8 ms** | **14.1 ms** | Simulated HTTP request forward pass |
| **Peak Tensor Memory / Sample** | ~0.5 MB | ~1.1 MB | ~1.3 MB | 32-bit floating point arrays |
| **Backend Integration Viability** | Real-time ready | Real-time ready | Real-time ready | Well within standard <50 ms web threshold |

---

## Table 9: Comprehensive Summary of Scientific & Clinical Limitations

| Dimension | Observed Characteristic | Impact on Clinical Validity | Required Future Investigation |
| :--- | :--- | :--- | :--- |
| **Supervision Granularity** | Weak recording-level annotations (8–15s). | Model penalizes normal cycles within abnormal recordings. | Cycle-level bounding box segmentation. |
| **Class Imbalance** | 10.32 : 1 (Normal to Wheeze in training). | Suppresses gradient updates for wheezing harmonic templates. | Few-shot learning, balanced generative replay. |
| **Wheeze Detection** | 0.00% recall under sliding-window inference. | Critical clinical blind spot for asthma and bronchiolitis. | High-resolution Mel bins (256 bins), harmonic STFT. |
| **Generalization Domain** | Single clinical center (SCMC, Shanghai). | Risk of acoustic sensor and demographic bias. | Multi-center external validation (multi-device cohort). |
| **Diagnostic Deployment** | Balanced Accuracy $\le 37.12\%$. | **Unsuitable for clinical triage or diagnostic advice.** | Complete model retraining with foundation backbones. |
