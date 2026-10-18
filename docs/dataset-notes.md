# SPRSound: SJTU Paediatric Respiratory Sound Database & Benchmark Notes

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Dataset Lead:** Gayatri (Backend & ML Integration)  
**Date:** October 2026  
**Status:** Downloaded, Audited, Verified & Ready for Preprocessing  

---

## 1. Executive Summary & Source Verification

The official primary dataset acquired for this research is the **SPRSound (SJTU Paediatric Respiratory Sound Database)**, created by the Shanghai Jiao Tong University BioCAS Research Group in collaboration with Shanghai Children's Medical Center (SCMC).

- **Official Repository:** `https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound`
- **Challenge Portal:** IEEE BioCAS Grand Challenge on Respiratory Sound Classification (`http://1.117.17.41/grand-challenge/`)
- **Primary Publication:** Q. Zhang et al., *"SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database"*, IEEE Transactions on Biomedical Circuits and Systems (TBioCAS), vol. 16, no. 5, pp. 867–881, Oct. 2022.
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** — Publicly accessible, permits academic and commercial use with proper attribution.
- **Access Protocol:** Public download directly from official repository; no login gating, registration, or manual credential acceptance required.

---

## 2. Dataset Comparative Analysis: SPRSound vs. ICBHI 2017

| Dimension | SPRSound (Primary Research Dataset) | ICBHI 2017 (Benchmark / Reference) | Project Suitability Verdict |
|---|---|---|---|
| **Target Population** | **100% Pediatric** (Ages: 2.4 months to 16.2 years; mean 5.8y) | **Predominantly Adult / Elderly** (Ages 0.2 to 93 years; mean ~66y) | **SPRSound is the ONLY dataset tailored to pediatric pulmonology.** ICBHI is mostly adult COPD. |
| **Auscultation Hardware** | Yunting Model II Electronic Stethoscope | Heterogeneous (Littmann, WelchAllyn, Inhealth, Meditron) | SPRSound has consistent sensor frequency response. |
| **Sampling Rate** | **8,000 Hz** (Uniform across all files) | Variable (4,000 Hz, 10,000 Hz, 44,100 Hz) | SPRSound provides uniform sampling; Nyquist = 4,000 Hz. |
| **Channels & Format** | Mono (1 channel), 16-bit PCM Linear WAV | Mono/Stereo, 16/24-bit WAV | SPRSound requires no channel decimation. |
| **Clinical Diagnoses** | **Provided per patient:** Pneumonia, Bronchitis, Asthma, Control | **Provided per patient:** COPD (dominant), Asthma, URTI, LRTI | SPRSound aligns directly with pediatric disease objectives (pneumonia, bronchitis). |
| **Sound Event Labels** | Normal, Wheeze, Fine Crackle, Coarse Crackle, Rhonchi, Stridor, Wheeze+Crackle | Normal, Crackle, Wheeze, Both (Crackle+Wheeze) | SPRSound has richer fine-grained sound event categories (including stridor, rhonchi). |
| **Record-Level Labels** | Normal, CAS, DAS, CAS & DAS, Poor Quality | None (cycle-level annotations only) | SPRSound provides explicit high-level record quality & classification. |
| **License** | **CC BY 4.0 (Open Access)** | Academic / Scientific Challenge Only | SPRSound has formal, permissive open-source licensing. |

---

## 3. Directory Layout on Local Disk

All raw audio files and annotations are placed in `ml/dataset/SPRSound/` and kept strictly separated from code, test fixtures, and future preprocessed feature tensors:

```
ml/dataset/
├── download_classification.py          # Automated idempotent fetch script
├── verify_dataset.py                   # Rigorous integrity and leakage audit script
└── SPRSound/
    ├── LICENSE                         # Official CC BY 4.0 license text
    ├── README.md                       # Official SJTU repository documentation
    ├── Patient Summary/
    │   ├── SPRSound_patient_summary.csv        # 292 patients with clinical disease diagnoses
    │   ├── Grand_Challenge_23_patient_summary.csv # Challenge 2023 patient metadata
    │   └── Grand_Challenge_24_patient_summary.csv # Challenge 2024 patient metadata (795 patients)
    └── Classification/
        ├── train_classification_wav/   # 1,949 training WAV audio recordings (8 kHz, 16-bit mono)
        ├── train_classification_json/  # 1,949 training JSON annotations
        ├── valid_classification_wav/   # 1,605 validation/test WAV recordings
        │   ├── 2022/                   # BioCAS 2022 validation WAV files
        │   └── 2023/                   # BioCAS 2023 challenge validation WAV files
        └── valid_classification_json/  # 1,605 validation/test JSON annotations
            ├── 2022/
            │   ├── inter_test_json/    # 355 inter-subject test annotations (ZERO patient leakage)
            │   └── intra_test_json/    # 379 intra-subject test annotations
            └── 2023/                   # 871 test annotations from 2023 challenge
```

---

## 4. Filename Naming Convention & Extracted Metadata

Every `.wav` audio recording and corresponding `.json` annotation follows a strict, standardized 5-component naming convention:

$$\text{\{patient\_num\}\_\{age\}\_\{gender\}\_\{location\}\_\{recording\_num\}.\dots}$$

1. **`patient_num`** (e.g., `40069321`, `65101170`): Unique patient medical identifier.
2. **`age`** (e.g., `0.4`, `5.0`, `15.3`): Age in decimal years (infants < 1y shown as fractional years).
3. **`gender`**:
   - `0`: Male
   - `1`: Female
4. **`location`** (Auscultation chest position):
   - `p1`: Left Posterior
   - `p2`: Left Lateral
   - `p3`: Right Posterior
   - `p4`: Right Lateral
5. **`recording_num`** (e.g., `981`, `2557`): Unique recording sequence number.

---

## 5. Verified Data Statistics & Ground-Truth Labels

### 5.1 Training Set Audio Specifications (1,949 files)
- **Container / Format:** RIFF WAV, 16-bit PCM Linear
- **Sample Rate ($f_s$):** **8,000 Hz** (Uniform across 100% of files)
- **Channels:** Single-channel Mono ($ch = 1$)
- **Duration Distribution:**
  - Minimum Duration: **8.51 seconds**
  - Maximum Duration: **15.36 seconds**
  - Mean Duration: **10.84 seconds**
  - Total Audio Length: ~5.87 hours of pediatric recordings in train set alone
- **Signal Integrity:** 0 corrupted audio files; 1-to-1 exact pairing with JSON annotations.

### 5.2 Record-Level Annotations (1,949 files)

| Record Annotation | Description | Train Count | Percentage |
|---|---|---|---|
| **`Normal`** | Clear vesicular breath sounds without adventitious noises | 1,303 | 66.85% |
| **`DAS`** | Discontinuous Adventitious Sounds (Crackles present) | 248 | 12.72% |
| **`Poor Quality`** | High ambient noise, movement artifact, or low acoustic SNR | 177 | 9.08% |
| **`CAS`** | Continuous Adventitious Sounds (Wheezes or Rhonchi present) | 126 | 6.46% |
| **`CAS & DAS`** | Combined continuous and discontinuous adventitious sounds | 95 | 4.87% |
| **Total** | | **1,949** | **100.00%** |

### 5.3 Event-Level Annotations (6,656 segmented breathing cycles)

Every file contains expert time-stamped respiratory segments (`start` and `end` in milliseconds):

| Event Class | Category | Event Count | Percentage |
|---|---|---|---|
| **`Normal`** | Normal breath cycle | 5,159 | 77.51% |
| **`Fine Crackle`** | Discontinuous adventitious sound (high frequency, short duration) | 912 | 13.70% |
| **`Wheeze`** | Continuous adventitious sound (musical, high pitched) | 452 | 6.79% |
| **`Coarse Crackle`** | Discontinuous adventitious sound (low frequency, longer duration) | 49 | 0.74% |
| **`Rhonchi`** | Continuous adventitious sound (low pitched, snoring/rattling) | 39 | 0.59% |
| **`Wheeze+Crackle`** | Co-occurring wheezing and crackling in same cycle | 30 | 0.45% |
| **`Stridor`** | High-pitched monophonic inspiratory sound | 15 | 0.23% |
| **Total Events** | | **6,656** | **100.00%** |

### 5.4 Patient Demographics & Clinical Disease Diagnoses
Cross-referencing the 251 unique patients in the training set against `SPRSound_patient_summary.csv` yields an exact 100% match:

| Clinical Diagnosis | Unique Patients (Train) | Percentage | Clinical Pulmonology Context |
|---|---|---|---|
| **`Pneumonia (non-severe)`** | 155 | 61.75% | Primary clinical pathology (crackles dominant) |
| **`Asthma`** | 25 | 9.96% | Obstructive airway disease (wheezes dominant) |
| **`Other respiratory diseases`**| 21 | 8.37% | Upper airway / mixed pathologies |
| **`Control Group`** | 18 | 7.17% | Healthy pediatric controls |
| **`Pneumonia (severe)`** | 16 | 6.37% | Advanced consolidation / respiratory distress |
| **`Bronchitis`** | 16 | 6.37% | Bronchial inflammation |
| **Total Unique Patients** | **251** | **100.00%** | |

- **Age Distribution:** 0.2 years (2.4 months) to 16.2 years. Mean: **5.8 years**.
- **Gender Balance:** 1,033 recordings male ($gen=0$), 916 recordings female ($gen=1$).
- **Auscultation Site Balance:** p1: 499, p2: 493, p3: 466, p4: 491 (evenly distributed across the 4 primary chest quadrants).

---

## 6. Patient Leakage Analysis & Data Splitting Strategy

### 6.1 The Risk of Patient Leakage
In pediatric auscultation datasets, multiple recordings are collected from the same child at different chest positions (`p1`–`p4`) or at sequential minutes during the exam. On average, the dataset contains **7.8 recordings per patient**.

- **Flawed Protocol (Random Recording-Level Split):**  
  If audio recordings are randomly assigned using `train_test_split(shuffle=True)`:
  $$\text{Patient } 40069321 \text{ (location } p1) \in \text{Train}, \quad \text{Patient } 40069321 \text{ (location } p2) \in \text{Test}$$
  The CNN inadvertently memorizes patient-specific acoustic transfer functions, body habitus resonances, and room noise signatures. This causes severe optimism bias (e.g. 95% validation accuracy) that completely fails when deployed on new patients in clinic.

### 6.2 Patient-Level Splitting Protocol (Zero Leakage)
1. **Official Inter-Subject Split:**
   The SPRSound dataset officially isolates an **`inter_test_json`** partition (355 recordings). Verification confirms:
   $$\text{Train Patients} \cap \text{Inter-Subject Test Patients} = \emptyset \quad \text{(Exactly Zero Overlap)}$$
   This partition serves as the authoritative, leak-free test benchmark.
2. **K-Fold Cross-Validation:**
   When training CNN models on the training set, use **`GroupKFold(n_splits=5)`** or **`GroupShuffleSplit`** grouping exclusively by `patient_num`. All recordings belonging to patient $P_i$ are assigned strictly to either the train or validation fold, never both.

---

## 7. Known Limitations & Preprocessing Guidelines

1. **Native Sample Rate is 8,000 Hz:**
   The audio was recorded at 8,000 Hz (Nyquist frequency 4,000 Hz). The backend audio preprocessor's initial default was 16,000 Hz; during preprocessing, the pipeline should either operate directly at 8,000 Hz or resample cleanly without hallucinating higher frequencies. The 50 Hz – 2000 Hz bandpass filter is fully supported by this sampling rate.
2. **Class Imbalance in Event Types:**
   Normal breath cycles represent 77.5% of events. Rare events (Stridor: 15 events, Rhonchi: 39 events) require class weighting, focal loss, or grouping (e.g., Continuous Adventitious Sound [CAS] vs Discontinuous Adventitious Sound [DAS]) for stable CNN training.
3. **Signal Quality Handling:**
   177 recordings (9.08%) are designated `Poor Quality` due to crying, sensor friction, or ambient pediatric ward noise. These can be used to train a frontline SNR/Quality rejection classifier or filtered out from diagnostic training.

---

## 8. Next Preprocessing Steps

1. **Audio Preprocessor Parameter Harmonization:**
   Update `AudioPreprocessor` default sample rate to 8,000 Hz, matching SPRSound native rate.
2. **Slice Generation (Event vs. Whole-Recording):**
   - For Event-Level Classification: Extract 6,656 segmented audio intervals based on `event_annotation` timestamps (`start` to `end`), zero-padded to uniform windows (e.g., 2.5s).
   - For Record-Level Classification: Apply a sliding window (e.g. 2.5s or 5.0s with 50% overlap) or pool features across the full recording (~10.8s).
3. **Time-Frequency Representation:**
   Generate Log Mel-Spectrograms ($n\_mels = 128$, $n\_fft = 1024$, $hop\_length = 512$, $f_{min} = 50\text{ Hz}$, $f_{max} = 2000\text{ Hz}$, $\text{to\_db} = \text{True}$) to match the backend inference specification.
4. **Offline Feature Cache:**
   Save extracted Mel-spectrogram tensors in `ml/features/` separated strictly by `patient_id` groups.
