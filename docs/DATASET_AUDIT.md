# SPRSound Dataset Verification & Reconciliation Audit

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Dataset:** SJTU Paediatric Respiratory Sound Database (SPRSound)  
**Official Sources:**
- Repository: [https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
- IEEE BioCAS Respiratory Sound Track Grand Challenges (2022, 2023, 2024, 2025)
- Primary Publication: Q. Zhang et al., *"SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database"*, IEEE Transactions on Biomedical Circuits and Systems (TBioCAS), vol. 16, no. 5, pp. 867–881, Oct. 2022.  
**License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**  
**Audit Date:** 2026-10-21  
**Status:** Physically Audited & 100% Reconciled Across All Partitions

---

## 1. Verified Partition Counts & Reconciliation

Direct file system inspection and SHA-256 hash analysis confirms:

| Partition Name | Storage Path | WAV Count | JSON Count | Pairing Status | Unique Patients |
|---|---|---|---|---|---|
| **Training Set** | `Classification/train_classification_*` | **1,949** | **1,949** | 100% paired (0 missing) | **251** |
| **2022 Inter-Subject Test** | `valid_classification_*/2022/inter_test_json` | **355** | **355** | 100% paired (0 missing) | **41** |
| **2022 Intra-Subject Test** | `valid_classification_*/2022/intra_test_json` | **379** | **379** | 100% paired (0 missing) | **162** |
| **2023 Challenge Test** | `valid_classification_*/2023` | **871** | **871** | 100% paired (0 missing) | **93** |
| **Total SPRSound Dataset** | | **3,554** | **3,554** | **1-to-1 Exact** | **385 unique patients** |

- Total validation directory WAV files: 734 (2022) + 871 (2023) = **1,605 WAV files**.
- Total validation directory JSON files: 355 (inter) + 379 (intra) + 871 (2023) = **1,605 JSON files**.
- Zero missing audio or annotation files detected across the entire 3,554-file dataset.

---

## 2. Audio Signal Verification

All 3,554 `.wav` audio files were read and verified via `wave` and `soundfile`:

- **Container Format:** RIFF WAV, 16-bit PCM Linear
- **Sampling Rate:** **8,000 Hz uniform across 100% of files** ($f_s = 8,000$ Hz, Nyquist = 4,000 Hz)
- **Channels:** Single-channel mono ($ch = 1$)
- **Corrupt Files:** Exactly 0 corrupt audio files detected.
- **Duration Distribution (Training Set):**
  - Minimum: 0.30 seconds
  - Maximum: 15.36 seconds
  - Mean: 10.84 seconds
  - Median: 9.22 seconds

---

## 3. Filename Metadata & Patient Identifiers

Audio and annotation files follow a strict 5-component naming convention:
$$\text{\{patient\_num\}\_\{age\}\_\{gender\}\_\{location\}\_\{recording\_num\}.\dots}$$

1. **`patient_num`**: Unique medical identifier.
2. **`age`**: Age in decimal years (e.g. `0.4` = ~5 months; max 16.2 years; mean 5.8 years). 100% pediatric ($\le 18$ years).
3. **`gender`**: `0` = Male (1,033 in train), `1` = Female (916 in train).
4. **`location`**: Auscultation site:
   - `p1`: Left Posterior (499 in train)
   - `p2`: Left Lateral (493 in train)
   - `p3`: Right Posterior (466 in train)
   - `p4`: Right Lateral (491 in train)
5. **`recording_num`**: Sequence recording number.

### 2023 Partition Patient Identifiers
All 871 files in the 2023 test partition possess standard 5-component filenames. None of the patient IDs are withheld or anonymous; they map to 93 distinct pediatric patients, documented in `Patient Summary/Grand_Challenge_23_patient_summary.csv`.

---

## 4. Patient Leakage & Partition Independence

Cross-referencing patient IDs across dataset partitions revealed:

1. **Train vs. 2022 Inter-Subject Test:**
   $$\text{Train Patients (251)} \cap \text{Inter-Subject Test Patients (41)} = \emptyset \quad \text{(EXACTLY ZERO LEAKAGE)}$$
   The 2022 inter-subject test set serves as the primary leak-free external generalization benchmark.

2. **Train vs. 2023 Test:**
   $$\text{Train Patients (251)} \cap \text{2023 Test Patients (93)} = \emptyset \quad \text{(EXACTLY ZERO LEAKAGE)}$$
   $$\text{2022 Inter-Test Patients (41)} \cap \text{2023 Test Patients (93)} = \emptyset \quad \text{(EXACTLY ZERO LEAKAGE)}$$
   The 2023 test set serves as a completely independent multi-patient secondary benchmark.

3. **Train vs. 2022 Intra-Subject Test:**
   $$\text{Train Patients (251)} \cap \text{Intra-Subject Test Patients (162)} = 162 \quad \text{(100\% OVERLAP)}$$
   **Methodological Rule:** The intra-subject test partition deliberately tests same-patient recognition under varying auscultation sites. It must **never** be combined with independent test partitions or used to claim generalizable diagnostic performance.

4. **Audio Duplicate Detection (SHA-256 Hashes):**
   A full-dataset hash comparison identified **exactly 1 audio duplicate group** across 3,554 files:
   - `train_classification_wav/41279835_14.1_0_p1_2945.wav`
   - `valid_classification_wav/2022/41279835_14.1_0_p1_2050.wav` (intra-subject test)
   These two files share identical SHA-256 byte contents, confirming the intra-subject test contains repeated auscultations from training patients.

---

## 5. Ground-Truth Annotation Structure

### Record-Level Distribution (Task 2 Ground Truth)

| Record Annotation | Training Set | 2022 Inter-Test | 2022 Intra-Test | 2023 Test | Description |
|---|---|---|---|---|---|
| **`Normal`** | 1,303 (66.85%) | 241 (67.89%) | 241 (63.59%) | 539 (61.88%) | Clear vesicular breath sounds |
| **`DAS`** | 248 (12.72%) | 24 (6.76%) | 75 (19.79%) | 133 (15.27%) | Discontinuous Adventitious Sounds (Crackles) |
| **`Poor Quality`** | 177 (9.08%) | 8 (2.25%) | 2 (0.53%) | 43 (4.94%) | High noise / uninterpretable SNR |
| **`CAS`** | 126 (6.46%) | 65 (18.31%) | 42 (11.08%) | 135 (15.50%) | Continuous Adventitious Sounds (Wheezes/Rhonchi) |
| **`CAS & DAS`** | 95 (4.87%) | 17 (4.79%) | 19 (5.01%) | 21 (2.41%) | Both continuous & discontinuous sounds |
| **Total** | **1,949** | **355** | **379** | **871** | |

### Event-Level Annotations (Breathing Cycles)
Training audio contains 6,656 segmented breathing cycles:
- `Normal`: 5,159 (77.51%)
- `Fine Crackle`: 912 (13.70%)
- `Wheeze`: 452 (6.79%)
- `Coarse Crackle`: 49 (0.74%)
- `Rhonchi`: 39 (0.59%)
- `Wheeze+Crackle`: 30 (0.45%)
- `Stridor`: 15 (0.23%)

---

## 6. Clinical Diagnoses Distinction vs. Acoustic Classification

The `Patient Summary` files (`SPRSound_patient_summary.csv` and `Grand_Challenge_23_patient_summary.csv`) contain clinical hospital diagnoses:
- Pneumonia (non-severe): 177 patients (60.6%)
- Asthma: 33 patients (11.3%)
- Other respiratory diseases: 23 patients (7.9%)
- Control Group: 20 patients (6.8%)
- Pneumonia (severe): 20 patients (6.8%)
- Bronchitis: 19 patients (6.5%)

### Critical Methodological Finding
A patient's clinical diagnosis is a systemic hospital admission/discharge diagnosis. It is **not** a localized property of every auscultation chest quadrant:
- A child hospitalized for unilateral pneumonia may present with crackles (DAS) at `p1` (left posterior) while auscultation at `p4` (right lateral) yields completely normal vesicular breath sounds.
- Training a CNN with recording-level cross-entropy against disease labels forces the network to penalize the prediction of "Normal" on physically normal breath sounds.
- Therefore, the defensible, scientifically honest intermediate target is **Acoustic Record Classification** (Task 2-2: Normal, CAS, DAS, CAS & DAS), with Poor Quality routed to quality rejection.
- Clinical disease prediction requires multi-site pooling and clinical evidence integration.

---

## 7. Dataset License & Attribution

The dataset is published under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
Academic attribution:
```bibtex
@article{zhang2022sprsound,
  title={SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database},
  author={Zhang, Q. and others},
  journal={IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)},
  volume={16},
  number={5},
  pages={867--881},
  year={2022}
}
```
