# Machine Learning Team Handoff & Model Delivery Guide

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Dataset:** SPRSound (SJTU Paediatric Respiratory Sound Database) CC BY 4.0  
**Date:** 2026-10-21  
**Status:** Preprocessing, Training Pipeline, Evaluation Engine, and Django Inference Fully Harmonized & Verified

---

## 1. Executive Summary

The entire acoustic epidemiology engineering pipeline has been implemented, standardized, and validated with 78 automated tests (47 Django tests, 18 audio preprocessing tests, 10 feature extraction and patient-leakage tests, and 3 evaluation engine tests).

All components are now completely harmonized around a single, unified contract:
1. **Deterministic Label Mapping:** Documented at [`ml/features/extracted/label_mapping.json`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/features/extracted/label_mapping.json).
2. **Standardized Preprocessing Contract:** Implemented identically across [`ml/preprocessing/respiratory_preprocessor.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/preprocessing/respiratory_preprocessor.py) and [`backend/ml_inference/services.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/backend/ml_inference/services.py).
3. **Training Pipeline:** Fully implemented at [`ml/training/train_cnn.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/training/train_cnn.py) with patient-grouped folds and class-balanced weights.
4. **Evaluation Engine:** Fully implemented at [`ml/evaluation/evaluate_model.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/evaluation/evaluate_model.py) with independent benchmarks (2022 inter-subject test and 2023 test).
5. **Model Registry & Safe Inference:** Implemented in `backend/ml_inference/management/commands/register_model.py` and `backend/ml_inference/services.py`.

---

## 2. Standardized Preprocessing Specification

Both training and production inference utilize the exact same acoustic parameters:

| Parameter | Production Contract Value | Rationale / Physiological Basis |
|---|---|---|
| **Sampling Rate ($f_s$)** | **8,000 Hz** | Native SPRSound sampling frequency (Nyquist = 4,000 Hz) |
| **Channels** | **1 (Mono)** | Auscultation audio averaged across channels if stereo |
| **Bandpass Filter** | **50 Hz – 2,000 Hz** | 5th-order zero-phase Butterworth SOS filter; eliminates heart sounds and noise |
| **Mel Filterbanks ($n\_mels$)** | **128** | Frequency bin resolution |
| **STFT Window Length ($n\_fft$)** | **1,024** | 128 ms frame window |
| **Hop Length ($hop\_length$)** | **512** | 64 ms frame step (50% overlap) |
| **Decibel Ceiling ($top\_db$)** | **80.0 dB** | Dynamic range ceiling relative to peak amplitude |
| **Normalization** | **`minmax_01`** | Linear scaling to $[0.0, 1.0]$: $\text{clip}((\text{dB} + 80)/80, 0, 1)$ |
| **Record Tensor Dimensions** | **`(1, 128, 128, 1)`** | Fixed width 128 time frames (symmetric center-pad / center-crop) |
| **Event Tensor Dimensions** | **`(1, 128, 40, 1)`** | Segmented breath cycle (2.5s window at 64 ms hop) |

---

## 3. Label Mapping & ML Classification Targets

Defined in `ml/features/extracted/label_mapping.json`:

### Primary Target: `SPRSOUND_RECORD_4CLASS`
Acoustic classification of auscultation recordings into baseline and adventitious respiratory categories:
- `0`: **`Normal`** (clear vesicular breath sounds)
- `1`: **`CAS`** (Continuous Adventitious Sounds: wheezes, rhonchi)
- `2`: **`DAS`** (Discontinuous Adventitious Sounds: crackles)
- `3`: **`CAS & DAS`** (co-occurring continuous and discontinuous sounds)
- **Quality Rejection:** Recordings annotated as `Poor Quality` are excluded from diagnostic classification and routed to frontline signal-to-noise rejection.

### Important Distinction: Clinical Disease vs. Acoustic Sound
Clinical disease labels (e.g., Pneumonia, Asthma, Bronchitis) are patient-level diagnoses from hospital discharge summaries. They must **never** be naively assigned as recording-level ground truth, because focal diseases (like lobar pneumonia) produce crackles at one chest location (e.g. `p1`) while other quadrants (e.g. `p4`) contain entirely normal breath sounds.

---

## 4. Training and Evaluation Execution Workflow

### Step 1: Install ML Dependencies (Requires Approval)
```bash
backend/.venv/bin/pip install tensorflow keras h5py matplotlib
```

### Step 2: Extract Training Features
```bash
backend/.venv/bin/python ml/features/extract_features.py --mode train
```

### Step 3: Run CNN Training
```bash
backend/.venv/bin/python ml/training/train_cnn.py \
  --task SPRSOUND_RECORD_4CLASS \
  --val-fold 0 \
  --epochs 30 \
  --batch-size 32
```
This generates:
- Model weights: `ml/models/saved_models/sprsound-cnn-v1-fold0.keras`
- Training history: `ml/models/saved_models/sprsound-cnn-v1-fold0_history.json`
- Model configuration: `ml/models/saved_models/sprsound-cnn-v1-fold0_config.json`

### Step 4: Evaluate on Independent Held-Out Benchmark
```bash
backend/.venv/bin/python ml/evaluation/evaluate_model.py \
  --model-artifact ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --partition inter_2022 \
  --task SPRSOUND_RECORD_4CLASS
```

### Step 5: Register the Model in Django
```bash
backend/.venv/bin/python backend/manage.py register_model \
  --model-version sprsound-cnn-v1-fold0 \
  --artifact-path ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --classes Normal CAS DAS "CAS & DAS" \
  --status ACTIVE \
  --architecture 2D-CNN-Spectrogram-4Block \
  --target-conv-layer last_conv_layer \
  --set-active
```
The model is now deployed in production. The Django REST API will automatically load the model into memory, execute forward passes for incoming audio, compute Grad-CAM heatmaps on request, and record clinician reviews.
