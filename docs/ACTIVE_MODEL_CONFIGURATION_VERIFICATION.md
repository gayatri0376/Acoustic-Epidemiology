# Active Model and Training Configuration Verification

**Project:** Acoustic Epidemiology – Pediatric Respiratory Sound Classification using SPRSound  
**Target Architecture:** Quality-Aware Hierarchical Multi-Task Learning Network (QH-MTLNet)  
**Lead:** Gayatri (Backend & CNN Integration)  
**Task:** Task 1 — Verify Active Model and Training Configuration  
**Audit Mode:** Read-Only Static and Dynamic Inspection  
**Date:** November 2026  
**Status:** Complete  

---

## 1. Executive Summary & Active Database Verification

### Active Database Verification
- **Settings Inspected:** `backend/config/settings.py` (`DATABASES['default']`)
- **Active Database Engine:** `django.db.backends.sqlite3`
- **Active Database Path:** `/Users/gayatri/Downloads/Django/Acoustic-Epidemiology/backend/db.sqlite3`
- **Confirmation:** Confirmed via live Django settings reflection (`settings.DATABASES['default']['NAME']`). There are no secondary or unapplied SQLite databases active.

---

## 2. Independent Verification Table

Each parameter was independently verified across physical `.keras` model containers, companion training configuration JSON files (`*_config.json`), dataset feature extraction manifests, and live Django ORM database records (`ModelVersion`).

| Parameter | Sound Model (`sprsound-cnn-v1-fold0`) | Disease Model (`sprsound-disease-cnn-v2`) | Verification Status |
|---|---|---|---|
| **Task Type** | `SOUND_CLASSIFICATION` (4-class sound category) | `DISEASE_CLASSIFICATION` (3-class pediatric diagnosis) | **VERIFIED** |
| **Model Status** | `ACTIVE` | `ACTIVE` | **VERIFIED** |
| **Artifact Path** | `ml/models/saved_models/sprsound-cnn-v1-fold0.keras` | `ml/models/saved_models/sprsound-disease-cnn-v2.keras` | **VERIFIED** |
| **Artifact Existence & Size** | Exists: `True` (5,172,706 bytes / 4.93 MB) | Exists: `True` (1,303,247 bytes / 1.24 MB) | **VERIFIED** |
| **Artifact Format** | Keras 3 Archive (`.keras` HDF5/zip container) | Keras 3 Archive (`.keras` HDF5/zip container) | **VERIFIED** |
| **Expected Sample Rate** | `8000 Hz` | `8000 Hz` | **VERIFIED** |
| **Bandpass Filtering** | `True` (Zero-phase Butterworth SOS) | `True` (Zero-phase Butterworth SOS) | **VERIFIED** (in training config) |
| **Filter Order** | `5` | `5` | **VERIFIED** |
| **Lower Cutoff ($f_{\min}$)** | `50.0 Hz` | `50.0 Hz` | **VERIFIED** |
| **Upper Cutoff ($f_{\max}$)** | `2000.0 Hz` | `2000.0 Hz` | **VERIFIED** |
| **Normalization Method** | `minmax_01` (scaled to $[0.0, 1.0]$) | `minmax_01` (scaled to $[0.0, 1.0]$) | **VERIFIED** (in training config) |
| **Top dB Ceiling** | `80.0 dB` | `80.0 dB` | **VERIFIED** |
| **FFT Size (`n_fft`)** | `1024` samples (128 ms at 8 kHz) | `1024` samples (128 ms at 8 kHz) | **VERIFIED** |
| **Hop Length (`hop_length`)** | `512` samples (64 ms at 8 kHz) | `512` samples (64 ms at 8 kHz) | **VERIFIED** |
| **Mel Filterbanks (`n_mels`)**| `128` | `128` | **VERIFIED** |
| **Minimum Frequency** | `50.0 Hz` | `50.0 Hz` | **VERIFIED** |
| **Maximum Frequency** | `2000.0 Hz` | `2000.0 Hz` | **VERIFIED** |
| **Expected Tensor Shape** | `(None, 128, 128, 1)` | `(None, 128, 128, 1)` | **VERIFIED** (inspected via Keras) |
| **Output Classes** | `['Normal', 'CAS', 'DAS', 'CAS & DAS']` | `['Healthy Control', 'Pneumonia', 'Wheezing Disorders']` | **VERIFIED** |
| **Class Order / Mapping** | 0: Normal, 1: CAS, 2: DAS, 3: CAS & DAS | 0: Healthy Control, 1: Pneumonia, 2: Wheezing Disorders | **VERIFIED** |
| **Output Activation Mode** | `softmax` (multi-class mutually exclusive) | `softmax` (multi-class mutually exclusive) | **VERIFIED** |
| **Target Conv Layer (Grad-CAM)**| `last_conv_layer` | `last_conv_layer` | **VERIFIED** |

---

## 3. Discrepancy Analysis: Database vs Training Configuration

### 3.1. Sound Model (`sprsound-cnn-v1-fold0`)
- **Companion Training Config (`sprsound-cnn-v1-fold0_config.json:18–28`):**
  ```json
  "preprocessing_config": {
    "sample_rate": 8000,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 512,
    "f_min": 50.0,
    "f_max": 2000.0,
    "apply_bandpass": true,
    "top_db": 80.0,
    "normalization": "minmax_01"
  }
  ```
- **Live Database Record in `db.sqlite3`:**
  ```json
  "preprocessing_config": {
    "sample_rate": 8000,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 512,
    "f_min": 50.0,
    "f_max": 2000.0,
    "to_db": true,
    "expected_input_shape": [null, 128, 128, 1],
    "output_shape": [null, 4],
    "output_activation": "softmax",
    "target_conv_layer": "last_conv_layer"
  }
  ```
- **Discrepancy Verified:**
  - `apply_bandpass` is **missing in DB** (evaluates to `False` at runtime in `services.py:203`).
  - `normalization` is **missing in DB** (evaluates to `None` at runtime in `services.py:204`).
  - `top_db` is **missing in DB**.
  - `filter_order` is **missing in DB**.
  - **Verdict:** The training configuration definitively proves that the model was trained with `apply_bandpass=True` and `normalization="minmax_01"`. The omission in the database is an artifact of `register_model.py` and not a design choice.

### 3.2. Disease Model (`sprsound-disease-cnn-v2`)
- **Companion Training Config (`sprsound-disease-cnn-v2_config.json:18–28`):**
  Matches SPRSound standard (`sample_rate: 8000`, `apply_bandpass: true`, `normalization: "minmax_01"`, `top_db: 80.0`).
- **Live Database Record in `db.sqlite3`:**
  ```json
  "preprocessing_config": {
    "sample_rate": 8000,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 512,
    "f_min": 50.0,
    "f_max": 2000.0,
    "apply_bandpass": true,
    "top_db": 80.0,
    "normalization": "minmax_01"
  }
  ```
- **Discrepancy Verified:**
  - Preprocessing parameters match training configuration perfectly.
  - Omitted fields: `expected_input_shape`, `output_shape`, `output_activation`, `target_conv_layer`. (Does not block inference because `getattr(loaded_model, 'input_shape')` discovers the shape dynamically).

---

## 4. Conclusion & Readiness for Task 2

1. Both models and their physical weights are completely intact, functional, and loadable via Keras 3.
2. The exact training parameters for `sprsound-cnn-v1-fold0` are 100% verified against `sprsound-cnn-v1-fold0_config.json`.
3. The active database has been identified as `/Users/gayatri/Downloads/Django/Acoustic-Epidemiology/backend/db.sqlite3`.
4. We are cleared to proceed with **Task 2**: formulating the database hotfix and code protections for the sound classification model.

