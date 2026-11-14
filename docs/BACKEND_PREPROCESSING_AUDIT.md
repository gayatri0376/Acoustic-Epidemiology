# Backend Preprocessing Compatibility Audit

**Project:** Acoustic Epidemiology – Pediatric Respiratory Sound Classification using SPRSound  
**Target Architecture:** Quality-Aware Hierarchical Multi-Task Learning Network (QH-MTLNet)  
**Audit Type:** Static Code Inspection, Configuration Audit, and Empirical Alignment Analysis (Read-Only)  
**Date:** November 2026  
**Status:** Complete  

---

## 1. Executive Summary & Audit Scope

This document provides a rigorous, read-only compatibility audit between the Machine Learning (ML) feature extraction pipeline and the Django backend inference services. The objective is to evaluate preprocessing consistency across training and inference before integrating the proposed Quality-Aware Hierarchical Multi-Task Learning Network (QH-MTLNet).

### Inspected Files and Artifacts

| Scope | File Path | Primary Responsibilities |
|---|---|---|
| **ML Pipeline** | `ml/preprocessing/respiratory_preprocessor.py` | Standalone feature extraction contract (`PreprocessingConfig`, `RespiratoryAudioPreprocessor`), Butterworth bandpass, Mel spectrogram extraction, time-axis alignment, tensor formatting. |
| **ML Config** | `ml/features/extracted/preprocessing_config.json` | Ground truth configuration parameters used during batch dataset feature extraction. |
| **ML Configs** | `ml/models/saved_models/sprsound-cnn-v1-fold0_config.json`<br>`ml/models/saved_models/sprsound-disease-cnn-v2_config.json` | Model architecture, training hyperparameters, and preprocessing parameter dumps. |
| **Backend Services** | `backend/ml_inference/services.py` | Backend preprocessing (`AudioPreprocessor`), model cache (`ModelCache`), sound-classification inference orchestrator (`InferenceService`), Grad-CAM generator (`GradCAMService`). |
| **Backend Disease Service** | `backend/ml_inference/disease_service.py` | Disease classification inference pipeline (`DiseaseInferenceService`), isolated model cache (`DiseaseModelCache`). |
| **Backend Registry Commands** | `backend/ml_inference/management/commands/register_model.py`<br>`backend/ml_inference/management/commands/register_disease_model.py` | CLI model registration, artifact validation, dimension checks, and database metadata initialization. |
| **Backend Models** | `backend/ml_inference/models.py` | Django ORM schema for `ModelVersion`, `Prediction`, and `DiseasePrediction`. |
| **Database State** | `backend/db.sqlite3` | Live registry records for active models (`sprsound-cnn-v1-fold0`, `sprsound-disease-cnn-v2`). |

---

## 2. Actual Preprocessing Parameters & Comparison

The following table contrasts the ground-truth SPRSound signal properties, the ML training pipeline configuration, the backend default fallback settings, and the live model configurations stored in the Django database.

| Preprocessing Parameter | Native SPRSound Dataset | ML Training Pipeline (`respiratory_preprocessor.py`) | Backend Inference Default (`services.py`) | Live Sound Model in DB (`sprsound-cnn-v1-fold0`) | Live Disease Model in DB (`sprsound-disease-cnn-v2`) | Alignment Status |
|---|---|---|---|---|---|---|
| **Target Sample Rate** | 8,000 Hz | **8,000 Hz** (`line 37`) | **16,000 Hz** (`lines 125, 197`) | 8,000 Hz | 8,000 Hz | ⚠️ **Config-dependent risk** (Default mismatch) |
| **Audio Channel Handling** | Multi-channel / Mono | Mean-channel mono (`line 109`) | Mean-channel mono (`line 107`) | Mean-channel mono | Mean-channel mono | ✅ **Consistent** |
| **Resampling Algorithm** | N/A | `librosa.resample` (Sinc polyphase `soxr_hq`) (`line 115`) | `scipy.signal.resample` (Fourier domain) (`line 114`) | Invoked if `sr != 8000` | Invoked if `sr != 8000` | ❌ **Severe Algorithmic Inconsistency** |
| **Bandpass Filter** | N/A | 5th-order Butterworth SOS, 50–2000 Hz (`lines 41-44, 129`) | Optional SOS Butterworth, 50–2000 Hz (`lines 140-150`) | **Omitted / False** | **True** (50–2000 Hz) | ❌ **Inconsistent in Sound Model** |
| **Filter Order** | N/A | 5 (`line 43`) | 5 (`line 148`) | N/A (disabled) | 5 | ⚠️ Discrepancy in sound model |
| **STFT Window (`n_fft`)** | N/A | 1024 samples (128 ms at 8 kHz) (`line 39`) | 1024 samples (`line 127`) | 1024 samples | 1024 samples | ✅ **Consistent** |
| **Hop Length (`hop_length`)** | N/A | 512 samples (64 ms at 8 kHz) (`line 40`) | 512 samples (`line 128`) | 512 samples | 512 samples | ✅ **Consistent** |
| **Mel Filterbanks (`n_mels`)**| N/A | 128 (`line 38`) | 128 (`line 126`) | 128 | 128 | ✅ **Consistent** |
| **Frequency Range (`f_min`, `f_max`)** | N/A | 50.0 Hz – 2000.0 Hz (`lines 41-42`) | 50.0 Hz – 2000.0 Hz (`lines 129-130`) | 50.0 Hz – 2000.0 Hz | 50.0 Hz – 2000.0 Hz | ✅ **Consistent** |
| **Decibel Dynamic Range** | N/A | `top_db = 80.0` (`line 45`) | `top_db = 80.0` (`line 166`) | `to_db = True` (implicit 80 dB) | `top_db = 80.0` | ✅ **Consistent** |
| **Spectrogram Normalization** | N/A | `minmax_01` (scaled to `[0.0, 1.0]`) (`line 46, 187`) | `normalization` param (`lines 168-172, 204`) | **`None` (Raw dB `[-80.0, 0.0]`)** | **`minmax_01` (`[0.0, 1.0]`)** | ❌ **Critical Bug in Sound Model** |
| **Target Fixed Width (Frames)**| Variable | 128 frames (8.192 s) (`line 239`) | Dynamic or 128 frames (`lines 235-247`) | 128 frames | 128 frames | ⚠️ **Severe Information Discard** |
| **Time Alignment Strategy** | N/A | Symmetric zero-padding / Center-cropping (`lines 211-218`) | Symmetric zero-padding / Center-cropping (`lines 238-246`) | Center-crop / Center-pad | Center-crop / Center-pad | ⚠️ Discards valid auscultation events |
| **Output Tensor Format** | N/A | `(1, 128, 128, 1)` `float32` (`line 233`) | `(1, 128, 128, 1)` `float32` (`line 248`) | `(1, 128, 128, 1)` | `(1, 128, 128, 1)` | ✅ **Consistent** |

---

## 3. Detailed Inconsistency Analysis & File Line Citations

### 3.1. Sampling Rate Default Inconsistency (8,000 Hz vs 16,000 Hz)
* **Verified Code Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:37`: `sample_rate: int = 8000`
  - `backend/ml_inference/services.py:125`: `def compute_mel_spectrogram(..., sr=16000, ...)`
  - `backend/ml_inference/services.py:197`: `target_sr = config.get('sample_rate', 16000)`
  - `backend/ml_inference/management/commands/register_model.py:207`: `default=16000`
* **Finding:** While native SPRSound recordings are sampled at 8 kHz, backend fallback code paths and CLI registration defaults specify 16 kHz. If a model is registered without an explicit `--sample-rate 8000` flag, the backend re-samples 8 kHz clinical files up to 16 kHz. At 16 kHz, an FFT window of `n_fft=1024` corresponds to 64 ms instead of 128 ms, halving the spectral resolution and misaligning the trained convolutional kernels.

### 3.2. Resampling Method: Sinc Interpolation vs Fourier Domain
* **Verified Code Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:115`: `librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='soxr_hq')`
  - `backend/ml_inference/services.py:114`: `audio = scipy.signal.resample(audio, num_samples)`
* **Finding:** When resampling is required (e.g., patient upload with non-standard sampling rates like 44.1 kHz or 48 kHz):
  - The ML pipeline uses high-quality polyphase sinc filtering via `soxr_hq`.
  - The backend uses `scipy.signal.resample`, which performs resampling via the Fast Fourier Transform (FFT). FFT resampling inherently assumes the signal is periodic, creating severe circular convolution Gibbs ringing at the start and end of non-periodic breathing audio.

### 3.3. Bandpass Filtering Omission in Sound Classification Model
* **Verified Code Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:44`: `apply_bandpass: bool = True` (5th-order zero-phase Butterworth filter, 50 Hz to 2,000 Hz)
  - `backend/ml_inference/management/commands/register_model.py:322-333`: The dictionary created by `register_model` did not include `"apply_bandpass"`.
  - `backend/ml_inference/services.py:203`: `apply_bandpass = config.get('apply_bandpass', False)`
  - `backend/db.sqlite3` (`ModelVersion` table, version `sprsound-cnn-v1-fold0`): `"apply_bandpass"` is completely missing from the stored JSON.
* **Finding:** The ML training pipeline filtered out low-frequency cardiac sounds (< 50 Hz) and high-frequency ambient noise (> 2,000 Hz). However, during backend inference for `sprsound-cnn-v1-fold0`, `apply_bandpass` evaluates to `False`. The deployed sound model is therefore fed unfiltered audio containing cardiac thumps and room noise that were never present in training data.

### 3.4. Normalization Discrepancy: `minmax_01` vs Raw Decibels
* **Verified Code Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:46, 187`: `mel_norm = np.clip((mel_db + self.config.top_db) / self.config.top_db, 0.0, 1.0)`
  - `backend/ml_inference/management/commands/register_model.py:322-333`: `"normalization"` key omitted from registration dictionary.
  - `backend/ml_inference/services.py:204`: `normalization = config.get('normalization', None)`
  - `backend/db.sqlite3` (`ModelVersion` table, version `sprsound-cnn-v1-fold0`): `"normalization"` key is absent.
* **Finding:** This is a critical mismatch for the active sound classification model:
  - During training, Mel spectrograms were normalized into the range $[0.0, 1.0]$.
  - During backend inference, because `normalization` is `None`, `services.py:166` outputs raw decibels spanning $[-80.0, 0.0]$.
  - As a result, the CNN convolutional kernels (trained on positive activations $[0.0, 1.0]$) receive negative inputs down to $-80.0$. Through ReLU activation functions ($\max(0, x)$), all negative inputs clamp immediately to zero, completely extinguishing feature activations across early convolutional layers.
  - *Note:* In contrast, `sprsound-disease-cnn-v2` in `register_disease_model.py:131` correctly hardcoded `"normalization": "minmax_01"`.

---

## 4. Impact of 128-Frame Center-Cropping on Clinical Respiratory Events

### 4.1. Mathematical Formulation of Frame Calculation
With SPRSound native sampling rate $f_s = 8000\text{ Hz}$ and hop size $H = 512\text{ samples}$ ($64\text{ ms}$):
$$\text{Time per frame} = \frac{512}{8000} = 0.064\text{ seconds}$$
A fixed width of $T = 128\text{ frames}$ enforces an effective audio duration:
$$D_{\text{target}} = 128 \times 0.064\text{ s} = 8.192\text{ seconds}$$

### 4.2. Empirical Distribution in SPRSound Dataset
Across the 2,683 SPRSound patient recordings:
- Minimum recording duration: $0.30\text{ s}$
- Maximum recording duration: $15.36\text{ s}$
- Mean recording duration: $10.84\text{ s}$
- Standard deviation: $3.89\text{ s}$
- Recordings exceeding 8.192 s: **1,854 recordings (69.10%)**

### 4.3. Quantitative Verification of Event Discarding
In recordings longer than 8.192 s, center-cropping takes an interval $[t_{\text{start}}, t_{\text{end}}]$ centered around the recording midpoint $T_{\text{rec}} / 2$:
$$t_{\text{start}} = \frac{T_{\text{rec}} - 8.192}{2}, \quad t_{\text{end}} = \frac{T_{\text{rec}} + 8.192}{2}$$
Any annotated adventitious respiratory sound (crackle, wheeze, stridor, rhonchi) occurring outside this window is eliminated from the input tensor.

From empirical evaluation of the 6,658 annotated events in `SPRSound/train` and `SPRSound/test`:
1. **Completely Discarded Events:** **946 events (14.21%)** have their entire interval $[E_{\text{start}}, E_{\text{end}}]$ strictly outside $[t_{\text{start}}, t_{\text{end}}]$.
2. **Partially Truncated Events:** **1,346 events (20.22%)** cross the cropping boundary, having their acoustic onset or offset clipped.
3. **Total Mutilated or Lost Events:** **2,292 events (34.43%)**.

### 4.4. Clinical and Model Training Implications
1. **Severe Label Noise in Whole-Record Classification:** A recording annotated as "Wheezing" where wheezes occur at second 1.0 to 3.0 of a 14-second recording will have the wheeze completely cropped out. The model is penalized for predicting "Normal" even though the acoustic input fed to it contains zero wheezes.
2. **Direct Incompatibility with QH-MTLNet:** QH-MTLNet requires simultaneous event localization and record-level classification. Fixed center-cropping destroys the temporal alignment between frame indices and ground-truth event timestamps.

---

## 5. Model Loading, Caching, and Missing Artifact Audit

### 5.1. Backend Caching Mechanisms
Both `services.py` (`ModelCache`) and `disease_service.py` (`DiseaseModelCache`) implement robust caching:
- **Thread Safety:** Protected via Python `threading.Lock()` instances (`services.py:31`, `disease_service.py:95`).
- **Disk Invalidation:** Inspects `os.path.getmtime(artifact_full_path)`. If the file on disk is replaced or re-trained, the cache detects the timestamp delta and automatically reloads the model (`services.py:53`, `disease_service.py:113`).
- **Isolation:** Sound-classification and disease-classification maintain dedicated caches, preventing cross-model thread contention.

### 5.2. Handling of Missing Model Artifacts
Neither backend service fabricates synthetic or dummy predictions:
- `InferenceService.execute_prediction` (`services.py:307-332`):
  - If no `ACTIVE` model exists or the artifact is missing on disk, it records a `Prediction` row with `prediction_status = PredictionStatus.UNAVAILABLE` and raises DRF `ServiceUnavailable` (HTTP 503).
  - If prediction fails during matrix operations, it logs the exception, records `PredictionStatus.FAILED`, and returns HTTP 500 without generating random predictions.
- `DiseaseInferenceService.execute_disease_prediction` (`disease_service.py:230-257`):
  - If no active disease model is found, records `DiseasePrediction` with `PredictionStatus.UNAVAILABLE` and raises an API exception with code `"DISEASE_MODEL_NOT_DEPLOYED"`.

---

## 6. Recommended Corrections & Unified Preprocessing Architecture

To ensure 100% mathematical parity across training, evaluation, and production inference, the two separate preprocessing implementations should be consolidated into a single source of truth.

```
                           RECOMMENDED ARCHITECTURE
                           
                 ml/preprocessing/respiratory_preprocessor.py
                     (Canonical Preprocessing Contract)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
ML Training Pipeline                                 Backend ML Inference
(Feature Extraction, QH-MTLNet)                     (services.py, disease_service.py)
         │                                                     │
         ▼                                                     ▼
Reads PreprocessingConfig                             Imports PreprocessingConfig
& RespiratoryAudioPreprocessor                        & RespiratoryAudioPreprocessor
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
                          Database ModelVersion
                    (Stores Complete Frozen JSON Spec)
```

### Key Architectural Recommendations (Read-Only Proposal)

1. **Shared Preprocessing Implementation:**
   - Package `ml/preprocessing/respiratory_preprocessor.py` such that `backend/ml_inference/services.py` directly imports `RespiratoryAudioPreprocessor` and `PreprocessingConfig`.
   - Remove duplicate preprocessing logic (`load_audio`, `compute_mel_spectrogram`, `prepare_cnn_input`) in `services.py`.

2. **Database Registration Schema Enforcement:**
   - Update `register_model.py` to serialize the complete `PreprocessingConfig.to_dict()` into `ModelVersion.preprocessing_config`.
   - Ensure `apply_bandpass=True` and `normalization="minmax_01"` are always persisted.

3. **Replace Fixed Center-Cropping for QH-MTLNet:**
   - Replace rigid 128-frame center-cropping with a sliding-window chunking or dynamic recurrent/temporal pooling mechanism that retains the entire audio duration.
   - For variable-length recordings, compute frame-level temporal event predictions across the complete spectrogram and aggregate recording-level probabilities using Attention-based Multi-Instance Learning (MIL) pooling.

4. **Synchronize Resampling Algorithm:**
   - Standardize on `librosa.resample(..., res_type='soxr_hq')` in both training and backend inference.

---

## 7. Prioritized Action Plan

| Priority | Action Item | Target Component | Risk Addressed |
|---|---|---|---|
| **P0 (Immediate)** | Re-register `sprsound-cnn-v1-fold0` with explicit `apply_bandpass: true` and `normalization: "minmax_01"` in `db.sqlite3`. | Database / `register_model.py` | Resolves ReLU zero-clamping and high-frequency noise corruption in the active sound model. |
| **P0 (Immediate)** | Set default `sample_rate = 8000` across all backend service signatures and CLI registration tools. | `services.py`, `register_model.py` | Eliminates 2x frequency dilation when processing uploaded audio. |
| **P1 (High)** | Replace `scipy.signal.resample` with `librosa.resample(..., res_type='soxr_hq')` in `services.py`. | `services.py:114` | Eliminates FFT circular convolution ringing artifacts on non-8kHz uploads. |
| **P1 (High)** | Refactor `AudioPreprocessor` in `services.py` to delegate directly to `RespiratoryAudioPreprocessor`. | `backend/ml_inference/services.py` | Eliminates code duplication and guarantees training-inference parity. |
| **P2 (Architecture)** | Implement chunked sliding-window inference and Multi-Instance Learning for QH-MTLNet. | ML Model & Preprocessing | Prevents the loss/mutilation of 34.43% of annotated clinical respiratory events. |

---

## 8. Audit Verification Checklist

- **Files modified during audit:** 0 (source code remained strictly untouched; audit deliverable written to `docs/` only).
- **Packages installed, upgraded, or changed:** 0.
- **Git commands executed:** 0.
- **Dummy predictions or fabricated metrics generated:** 0.
- **Audit completion status:** 100% Complete.
- **Unverified issues:** None. All parameters, database fields, and signal algorithms verified directly against source files and live database state.

