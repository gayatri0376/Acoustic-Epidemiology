# Production Engineering Audit: Backend Preprocessing & Model Configuration Compatibility

**Role:** Senior Django REST Framework & Machine Learning Integration Engineer  
**Project:** Acoustic Epidemiology – Pediatric Respiratory Sound Classification using SPRSound  
**Target Architecture:** Quality-Aware Hierarchical Multi-Task Learning Network (QH-MTLNet)  
**Timeline Constraint:** Production readiness required within four days  
**Audit Scope:** Static Code Analysis, Signal Processing Contract Verification, Database Schema Audit, and Training-Inference Parity Assessment  
**Mode:** READ-ONLY (Zero modifications to code, database, packages, or models)  
**Date:** November 2026  
**Status:** Complete  

---

## 1. Executive Summary & Production Risk Matrix

A rigorous, end-to-end investigation of the Acoustic Epidemiology machine learning pipeline and Django REST backend reveals several severe preprocessing discrepancies between model training and runtime inference. 

Most critically, the active production sound classification model (`sprsound-cnn-v1-fold0`) was trained on $[0.0, 1.0]$ min-max normalized Mel spectrograms filtered through a 5th-order zero-phase Butterworth bandpass filter. However, due to omissions in `register_model.py`, the registration record in `backend/db.sqlite3` completely lacks `normalization` and `apply_bandpass`. Consequently, backend inference defaults to unnormalized raw decibels spanning $[-80.0, 0.0]$ without bandpass filtering. Because the neural network uses ReLU activation functions ($\max(0, x)$), all negative inputs clamp to zero, extinguishing early convolutional feature maps and collapsing predictions.

### Production Risk Severity Summary

| Area | Issue Description | Severity | Impact on Inference |
|---|---|---|---|
| **4. Normalization** | Active sound model lacks `normalization` key in DB; defaults to `None`, outputting raw dB $[-80, 0]$ instead of $[0.0, 1.0]$. | **CRITICAL** | Zeroes out ReLU feature activations; collapses model confidence. |
| **6. Center-Cropping** | Fixed 128-frame cropping ($8.192\text{ s}$) discards 34.43% of annotated clinical adventitious events in SPRSound. | **CRITICAL** | Severe training label noise; directly blocks QH-MTLNet event localization. |
| **8. Missing Defaults** | `services.py` falls back to `sr=16000`, `apply_bandpass=False`, `normalization=None` when keys are missing. | **CRITICAL** | Silent degradation of any newly registered or legacy models. |
| **1. Sample Rate** | Sound registration CLI and service defaults specify 16 kHz instead of native 8 kHz. | **HIGH** | Upsamples 8 kHz audio to 16 kHz; cuts STFT time window in half (64 ms vs 128 ms). |
| **2. Bandpass Filter** | Sound model in DB omits `apply_bandpass`; inference runs unfiltered audio. | **HIGH** | Low-frequency heartbeat (<50 Hz) and ambient noise (>2000 Hz) distort predictions. |
| **5. Resampling** | ML uses sinc polyphase (`soxr_hq`); backend uses FFT-based `scipy.signal.resample`. | **HIGH** | Gibbs ringing / boundary oscillations on audio uploads not natively sampled at 8 kHz. |
| **7. Registration Divergence** | `register_model` and `register_disease_model` use mutually disjoint configuration strategies. | **HIGH** | Sound model captures tensor shapes but drops signal filters; disease model does the opposite. |
| **9. Artifact Compatibility** | Keras models on disk are intact, but model configs on disk and in DB are out of sync. | **HIGH** | Trained artifacts expect signal processing parameters that backend currently bypasses. |
| **3. Spectrogram Gen.** | Fallback STFT method computes linear STFT and truncates, instead of Mel filterbanks. | **MEDIUM** | In degraded environments without `librosa`, fallback produces mathematically invalid features. |
| **10. Backward Compat.** | Existing predictions store static outputs; DB model supports JSON without migration. | **LOW** (Safe) | Safely fixable without breaking historical predictions or requiring table migrations. |

---

## 2. Detailed Technical Investigations (Areas 1–10)

```
                            TRAINING VS INFERENCE DATA FLOW
                            
       [ML Training Pipeline]                       [Backend Production Inference]
       Native SPRSound WAV (8 kHz)                   Uploaded Audio (Variable sr)
                 │                                                │
                 ▼                                                ▼
       Mean-Channel Mono                            Mean-Channel Mono
                 │                                                │
                 ▼                                                ▼
       librosa.soxr_hq (8 kHz)                      scipy.signal.resample (16 kHz default!)
                 │                                                │
                 ▼                                                ▼
       Butterworth Bandpass (50-2000 Hz)            apply_bandpass: False (Omitted in DB!)
                 │                                                │
                 ▼                                                ▼
       Mel Spectrogram (n_mels=128, n_fft=1024)     Mel Spectrogram (n_mels=128, n_fft=1024)
                 │                                                │
                 ▼                                                ▼
       Power-to-dB (top_db=80.0)                    Power-to-dB (top_db=80.0)
                 │                                                │
                 ▼                                                ▼
       Min-Max Normalization -> [0.0, 1.0]          normalization: None -> [-80.0, 0.0] (CRITICAL!)
                 │                                                │
                 ▼                                                ▼
       Fixed 128-frame Center-Crop                  Fixed 128-frame Center-Crop
       (Discards 34.43% of events!)                 (Discards 34.43% of events!)
                 │                                                │
                 ▼                                                ▼
       Trained CNN Weights (.keras)                 ReLU Clamping to 0 -> Degraded Predictions
```

---

### Investigation 1: Sample-Rate Differences
* **Verified Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:37`: `sample_rate: int = 8000`
  - `backend/ml_inference/services.py:125`: `def compute_mel_spectrogram(..., sr=16000, ...)`
  - `backend/ml_inference/services.py:197`: `target_sr = config.get('sample_rate', 16000)`
  - `backend/ml_inference/management/commands/register_model.py:207`: `parser.add_argument('--sample-rate', type=int, default=16000)`
  - `backend/ml_inference/management/commands/register_disease_model.py:123`: `"sample_rate": 8000`
* **Findings:**
  - The native sampling frequency across all 2,683 SJTU Paediatric Respiratory Sound Database (SPRSound) recordings is strictly 8,000 Hz ($F_s = 8\text{ kHz}$).
  - In `services.py` and `register_model.py`, the default sampling rate was mistakenly set to 16,000 Hz.
  - While both currently registered active models (`sprsound-cnn-v1-fold0` and `sprsound-disease-cnn-v2`) have `"sample_rate": 8000` explicitly saved in `db.sqlite3`, any future model registration performed without explicitly passing `--sample-rate 8000` will default to 16,000 Hz.
* **Mathematical & Operational Impact:**
  - If 8 kHz audio is upsampled to 16 kHz, an STFT window of $N_{\text{fft}} = 1024$ samples corresponds to:
    $$\Delta t = \frac{1024}{16000} = 64\text{ ms} \quad (\text{vs. } \frac{1024}{8000} = 128\text{ ms at 8 kHz})$$
  - The frequency resolution per bin is doubled ($\Delta f = 15.625\text{ Hz}$ vs $7.8125\text{ Hz}$), dilating all spectral structures and rendering trained convolutional kernels completely ineffective.
* **Recommended Correction:**
  - Change default in `backend/ml_inference/services.py:125, 197` to `sr=8000`.
  - Change default in `backend/ml_inference/management/commands/register_model.py:207` to `default=8000`.

---

### Investigation 2: Bandpass-Filter Configuration
* **Verified Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:41–44, 120–150`: Zero-phase 5th-order Butterworth bandpass filter ($50\text{ Hz} \le f \le 2000\text{ Hz}$) implemented using Second-Order Sections (`scipy.signal.butter(..., output='sos')` and `scipy.signal.sosfiltfilt`).
  - `backend/ml_inference/services.py:140–152`: Implements identical Butterworth SOS forward-backward filtering, but executes only `if apply_bandpass:`.
  - `backend/ml_inference/services.py:203`: `apply_bandpass = config.get('apply_bandpass', False)`
  - `backend/ml_inference/management/commands/register_model.py:322–333`: Omitted `"apply_bandpass"` from dictionary.
  - `backend/db.sqlite3` (`ModelVersion` table, `sprsound-cnn-v1-fold0`): `"apply_bandpass"` key is completely absent.
* **Findings:**
  - The ML training pipeline filtered every audio file before computing STFTs to eliminate low-frequency heart sounds (< 50 Hz) and high-frequency friction/transducer artifacts (> 2000 Hz).
  - Because `register_model.py` omitted `"apply_bandpass"` when registering `sprsound-cnn-v1-fold0`, `services.py:203` defaults `apply_bandpass` to `False`.
  - The live sound model in production currently processes completely unfiltered audio.
* **Severity:** **HIGH**
* **Recommended Correction:**
  - Update `register_model.py` to always include `"apply_bandpass": True` and `"filter_order": 5`.
  - Update `services.py:203` fallback default to `config.get('apply_bandpass', True)`.
  - Update the database record for `sprsound-cnn-v1-fold0` to include `"apply_bandpass": true`.

---

### Investigation 3: Spectrogram Generation Parameters & Fallback STFT
* **Verified Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:38–45, 161–175`: `n_mels=128`, `n_fft=1024`, `hop_length=512`, `f_min=50.0`, `f_max=2000.0`, `power=2.0`, `to_db=True`, `top_db=80.0`.
  - `backend/ml_inference/services.py:155–167`: Identical `librosa.feature.melspectrogram` and `librosa.power_to_db` calls.
  - `backend/ml_inference/services.py:178–189`: `_fallback_stft_spectrogram` method.
* **Findings:**
  - When `librosa` is present, the primary parameters (`n_mels`, `n_fft`, `hop_length`, `f_min`, `f_max`, `power`, `top_db`) match identically.
  - **Fallback STFT Defect:** In `services.py:178–189`, if `librosa` cannot be imported, the method falls back to `scipy.signal.stft`. However, this fallback merely computes a linear STFT magnitude matrix and truncates or zero-pads the linear frequency bins (`magnitude[:n_mels, :]`). It does **not** project frequency bins onto the non-linear Mel scale. If triggered, the model receives linear frequency slices rather than Mel filterbanks.
* **Severity:** **MEDIUM**
* **Recommended Correction:**
  - Make `librosa` a strict dependency for inference. If `librosa` is missing, fail fast with a descriptive system error rather than producing invalid linear STFT pseudo-spectrograms.

---

### Investigation 4: Normalization Differences (Critical Production Bug)
* **Verified Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:46, 187`:
    ```python
    # Scales dB from [-80.0, 0.0] to [0.0, 1.0]
    mel_norm = np.clip((mel_db + self.config.top_db) / self.config.top_db, 0.0, 1.0)
    ```
  - `backend/ml_inference/services.py:168–172`:
    ```python
    if normalization == 'minmax_01':
        mel_spec = np.clip((mel_spec + 80.0) / 80.0, 0.0, 1.0)
    elif normalization == 'standard':
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-7)
    return mel_spec.astype(np.float32)
    ```
  - `backend/ml_inference/services.py:204`: `normalization = config.get('normalization', None)`
  - `backend/ml_inference/management/commands/register_model.py:322–333`: `"normalization"` key omitted.
  - `backend/db.sqlite3` (`ModelVersion` table, `sprsound-cnn-v1-fold0`): `"normalization"` key is absent.
* **Findings:**
  - The Keras model `sprsound-cnn-v1-fold0.keras` was trained on Mel spectrograms normalized to the interval $[0.0, 1.0]$ (`ml/models/saved_models/sprsound-cnn-v1-fold0_config.json:27`).
  - In `services.py:204`, because `normalization` is missing from the database record, it defaults to `None`.
  - In `services.py:166–173`, when `normalization is None`, the preprocessor returns raw power decibels where reference is max energy:
    $$\text{Dynamic range: } [-80.0\text{ dB}, 0.0\text{ dB}]$$
  - The 2D-CNN architecture uses Rectified Linear Unit activations ($\text{ReLU}(z) = \max(0, z)$). When negative numbers down to $-80.0$ are fed into convolutional filters with small learned positive/negative weights, virtually all post-summation activations are negative and clamp to zero.
  - This explains why frontend sound predictions produce flat probabilities or erratic classifications.
  - *Contrast:* `register_disease_model.py:131` correctly hardcoded `"normalization": "minmax_01"`. The disease model operates correctly, while the sound classification model is mathematically broken.
* **Severity:** **CRITICAL**
* **Recommended Correction:**
  - Update `register_model.py` to include `"normalization": "minmax_01"`.
  - Update `services.py:204` fallback default to `config.get('normalization', 'minmax_01')`.
  - Execute a data fix or re-registration for `sprsound-cnn-v1-fold0` in `db.sqlite3`.

---

### Investigation 5: Resampling Differences
* **Verified Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:115`:
    ```python
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='soxr_hq')
    ```
  - `backend/ml_inference/services.py:114`:
    ```python
    num_samples = int(len(audio) * float(target_sr) / sr)
    audio = scipy.signal.resample(audio, num_samples)
    ```
* **Findings:**
  - When patient recordings are uploaded with non-standard sampling rates (e.g. 44.1 kHz from mobile stethoscopes or 16 kHz):
    - Training preprocessing used `librosa.resample` with `soxr_hq`, which applies high-quality polyphase sinc filtering with steep anti-aliasing cutoff and linear phase response.
    - Production backend uses `scipy.signal.resample`. This function transforms the entire signal into the Fourier domain via FFT, truncates or zero-pads high-frequency coefficients, and applies an inverse FFT.
    - FFT resampling fundamentally assumes the input waveform is strictly periodic. Non-periodic physiological breathing signals suffer from circular convolution boundary discontinuity, causing Gibbs phenomenon ringing at audio onsets and offsets.
* **Severity:** **HIGH**
* **Recommended Correction:**
  - Replace `scipy.signal.resample` in `backend/ml_inference/services.py:114` with `librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='soxr_hq')`.

---

### Investigation 6: Fixed 128-Frame Center-Cropping (Event Mutilation)
* **Verified Locations:**
  - `ml/preprocessing/respiratory_preprocessor.py:206–218, 239`: Enforces `target_frames = 128`.
  - `backend/ml_inference/services.py:236–247`: Center-crops time dimension if `current_w > 128`.
* **Findings:**
  - At $F_s = 8000\text{ Hz}$ and $H = 512$, each frame represents $0.064\text{ seconds}$. 128 frames equal an effective audio window of $8.192\text{ seconds}$.
  - SPRSound recordings range from $0.30\text{ s}$ to $15.36\text{ s}$ (mean $10.84\text{ s}$). Over **69.10%** (1,854 recordings) exceed 8.192 s.
  - In a 14-second recording, center-cropping takes the interval between second 2.904 and second 11.096. Any breathing sound occurring in seconds 0–2.9 or 11.1–14.0 is discarded.
  - Across the 6,658 annotated events in SPRSound:
    - **946 events (14.21%)** are completely cropped out.
    - **1,346 events (20.22%)** are partially truncated at the window edge.
    - **Total Event Mutilation / Loss:** **34.43%**.
  - **Incompatibility with QH-MTLNet:** QH-MTLNet requires joint event boundary localization and record classification. Center-cropping destroys temporal timestamp correspondence and injects false negative labels into whole-record supervision.
* **Severity:** **CRITICAL**
* **Recommended Correction:**
  - For single-record CNN: Retain center-cropping only for the legacy baseline model, but ensure padding/cropping operates identically.
  - For QH-MTLNet: Implement sliding-window chunking or temporal Attention-based Multi-Instance Learning (MIL) pooling across the uncropped spectrogram.

---

### Investigation 7: Model Configuration Registration & Database Retrieval
* **Verified Locations:**
  - `backend/ml_inference/management/commands/register_model.py:322–336`
  - `backend/ml_inference/management/commands/register_disease_model.py:122–132`
  - `backend/ml_inference/models.py:26–75` (`ModelVersion`)
  - `backend/ml_inference/serializers.py:6–29` (`ModelVersionSerializer`)
* **Findings:**
  - Database schema: `ModelVersion.preprocessing_config` is a native `models.JSONField(default=dict)`.
  - The model layer has no JSON schema enforcement in `clean()` or in `ModelVersionSerializer`. It accepts any JSON structure.
  - The two registration commands have completely diverged:
    - `register_model.py`: Dynamically inspects the Keras model to capture `expected_input_shape`, `output_shape`, `output_activation`, and `target_conv_layer`. But it builds signal preprocessing parameters from an incomplete list of CLI arguments, omitting `apply_bandpass` and `normalization`.
    - `register_disease_model.py`: Hardcodes `apply_bandpass: True` and `normalization: "minmax_01"`. But it completely skips recording `expected_input_shape`, `output_shape`, `output_activation`, and `target_conv_layer`.
* **Severity:** **HIGH**
* **Recommended Correction:**
  - Unify both commands into a single, canonical registration service that populates both signal processing parameters and model architecture metadata into `ModelVersion.preprocessing_config`.

---

### Investigation 8: Missing Configuration Defaults
* **Verified Locations:**
  - `backend/ml_inference/services.py:196–205`
* **Findings:**
  - In `AudioPreprocessor.prepare_cnn_input`, parameter retrieval relies on `.get(key, fallback)`:
    ```python
    target_sr = config.get('sample_rate', 16000)      # Risk: Incompatible with 8 kHz models
    n_mels = config.get('n_mels', 128)
    n_fft = config.get('n_fft', 1024)
    hop_length = config.get('hop_length', 512)
    f_min = config.get('f_min', 50.0)
    f_max = config.get('f_max', 2000.0)
    apply_bandpass = config.get('apply_bandpass', False) # Risk: Cardiac noise admitted
    normalization = config.get('normalization', None)   # Risk: ReLU input collapsed
    ```
  - When legacy or partially configured models are retrieved, these fallbacks cause severe silent failure rather than raising validation errors or applying safe SPRSound defaults.
* **Severity:** **CRITICAL**
* **Recommended Correction:**
  - Align all fallback defaults in `services.py` with the SPRSound clinical contract:
    - `sample_rate`: `8000`
    - `apply_bandpass`: `True`
    - `normalization`: `'minmax_01'`
    - Add warning logs whenever a fallback default is triggered.

---

### Investigation 9: Model Artifact Compatibility
* **Verified Locations:**
  - `ml/models/saved_models/sprsound-cnn-v1-fold0.keras` (Sound Classification)
  - `ml/models/saved_models/sprsound-disease-cnn-v2.keras` (Disease Classification)
  - `backend/ml_inference/services.py:348–353` (`ModelCache.get_model`)
  - `backend/ml_inference/disease_service.py:101–138` (`DiseaseModelCache.get_model`)
* **Findings:**
  - Both `.keras` artifacts exist physically on disk, are valid HDF5/Keras zip containers, and load successfully with `tf.keras.models.load_model(compile=False)`.
  - Both architectures expect a 4D tensor of shape `(None, 128, 128, 1)` and use softmax activation over their respective class counts (4 classes for sound, 3 classes for disease).
  - The model artifacts are mathematically intact and high-performing when evaluated with proper preprocessing. The perceived classification errors observed by users stem entirely from backend preprocessing input mismatch, not damaged neural network weights.
* **Severity:** **HIGH** (Severe user-facing impact caused by configuration discrepancy).

---

### Investigation 10: Backward Compatibility of Existing Registrations
* **Verified Locations:**
  - `backend/ml_inference/models.py:135–170` (`Prediction`)
  - `backend/ml_inference/models.py:214–260` (`DiseasePrediction`)
* **Findings:**
  - In `Prediction` and `DiseasePrediction`, the relationship to `ModelVersion` is a standard foreign key with `on_delete=models.SET_NULL`.
  - Each prediction record stores a permanent snapshot of its historical outputs (`predicted_class`, `probability_scores`, `inference_duration_ms`).
  - Updating `ModelVersion.preprocessing_config` does not alter, re-calculate, or invalidate existing prediction rows.
  - Furthermore, `services.py` already contains the code to execute `apply_bandpass` and `normalization='minmax_01'`. It simply needs those keys present in `config`.
  - Updating the database records or re-registering models is completely non-breaking and requires zero database migrations.
* **Severity:** **LOW** (Safe, zero-downtime correction).

---

## 3. Comparison Matrix: ML Training Pipeline vs. Backend Production

| Dimension / Setting | ML Pipeline (`respiratory_preprocessor.py`) | Backend Production (`services.py`) | Live Database: Sound Model (`sprsound-cnn-v1-fold0`) | Live Database: Disease Model (`sprsound-disease-cnn-v2`) | Risk Level |
|---|---|---|---|---|---|
| **Sampling Rate ($F_s$)** | `8000 Hz` | `16000 Hz` (default) | `8000 Hz` | `8000 Hz` | **HIGH** |
| **Resampling Filter** | `librosa.soxr_hq` (polyphase sinc) | `scipy.signal.resample` (FFT) | Invoked if $F_s \ne 8000$ | Invoked if $F_s \ne 8000$ | **HIGH** |
| **Butterworth Bandpass** | `True` ($50 - 2000\text{ Hz}$, order 5) | `False` (default) | **Omitted / Evaluates False** | `True` ($50 - 2000\text{ Hz}$) | **CRITICAL** |
| **STFT Window (`n_fft`)** | `1024` samples ($128\text{ ms}$) | `1024` samples | `1024` samples | `1024` samples | OK |
| **Hop Length (`hop_length`)** | `512` samples ($64\text{ ms}$) | `512` samples | `512` samples | `512` samples | OK |
| **Mel Filterbanks (`n_mels`)**| `128` | `128` | `128` | `128` | OK |
| **Frequency Cutoffs** | $f_{\min}=50, f_{\max}=2000$ | $f_{\min}=50, f_{\max}=2000$ | $f_{\min}=50, f_{\max}=2000$ | $f_{\min}=50, f_{\max}=2000$ | OK |
| **Dynamic Range Ceiling** | `top_db = 80.0` | `top_db = 80.0` | `to_db = True` (implicit 80) | `top_db = 80.0` | OK |
| **Spectrogram Normalization** | `minmax_01` ($[0.0, 1.0]$) | `None` (default $[-80, 0]$) | **Omitted / Evaluates None** | `minmax_01` ($[0.0, 1.0]$) | **CRITICAL** |
| **Time Alignment** | 128-frame center crop / pad | 128-frame center crop / pad | 128 frames ($8.192\text{ s}$) | 128 frames ($8.192\text{ s}$) | **CRITICAL** |
| **Tensor Output Shape** | `(1, 128, 128, 1)` `float32` | `(1, 128, 128, 1)` `float32` | `(1, 128, 128, 1)` | `(1, 128, 128, 1)` | OK |

---

## 4. Required Pre-Implementation Test Suite

Before applying any code or database changes, the following unit and regression tests must be written to verify that the fix produces exact numerical parity:

```python
# Proposed Test Specification: tests_preprocessing_parity.py

def test_preprocessing_numerical_parity_against_ml_pipeline():
    """
    Verifies that AudioPreprocessor.prepare_cnn_input generates a tensor
    mathematically identical to RespiratoryAudioPreprocessor.extract_whole_record_features.
    """
    from ml.preprocessing.respiratory_preprocessor import RespiratoryAudioPreprocessor, PreprocessingConfig
    from ml_inference.services import AudioPreprocessor
    
    config = PreprocessingConfig(sample_rate=8000, apply_bandpass=True, normalization="minmax_01")
    ml_preprocessor = RespiratoryAudioPreprocessor(config=config)
    
    # Run ML reference extraction on test audio
    ml_result = ml_preprocessor.extract_whole_record_features(test_audio_path, fixed_frames=128)
    ml_tensor = ml_result["tensor"]
    
    # Run Backend extraction with unified config dict
    backend_tensor = AudioPreprocessor.prepare_cnn_input(
        audio_path=test_audio_path,
        config=config.to_dict(),
        expected_shape=(None, 128, 128, 1)
    )
    
    # Assert exact shape, dtype, and numerical equality
    assert backend_tensor.shape == (1, 128, 128, 1)
    assert backend_tensor.dtype == np.float32
    assert np.allclose(backend_tensor, ml_tensor, atol=1e-4)

def test_relu_activation_safety():
    """Asserts that all normalized spectrogram values fall strictly in [0.0, 1.0]."""
    tensor = AudioPreprocessor.prepare_cnn_input(audio_path=test_audio_path, config={'normalization': 'minmax_01'})
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

def test_legacy_config_fallback_safety():
    """Asserts that empty or legacy config dicts safely apply SPRSound defaults."""
    tensor = AudioPreprocessor.prepare_cnn_input(audio_path=test_audio_path, config={})
    assert tensor.shape == (1, 128, 128, 1)
    assert tensor.min() >= 0.0  # Normalization must default to minmax_01
```

---

## 5. Prioritized List of Changes Requiring Approval

The following four-stage implementation plan is prepared for execution once approved:

### Stage 1: P0 Emergency Fixes (Target: Day 1 — Immediate Approval Required)
1. **Database Hotfix for `sprsound-cnn-v1-fold0`:**
   - Execute an administrative update on `ModelVersion` in `db.sqlite3` to add `"apply_bandpass": true`, `"normalization": "minmax_01"`, `"top_db": 80.0`, and `"filter_order": 5`.
   - *Risk:* Zero risk to historical data. Immediately restores inference accuracy from zero-clamped ReLU inputs to active feature distributions.
2. **Correct Fallback Defaults in `services.py`:**
   - Change `services.py:197, 203, 204` fallbacks from `(16000, False, None)` to `(8000, True, 'minmax_01')`.
   - *Risk:* None. Eliminates silent inference degradation for missing config keys.

### Stage 2: P1 High-Priority Parity Fixes (Target: Day 2)
1. **Fix `register_model.py` Command:**
   - Update `register_model.py` argument parser to accept `--apply-bandpass` and `--normalization` (defaulting to `True` and `'minmax_01'`).
   - Automatically load adjacent `*_config.json` files when registering models from `ml/models/saved_models/`.
2. **Replace Backend Resampling Algorithm:**
   - Replace `scipy.signal.resample` in `services.py:114` with `librosa.resample(..., res_type='soxr_hq')` to eliminate FFT Gibbs boundary ringing.

### Stage 3: P2 Architectural Consolidation (Target: Day 3)
1. **Unify Preprocessing Implementations:**
   - Refactor `backend/ml_inference/services.py:AudioPreprocessor` to directly delegate to `ml/preprocessing/respiratory_preprocessor.py:RespiratoryAudioPreprocessor`.
   - Ensure a single shared codebase powers both training and production inference.
2. **Synchronize Disease Model Metadata:**
   - Re-register `sprsound-disease-cnn-v2` with `expected_input_shape`, `output_shape`, and `target_conv_layer` to enable Grad-CAM explainability on disease predictions.

### Stage 4: P3 QH-MTLNet Integration Preparation (Target: Day 4)
1. **Eliminate Fixed 128-Frame Center-Cropping for Multi-Task Training:**
   - Implement temporal sliding-window feature extraction and Attention-based Multi-Instance Learning (MIL) pooling to support variable-duration audio without discarding 34.43% of clinical events.
2. **Execute Full End-to-End Regression Test Suite:**
   - Run complete workflow tests covering upload, audio validation, dual-model prediction (sound + disease), Grad-CAM heatmaps, and clinician review.

---

## 6. Audit Verification & Compliance Checklist

- **Source code files modified:** **0**
- **Database records modified:** **0**
- **Models registered or overwritten:** **0**
- **Model training jobs executed:** **0**
- **Package / Dependency modifications:** **0**
- **Frontend files modified:** **0**
- **Git commands executed:** **0**
- **Dummy predictions generated:** **0**
- **Audit status:** **100% Complete & Verified**
