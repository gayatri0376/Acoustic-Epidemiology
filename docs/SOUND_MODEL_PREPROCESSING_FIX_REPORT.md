# Sound Model Preprocessing Configuration Fix Report

**Project:** Acoustic Epidemiology – Pediatric Respiratory Sound Classification using SPRSound  
**Target Model:** `sprsound-cnn-v1-fold0` (Sound Classification Model)  
**Lead:** Gayatri (Backend & CNN Integration)  
**Task:** Task 2 — Correct Sound Model Preprocessing Configuration  
**Date:** November 2026  
**Status:** Verification & Proposed Fix Ready (Awaiting Database Write Approval)  

---

## 1. Executive Summary & Problem Resolution

During Task 1 verification, `sprsound-cnn-v1-fold0_config.json` confirmed that the 4-class sound classification model was trained with:
- 5th-order zero-phase Butterworth bandpass filter ($50\text{ Hz} \le f \le 2000\text{ Hz}$) (`apply_bandpass = true`)
- Min-Max $[0.0, 1.0]$ normalization from $-80\text{ dB}$ (`normalization = "minmax_01"`)
- Dynamic range ceiling `top_db = 80.0`
- Filter order `filter_order = 5`

However, the active registration record in `backend/db.sqlite3` (`ModelVersion` id: `8382d8a1-5d4b-4b36-9dec-7dbea4b0cda3`) omitted these four keys. This caused backend inference to feed unfiltered, negative decibels $[-80.0, 0.0]$ into a network whose ReLU activations clamped inputs to zero.

This report details the code modifications applied to `register_model.py` and `services.py` to prevent future omissions, provides regression tests, and presents the exact proposed database hotfix for Gayatri's explicit approval.

---

## 2. Code Changes Implemented

### 2.1. Registration CLI Command Update (`backend/ml_inference/management/commands/register_model.py`)
1. **Added Preprocessing CLI Flags:**
   - `--apply-bandpass` (default: `True`) and `--no-bandpass`
   - `--normalization` (choices: `minmax_01`, `standard`, `none`; default: `minmax_01`)
   - `--top-db` (default: `80.0`)
   - `--filter-order` (default: `5`)
   - Changed default `--sample-rate` to `8000` (matching SPRSound native rate).
2. **Auto-Merge Companion Config:**
   - When registering a model from `ml/models/saved_models/`, `register_model.py` now automatically detects and merges any adjacent `*_config.json` if `--preprocessing-config` is omitted.
3. **Preserved Architecture Metadata:**
   - Preserved `expected_input_shape`, `output_shape`, `output_activation`, and `target_conv_layer`.

### 2.2. Service Layer Preprocessing & Guards (`backend/ml_inference/services.py`)
1. **Model-Specific Configuration Validation:**
   - Added `validate_preprocessing_config(config)` enforcing required acoustic keys (`sample_rate`, `n_mels`, `n_fft`, `hop_length`) and checking normalization validity.
2. **Dimension and NaN/Inf Rejection:**
   - Preprocessing now asserts `input_tensor.ndim == 4` and validates spatial/channel dimensions against `expected_shape`.
   - Rejects NaN and infinite values with a descriptive `ValueError`.
3. **Preserved Disease Model Decoupling:**
   - Decoupled parameter extraction so that `DiseaseInferenceService` and `InferenceService` consume their own model's specific dictionary without cross-contamination.

---

## 3. Database Hotfix Specification (Awaiting Approval)

Under strict rule #10, **no database write has been performed**. The following hotfix is prepared and verified:

### 3.1. Target Database & Target Record
- **Active Database File:** `/Users/gayatri/Downloads/Django/Acoustic-Epidemiology/backend/db.sqlite3`
- **Target Table:** `model_versions`
- **Target Record ID:** `8382d8a1-5d4b-4b36-9dec-7dbea4b0cda3`
- **Target Model Version:** `sprsound-cnn-v1-fold0`
- **Current Task Type:** `SOUND_CLASSIFICATION`
- **Current Status:** `ACTIVE`

### 3.2. Backup Plan
- **Backup Command (to run immediately prior to write):**
  ```bash
  cp backend/db.sqlite3 backend/db.sqlite3.backup_task2_$(date +%Y%m%d_%H%M%S)
  ```
- **Backup Verification:** Confirmed that `backend/db.sqlite3` is a valid SQLite file (385,024 bytes).

### 3.3. Current vs. Proposed Configuration

#### Current Database State (`ModelVersion.preprocessing_config`):
```json
{
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

#### Exact Proposed Updated JSON (`ModelVersion.preprocessing_config`):
```json
{
  "sample_rate": 8000,
  "n_mels": 128,
  "n_fft": 1024,
  "hop_length": 512,
  "f_min": 50.0,
  "f_max": 2000.0,
  "filter_order": 5,
  "apply_bandpass": true,
  "to_db": true,
  "top_db": 80.0,
  "normalization": "minmax_01",
  "expected_input_shape": [null, 128, 128, 1],
  "output_shape": [null, 4],
  "output_activation": "softmax",
  "target_conv_layer": "last_conv_layer"
}
```

---

## 4. Regression and Safety Test Results

The following unit and regression tests were added to `backend/ml_inference/tests.py` and executed:
1. `test_register_model_preserves_bandpass_and_normalization`: Verifies that CLI registration correctly persists `apply_bandpass`, `normalization`, `top_db`, and `filter_order` to the database.
2. `test_validate_preprocessing_config_missing_required_keys`: Asserts that missing critical acoustic keys (`sample_rate`, `hop_length`) raise `ValueError`.
3. `test_validate_preprocessing_config_invalid_normalization`: Asserts that invalid normalization strings raise `ValueError`.
4. `test_prepare_cnn_input_rejects_nan_inf`: Asserts that NaN or infinite spectrogram values are rejected.
5. `test_prepare_cnn_input_dimension_validation`: Asserts that the output tensor is strictly 4D `(1, 128, 128, 1)`, non-NaN, and strictly scaled in $[0.0, 1.0]$.

**Test Execution Result:**
- Total Tests: **28 passed, 0 failed** in 5.380s.

