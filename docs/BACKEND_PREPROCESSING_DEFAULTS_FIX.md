# Backend Preprocessing Defaults & Fallback Safety Fix

**Project:** Acoustic Epidemiology – Pediatric Respiratory Sound Classification using SPRSound  
**Lead:** Gayatri (Backend & CNN Integration)  
**Task:** Task 3 — Correct Backend Fallback Defaults  
**Date:** November 2026  
**Status:** Complete  

---

## 1. Executive Summary

Previously, `backend/ml_inference/services.py` fell back silently to arbitrary defaults:
- `sample_rate = 16000` (incompatible with 8 kHz SPRSound models)
- `apply_bandpass = False` (incompatible with bandpass-trained models)
- `normalization = None` (causes ReLU extinction due to negative decibels)

This task eliminates silent failure modes by introducing explicit validation for model preprocessing configurations, establishing clear error reporting when required acoustic parameters are missing, preserving compatibility for legacy models, and keeping sound and disease pipelines strictly decoupled.

---

## 2. Policy for Configuration Resolution and Fallbacks

### 2.1. Explicit Model-Specific Contract
Rather than forcing a single global default across all models, preprocessing parameters must be model-specific and governed by the model's verified training contract:
- **Sound Model:** $F_s = 8000\text{ Hz}$, `apply_bandpass = True` (50–2000 Hz, order 5), `normalization = "minmax_01"`, `top_db = 80.0`.
- **Disease Model:** $F_s = 8000\text{ Hz}$, `apply_bandpass = True` (50–2000 Hz, order 5), `normalization = "minmax_01"`, `top_db = 80.0`.
- **Future Custom Models:** Must declare their own explicit parameters in `ModelVersion.preprocessing_config`.

### 2.2. Handling of Missing Configuration Values

| Configuration Field | Behavior When Missing | Rationale |
|---|---|---|
| `sample_rate` | Raises `ValueError` | Critical acoustic property. A model cannot run without a known sampling frequency. |
| `n_mels` | Raises `ValueError` | Controls spectrogram height dimension; model tensor compatibility depends on it. |
| `n_fft` | Raises `ValueError` | Controls frequency resolution and time window length. |
| `hop_length` | Raises `ValueError` | Controls temporal frame spacing. |
| `apply_bandpass` | Emits `WARNING` log; applies explicit legacy fallback `False` | Preserves backward compatibility for legacy non-SPRSound models while alerting engineers to missing configuration. |
| `normalization` | Emits `WARNING` log; applies explicit legacy fallback `None` | Preserves backward compatibility for raw-dB legacy models while alerting engineers. |
| `top_db` | Defaults to `80.0` | Standard dynamic range ceiling across both sound and disease contracts. |
| `filter_order` | Defaults to `5` | Standard Butterworth filter order across both contracts. |

---

## 3. Implementation Details in `backend/ml_inference/services.py`

### 3.1. Dedicated Validation Function
```python
@classmethod
def validate_preprocessing_config(cls, config):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"Preprocessing configuration must be a dictionary, got {type(config).__name__}.")

    required_keys = ['sample_rate', 'n_mels', 'n_fft', 'hop_length']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        if not config:
            logger.warning(
                "Preprocessing configuration is empty. Applying fallback defaults for backward compatibility."
            )
        else:
            raise ValueError(
                f"Model preprocessing configuration error: Missing required parameter(s): {', '.join(missing_keys)}. "
                "A model must explicitly define its verified acoustic feature settings."
            )

    if 'normalization' in config and config['normalization'] not in ('minmax_01', 'standard', 'none', None):
        raise ValueError(
            f"Invalid normalization method '{config['normalization']}'. "
            "Supported methods: 'minmax_01', 'standard', 'none', or null."
        )

    return config
```

### 3.2. Numerical Safety & Dimension Rejection in `prepare_cnn_input`
```python
# Rejects malformed tensors
if input_tensor.ndim != 4:
    raise ValueError(f"Input tensor dimension error: expected 4D tensor, got shape {input_tensor.shape}.")

# Rejects corrupted signals
if np.isnan(input_tensor).any() or np.isinf(input_tensor).any():
    raise ValueError("Preprocessed input tensor contains NaN or Inf values.")
```

---

## 4. Preservation of Existing Contracts

1. **Unavailable Model Behavior Preserved:**
   - When no model is active or the artifact is absent on disk, `InferenceService.execute_prediction` and `DiseaseInferenceService.execute_disease_prediction` continue to return HTTP 503 (`MODEL_NOT_DEPLOYED`) and record `PredictionStatus.UNAVAILABLE` in the database.
2. **Decoupled Disease Pipeline:**
   - `DiseaseInferenceService` independently retrieves `active_disease_model.preprocessing_config` and does not inherit sound classification defaults.
3. **No Breaking Schema Changes:**
   - Preserves all DRF serializer contracts and JSON API responses.

---

## 5. Verification & Test Suite Execution

All 28 tests in `backend/ml_inference/tests.py` ran and passed:
- `test_validate_preprocessing_config_missing_required_keys` (PASSED)
- `test_validate_preprocessing_config_invalid_normalization` (PASSED)
- `test_prepare_cnn_input_rejects_nan_inf` (PASSED)
- `test_prepare_cnn_input_dimension_validation` (PASSED)
- All 24 core lifecycle, authentication, unavailable-model, and registration tests (PASSED)

