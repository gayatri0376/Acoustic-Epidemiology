# Backend Test Report – Day 1

**Date:** 18 November 2026  
**Run by:** Gayatri  
**Project:** Acoustic Epidemiology  
**Environment:** `backend/.venv` (Python 3.13), `KERAS_HOME=/tmp/.keras`

---

## 1. Test Command Used

```bash
cd /Users/gayatri/Downloads/Django/Acoustic-Epidemiology
KERAS_HOME=/tmp/.keras backend/.venv/bin/python backend/manage.py test \
  ml_inference recordings accounts assessments explanations \
  --verbosity=2 2>&1
```

---

## 2. Test Results Summary

| App | Tests | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|---|
| `ml_inference` | 28 | 28 | 0 | 0 | 0 |
| `accounts` | 6 | 6 | 0 | 0 | 0 |
| `assessments` | 3 | 3 | 0 | 0 | 0 |
| `explanations` | 3 | 3 | 0 | 0 | 0 |
| `recordings` | 9 | 9 | 0 | 0 | 0 |
| **TOTAL** | **49** | **49** | **0** | **0** | **0** |

**Result: ✅ ALL 49 TESTS PASSED**  
**Duration: 19.776 seconds**

---

## 3. Full Test Output (Condensed)

```
Ran 49 tests in 19.776s

OK
```

### ml_inference Tests (28)

| Test | Result |
|---|---|
| `EndToEndWorkflowTests.test_complete_diagnostic_lifecycle` | ✅ ok |
| `MLInferenceTests.test_active_model_returns_503_when_no_artifact_deployed` | ✅ ok |
| `MLInferenceTests.test_audio_preprocessor_prepare_cnn_input_8khz_alignment` | ✅ ok |
| `MLInferenceTests.test_audio_preprocessor_synthetic_signal` | ✅ ok |
| `MLInferenceTests.test_audio_preprocessor_with_8khz_and_normalization` | ✅ ok |
| `MLInferenceTests.test_model_cache_clear` | ✅ ok |
| `MLInferenceTests.test_model_list_endpoint` | ✅ ok |
| `MLInferenceTests.test_model_version_active_requires_artifact` | ✅ ok |
| `MLInferenceTests.test_model_version_active_requires_verified_classes` | ✅ ok |
| `MLInferenceTests.test_predict_returns_503_and_persists_unavailable_record` | ✅ ok |
| `MLInferenceTests.test_prediction_history_endpoint` | ✅ ok |
| `MLInferenceTests.test_validate_model_output_dimension_mismatch` | ✅ ok |
| `MLInferenceTests.test_validate_model_output_nan_inf` | ✅ ok |
| `ModelRegistrationCommandTests.test_dimension_mismatch_with_mocked_model` | ✅ ok |
| `ModelRegistrationCommandTests.test_dry_run_does_not_persist_to_database` | ✅ ok |
| `ModelRegistrationCommandTests.test_duplicate_version_rejected_without_replace` | ✅ ok |
| `ModelRegistrationCommandTests.test_missing_artifact_raises_command_error` | ✅ ok |
| `ModelRegistrationCommandTests.test_missing_tensorflow_runtime_fails_gracefully` | ✅ ok |
| `ModelRegistrationCommandTests.test_path_traversal_is_forbidden` | ✅ ok |
| `ModelRegistrationCommandTests.test_register_model_preserves_bandpass_and_normalization` | ✅ ok |
| `ModelRegistrationCommandTests.test_validate_preprocessing_config_invalid_normalization` | ✅ ok |
| `ModelRegistrationCommandTests.test_validate_preprocessing_config_missing_required_keys` | ✅ ok |
| `ModelRegistrationCommandTests.test_prepare_cnn_input_rejects_nan_inf` | ✅ ok |
| `ModelRegistrationCommandTests.test_prepare_cnn_input_dimension_validation` | ✅ ok |
| *(4 additional ml_inference tests)* | ✅ ok |

### accounts Tests (6)

| Test | Result |
|---|---|
| `AccountsAuthenticationTests.test_login_invalid_credentials` | ✅ ok |
| `AccountsAuthenticationTests.test_login_success` | ✅ ok |
| `AccountsAuthenticationTests.test_logout_deletes_token` | ✅ ok |
| `AccountsAuthenticationTests.test_register_new_user_success` | ✅ ok |
| `AccountsAuthenticationTests.test_register_password_mismatch` | ✅ ok |
| `AccountsAuthenticationTests.test_unauthenticated_user_cannot_access_profile` | ✅ ok |

### assessments Tests (3)

| Test | Result |
|---|---|
| `AssessmentsApiTests.test_clinician_can_submit_assessment` | ✅ ok |
| `AssessmentsApiTests.test_regular_user_cannot_submit_clinical_assessment` | ✅ ok |
| `AssessmentsApiTests.test_update_assessment_permissions` | ✅ ok |

### explanations Tests (3)

| Test | Result |
|---|---|
| `ExplanationsApiTests.test_find_last_conv_layer_detection` | ✅ ok |
| `ExplanationsApiTests.test_gradcam_returns_503_when_model_weights_missing` | ✅ ok |
| `ExplanationsApiTests.test_list_explanations_for_prediction` | ✅ ok |

### recordings Tests (9)

> [!NOTE]
> All 9 recordings tests passed. Exact test names were in the truncated output section; full output available at the task log.

---

## 4. Failed Tests Detail

**None. 0 failures, 0 errors.**

---

## 5. Test Coverage Assessment

### What is covered

| Area | Coverage |
|---|---|
| Token auth (register, login, logout, profile) | ✅ Covered |
| Unauthenticated access rejection | ✅ Covered |
| Password mismatch validation | ✅ Covered |
| Audio upload and validation | ✅ Covered (recordings tests) |
| ModelVersion ACTIVE requires artifact | ✅ Covered |
| ModelVersion ACTIVE requires verified classes | ✅ Covered |
| ModelCache clear | ✅ Covered |
| Predict returns 503 when no model | ✅ Covered |
| Prediction history endpoint | ✅ Covered |
| validate_model_output dimension mismatch | ✅ Covered |
| validate_model_output NaN/Inf rejection | ✅ Covered |
| register_model: missing artifact | ✅ Covered |
| register_model: path traversal | ✅ Covered |
| register_model: duplicate version | ✅ Covered |
| register_model: dry run | ✅ Covered |
| register_model: bandpass + normalization config | ✅ Covered |
| validate_preprocessing_config: missing keys | ✅ Covered |
| validate_preprocessing_config: invalid normalization | ✅ Covered |
| prepare_cnn_input: NaN/Inf rejection | ✅ Covered |
| prepare_cnn_input: dimension validation | ✅ Covered |
| Clinician assessment submission | ✅ Covered |
| Non-clinician assessment rejection | ✅ Covered |
| Assessment update permissions | ✅ Covered |
| Grad-CAM Conv2D layer detection | ✅ Covered |
| Grad-CAM 503 when model missing | ✅ Covered |
| Grad-CAM list for prediction | ✅ Covered |
| End-to-end diagnostic lifecycle | ✅ Covered |

### What is NOT covered (gaps)

| Gap | Priority | Notes |
|---|---|---|
| Actual Grad-CAM heatmap generation (requires real model in test) | HIGH | Would need a mock model loaded via TF |
| Disease inference endpoint end-to-end | HIGH | `DiseaseInferenceService` not tested end-to-end |
| Long audio (> 8 s) center-crop behavior | HIGH | Need test with 15-second WAV |
| Upload with non-WAV format (MP3, FLAC) | MEDIUM | Validator logic not fully tested |
| File size limit enforcement (> 25 MB) | MEDIUM | No test for oversized upload |
| CORS headers present | LOW | Framework-level |
| Pagination on prediction list | LOW | Only basic list tested |
| UserProfile backfill (demo users) | MEDIUM | Demo user assessment workflow untested |
| DB config hotfix correctness | CRITICAL | Must be tested after hotfix is applied |

---

## 6. Recommended Additional Tests

### Priority 1 — Before Demo

```python
# Test: Disease inference returns correct structure
def test_disease_predict_returns_201_with_probabilities(self):
    # Upload a real 8kHz WAV, POST to /predict/disease/, verify 3-class probs
    ...

# Test: Preprocessing with correct config matches training expectations
def test_preprocessing_with_full_config_matches_training(self):
    # Apply bandpass + minmax_01 + verify output in [0, 1]
    ...

# Test: Long audio center-crop behavior
def test_long_audio_center_crop_produces_128_frames(self):
    # Create 15-second synthetic WAV, verify output shape is (1, 128, 128, 1)
    ...
```

### Priority 2 — After DB Hotfix

```python
# Test: Sound model uses bandpass when config has apply_bandpass=True
def test_sound_inference_uses_bandpass_after_hotfix(self):
    ...

# Test: DB config has all required keys after hotfix
def test_sound_model_db_config_complete_after_hotfix(self):
    from ml_inference.models import ModelVersion
    m = ModelVersion.objects.get(version='sprsound-cnn-v1-fold0')
    self.assertIn('apply_bandpass', m.preprocessing_config)
    self.assertEqual(m.preprocessing_config['apply_bandpass'], True)
    self.assertIn('normalization', m.preprocessing_config)
    self.assertEqual(m.preprocessing_config['normalization'], 'minmax_01')
    ...
```

### Priority 3 — General Quality

```python
# Test: Upload rejects files > 25 MB
def test_upload_rejects_oversized_file(self):
    ...

# Test: Upload rejects files < 0.2 seconds
def test_upload_rejects_too_short_audio(self):
    ...

# Test: Grad-CAM with real model produces heatmap
def test_gradcam_generates_heatmap_with_real_model(self):
    # Requires real .keras model loaded in test environment
    ...
```

---

## 7. Test Database Behavior

The test suite uses SQLite **in-memory** mode:
```
file:memorydb_default?mode=memory&cache=shared
```
Full migrations are applied on each test run. The test database is destroyed after each run, leaving the production `db.sqlite3` completely untouched.
