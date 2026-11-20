# DAY 1 – Backend Audit Report

**Date:** 18 November 2026  
**Engineer:** Gayatri  
**Project Deadline:** 18 November 2026  
**Repo Root:** `/Users/gayatri/Downloads/Django/Acoustic-Epidemiology`  
**Backend Root:** `backend/`  
**DB Engine:** SQLite (`backend/db.sqlite3`)  
**Python Env:** `backend/.venv` (Python 3.13)  

---

## 1. Executive Summary

| Item | Status |
|---|---|
| All 5 backend apps present and importable | ✅ VERIFIED |
| Authentication (Token-based) | ✅ VERIFIED |
| Audio upload and validation | ✅ VERIFIED |
| CNN model loading and inference | ✅ VERIFIED |
| Grad-CAM explainability | ✅ VERIFIED |
| Clinical assessments | ✅ VERIFIED |
| Test suite (49 tests) | ✅ ALL PASSED |
| Active model `sprsound-cnn-v1-fold0` DB config | ⚠️ MISSING 3 KEYS |
| Active model `sprsound-disease-cnn-v2` DB config | ⚠️ MISSING 4 KEYS |
| CNN evaluation concern (all predictions → Healthy Control) | 🔴 OPEN ISSUE |

---

## 2. Project Structure

```
backend/
├── config/                    # Django project config (settings, main urls)
│   ├── settings.py
│   ├── urls.py
│   └── views.py               # HealthCheck, Readiness, ApiIndex views
├── accounts/                  # Auth and RBAC
├── recordings/                # Audio upload and storage
├── ml_inference/              # CNN inference, model registry, predictions
├── explanations/              # Grad-CAM explainability
├── assessments/               # Clinician review workflow
├── manage.py
└── db.sqlite3

ml/
└── models/saved_models/       # Model artifacts (outside backend/)
    ├── sprsound-cnn-v1-fold0.keras        (5.17 MB)
    ├── sprsound-cnn-v1-fold0_config.json
    ├── sprsound-disease-cnn-v2.keras      (1.30 MB)
    ├── sprsound-disease-cnn-v2_config.json
    └── sprsound-disease-cnn-v1.keras      (earlier version, not active)
```

> [!NOTE]
> The Django project package is named `config` (not `acoustic_epidemiology`). The `ROOT_URLCONF` is `config.urls`.

---

## 3. Authentication & Authorization (`accounts` app)

### User Model
- Extends Django's built-in `User` via `UserProfile` (OneToOne, UUID PK)
- **Roles:** `CLINICIAN`, `RESEARCHER`, `UPLOADER`, `ADMINISTRATOR`
- Default role for new registrations: `UPLOADER`
- Superusers auto-assigned `ADMINISTRATOR` via `post_save` signal

### Role Properties
| Property | Roles that qualify |
|---|---|
| `is_clinician` | CLINICIAN, ADMINISTRATOR |
| `is_researcher` | RESEARCHER, ADMINISTRATOR |
| `is_administrator` | ADMINISTRATOR, is_staff, is_superuser |

### Auth Mechanism
- **DRF Token Authentication** (`rest_framework.authtoken`)
- Token returned on register and login
- Session authentication also configured

### API Endpoints (accounts)
| Method | URL | Permission | Notes |
|---|---|---|---|
| POST | `/api/v1/auth/register/` | AllowAny | Returns token on success |
| POST | `/api/v1/auth/login/` | AllowAny | Accepts username or email |
| POST | `/api/v1/auth/logout/` | IsAuthenticated | Deletes token |
| GET/PATCH | `/api/v1/auth/me/` | IsAuthenticated | PATCH: org/dept only |

### Permission Classes
| Class | Grants Access To |
|---|---|
| `IsClinician` | CLINICIAN, ADMINISTRATOR, staff/superuser |
| `IsResearcher` | RESEARCHER, CLINICIAN, ADMINISTRATOR, staff/superuser |
| `IsUploader` | Any authenticated user |
| `IsOwnerOrClinician` | Object owner, CLINICIAN, RESEARCHER, ADMINISTRATOR |
| `IsOwnerOrReadOnly` | Read: any allowed; write: owner or superuser |

### Role Mapping to Frontend
`UserSerializer.get_role()` maps DB roles → frontend role strings:
- CLINICIAN / RESEARCHER → `"doctor"`
- ADMINISTRATOR / superuser → `"admin"`
- Everything else → `"patient"`

> [!NOTE]
> The 4 demo users (`patient`, `doctor`, `admin`, `demo`) have no linked `UserProfile` records. Their role is returned as `"admin"` (is_staff) or `"patient"` (fallback). This is not a bug — the signal only fires on new user creation; manually-created users need a profile backfill.

---

## 4. Audio Upload & Validation (`recordings` app)

### Recording Model Fields
| Field | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `audio_file` | FileField | Stored at `recordings/<uuid>.<ext>` |
| `original_filename` | CharField(255) | Sanitized via `sanitize_filename()` |
| `file_format` | CharField(16) | Detected from file content |
| `file_size_bytes` | PositiveBigIntegerField | |
| `duration_seconds` | FloatField (nullable) | |
| `sample_rate` | PositiveIntegerField (nullable) | Native rate of uploaded file |
| `channels` | PositiveSmallIntegerField | Default 1 |
| `recording_status` | CharField | UPLOADED/PROCESSING/VALIDATED/FAILED |
| `anatomical_location` | CharField | 8 choices |
| `patient_age_group` | CharField | 9 choices |
| `patient_pseudo_id` | CharField | Anonymized identifier |
| `uploader` | FK → User | SET_NULL on delete |

### Upload Validation Pipeline (`validators.py`)
1. **Size check:** Max 25 MB (`MAX_UPLOAD_SIZE = 25 * 1024 * 1024`)
2. **Extension check:** `.wav`, `.flac`, `.ogg`, `.mp3`
3. **Magic byte verification:** Checks actual file header (RIFF/WAVE, fLaC, OggS, ID3)
4. **Audio metadata extraction:** Uses `soundfile` with `wave` fallback for WAV
5. **Duration check:** Rejects < 0.2 seconds

### Upload Endpoint
| Method | URL | Permission |
|---|---|---|
| GET | `/api/v1/recordings/` | IsAuthenticated (scope: own or all for clinician) |
| POST | `/api/v1/recordings/` | IsAuthenticated |
| GET | `/api/v1/recordings/<uuid>/` | IsAuthenticated + IsOwnerOrClinician |
| DELETE | `/api/v1/recordings/<uuid>/` | IsAuthenticated (owner or superuser only) |

### Storage
- Files saved to `backend/media/recordings/<uuid>.<ext>`
- File deleted from disk when DB record is deleted (via `post_delete` signal)
- No mono conversion at upload time — raw file is stored; mono conversion happens in preprocessing before inference

> [!WARNING]
> **No maximum duration validation.** Long recordings (> 15 s) will cause center-cropping to discard 34% of annotated events during inference. See CNN Integration Readiness report for sliding-window recommendation.

---

## 5. ML Inference & Model Loading (`ml_inference` app)

### ModelVersion Model Fields
| Field | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `version` | CharField(64) | Unique |
| `architecture` | CharField(128) | Default "2D-CNN-Spectrogram" |
| `artifact_path` | CharField(512) | Relative to project root |
| `verified_classes` | JSONField | Ordered class list |
| `preprocessing_config` | JSONField | All preprocessing params |
| `task_type` | CharField | SOUND_CLASSIFICATION / DISEASE_CLASSIFICATION |
| `status` | CharField | DEVELOPMENT / ACTIVE / RETIRED |
| `evaluation_metadata` | JSONField | Metrics (may be empty) |

### Model Loading (ModelCache)
- **Thread-safe** singleton using `threading.Lock()`
- **Disk mtime-based cache invalidation** — reloads if file changes
- **Loads via** `tf.keras.models.load_model(path, compile=False)`
- Returns `None` if artifact file missing; clears cache

### Prediction Model
- `Prediction`: Sound model predictions (4-class)
- `DiseasePrediction`: Disease model predictions (3-class)
- Both store `predicted_class`, `probability_scores` (JSON), `inference_duration_ms`
- Status: `PENDING / SUCCESS / FAILED / UNAVAILABLE`

### Inference Pipeline (services.py)
1. `InferenceService.get_active_model()` — queries `ModelVersion(task_type=SOUND_CLASSIFICATION, status=ACTIVE)`
2. If no active model → creates `UNAVAILABLE` prediction, raises HTTP 503
3. `AudioPreprocessor.load_audio()` — soundfile read, mono conversion, scipy resample
4. `AudioPreprocessor.compute_mel_spectrogram()` — bandpass filter → mel-spec → dB → normalize
5. `AudioPreprocessor.prepare_cnn_input()` — 128-frame center-crop, reshape to `(1, 128, 128, 1)`
6. `model.predict()` → softmax probabilities
7. `InferenceService.validate_model_output()` — NaN/Inf check, dimension vs verified_classes check
8. `Prediction.objects.create()` — persist result

### Inference Endpoints
| Method | URL | Permission |
|---|---|---|
| GET | `/api/v1/models/` | IsAuthenticatedOrReadOnly |
| GET | `/api/v1/models/active/` | IsAuthenticatedOrReadOnly |
| POST | `/api/v1/recordings/<uuid>/predict/` | IsAuthenticated (owner/clinician) |
| GET | `/api/v1/predictions/` | IsAuthenticated (scope by role) |
| GET | `/api/v1/predictions/<uuid>/` | IsAuthenticated |
| POST | `/api/v1/recordings/<uuid>/predict/disease/` | IsAuthenticated |
| GET | `/api/v1/disease-predictions/<uuid>/` | IsAuthenticated |

### Critical DB Config Gap
| Model | Missing Keys | Risk |
|---|---|---|
| `sprsound-cnn-v1-fold0` | `apply_bandpass`, `normalization`, `top_db`, `filter_order` | Defaults to no bandpass, no normalization — degrades accuracy |
| `sprsound-disease-cnn-v2` | `expected_input_shape`, `output_shape`, `output_activation`, `target_conv_layer` | Grad-CAM layer resolution falls back to auto-detect |

---

## 6. Explainability – Grad-CAM (`explanations` app)

### Explanation Model
| Field | Type |
|---|---|
| `id` | UUID |
| `prediction` | FK → Prediction |
| `method` | GRAD_CAM / MEL_SPECTROGRAM_SALIENCY |
| `status` | PENDING / GENERATED / UNSUPPORTED / FAILED |
| `artifact_path` | Relative path to PNG heatmap |
| `structured_data` | JSON (time/freq saliency profiles) |
| `error_details` | Text |

### Grad-CAM Implementation
- Locates last Conv2D layer by scanning `model.layers` in reverse
- Builds sub-model: `Model(inputs, [conv_layer.output, model.output])`
- Computes gradients via `tf.GradientTape`
- Pools gradients and weights conv feature maps
- ReLU + normalizes heatmap to [0, 1]
- Saves PNG via matplotlib to `backend/media/explanations/<prediction_id>_gradcam.png`
- Returns time and frequency saliency profiles as JSON arrays

### Endpoints
| Method | URL | Permission |
|---|---|---|
| GET/POST | `/api/v1/predictions/<uuid>/explanations/` | IsAuthenticated (owner/clinician) |
| GET | `/api/v1/explanations/<uuid>/` | IsAuthenticated (owner/clinician) |

> [!NOTE]
> Grad-CAM uses `find_last_conv_layer()` as fallback layer detection, but the DB config for `sprsound-cnn-v1-fold0` has `target_conv_layer: "last_conv_layer"` stored. The service uses `model.get_layer(last_conv_name)` which will work correctly as long as the layer name matches.

---

## 7. Clinical Assessments (`assessments` app)

### Assessment Model
- Linked to `Prediction` (not Recording directly)
- Fields: `review_status` (IN_REVIEW / CONFIRMED / DISCORDANT / INCONCLUSIVE), `reviewer_conclusion` (6 acoustic findings), `diagnostic_agreement` (bool), `clinical_notes`
- `reviewer` FK → User, SET_NULL on delete

### Endpoints
| Method | URL | Permission |
|---|---|---|
| POST | `/api/v1/predictions/<uuid>/assessments/` | IsAuthenticated + IsClinician |
| GET | `/api/v1/assessments/` | IsAuthenticated (scope by role) |
| GET/PATCH | `/api/v1/assessments/<uuid>/` | IsAuthenticated (PATCH: reviewer only) |

> [!NOTE]
> Assessments are restricted to `IsClinician` for creation. The 4 demo users (`patient`, `doctor`, etc.) have no linked `UserProfile`, so none can currently POST a new assessment unless their profile is backfilled with CLINICIAN role.

---

## 8. URL Routing & Django Settings

### Root URL Config (`config/urls.py`)
| Prefix | App |
|---|---|
| `admin/` | Django admin |
| `api/v1/` | ApiIndexView |
| `api/v1/health/` | HealthCheckView |
| `api/v1/ready/` | ReadinessCheckView |
| `api/v1/auth/` | accounts.urls |
| `api/v1/recordings/` | recordings.urls |
| `api/v1/` | ml_inference.urls |
| `api/v1/` | explanations.urls |
| `api/v1/` | assessments.urls |

### Django Settings Summary
| Setting | Value |
|---|---|
| `ROOT_URLCONF` | `config.urls` |
| `DEBUG` | `True` (env-configurable) |
| `DATABASE` | SQLite at `backend/db.sqlite3` |
| `MEDIA_ROOT` | `backend/media/` |
| `MEDIA_URL` | `media/` |
| `MAX_UPLOAD` | 25 MB (`DATA_UPLOAD_MAX_MEMORY_SIZE`) |
| `CORS_ALLOW_ALL_ORIGINS` | `True` (when DEBUG=True) |
| `DEFAULT_AUTH` | TokenAuthentication, SessionAuthentication |
| `DEFAULT_PERMISSION` | IsAuthenticatedOrReadOnly |
| `PAGE_SIZE` | 20 |

---

## 9. API Endpoint Inventory

| Method | URL | Permission | Status |
|---|---|---|---|
| GET | `/api/v1/` | Public | ✅ VERIFIED |
| GET | `/api/v1/health/` | Public | ✅ VERIFIED |
| GET | `/api/v1/ready/` | Public | ✅ VERIFIED |
| POST | `/api/v1/auth/register/` | Public | ✅ VERIFIED |
| POST | `/api/v1/auth/login/` | Public | ✅ VERIFIED |
| POST | `/api/v1/auth/logout/` | Token | ✅ VERIFIED |
| GET/PATCH | `/api/v1/auth/me/` | Token | ✅ VERIFIED |
| GET | `/api/v1/recordings/` | Token | ✅ VERIFIED |
| POST | `/api/v1/recordings/` | Token | ✅ VERIFIED |
| GET | `/api/v1/recordings/<uuid>/` | Token + Owner/Clinician | ✅ VERIFIED |
| DELETE | `/api/v1/recordings/<uuid>/` | Token + Owner/Admin | ✅ VERIFIED |
| GET | `/api/v1/models/` | Token or Read | ✅ VERIFIED |
| GET | `/api/v1/models/active/` | Token or Read | ✅ VERIFIED |
| POST | `/api/v1/recordings/<uuid>/predict/` | Token | ✅ VERIFIED |
| GET | `/api/v1/predictions/` | Token | ✅ VERIFIED |
| GET | `/api/v1/predictions/<uuid>/` | Token | ✅ VERIFIED |
| POST | `/api/v1/recordings/<uuid>/predict/disease/` | Token | ✅ VERIFIED |
| GET | `/api/v1/disease-predictions/<uuid>/` | Token | ✅ VERIFIED |
| GET/POST | `/api/v1/predictions/<uuid>/explanations/` | Token | ✅ VERIFIED |
| GET | `/api/v1/explanations/<uuid>/` | Token | ✅ VERIFIED |
| POST | `/api/v1/predictions/<uuid>/assessments/` | Token + IsClinician | ✅ VERIFIED |
| GET | `/api/v1/assessments/` | Token | ✅ VERIFIED |
| GET/PATCH | `/api/v1/assessments/<uuid>/` | Token | ✅ VERIFIED |

---

## 10. Database State – Active Models

### Model 1: `sprsound-cnn-v1-fold0`
- **ID:** `8382d8a1-5d4b-4b36-9dec-7dbea4b0cda3`
- **Task:** SOUND_CLASSIFICATION | **Status:** ACTIVE
- **Artifact:** `ml/models/saved_models/sprsound-cnn-v1-fold0.keras` ✅ exists
- **Classes:** `['Normal', 'CAS', 'DAS', 'CAS & DAS']`
- **DB preprocessing_config (current):**
  ```json
  {"sample_rate": 8000, "n_mels": 128, "n_fft": 1024, "hop_length": 512,
   "f_min": 50.0, "f_max": 2000.0, "to_db": true,
   "expected_input_shape": [null,128,128,1], "output_shape": [null,4],
   "output_activation": "softmax", "target_conv_layer": "last_conv_layer"}
  ```
- **Missing from DB:** `apply_bandpass`, `normalization`, `top_db`, `filter_order`
- **Config file has:** `apply_bandpass: true`, `normalization: "minmax_01"`, `top_db: 80.0`

### Model 2: `sprsound-disease-cnn-v2`
- **ID:** `fcb700e1-d55c-4698-8724-1dbd35979dde`
- **Task:** DISEASE_CLASSIFICATION | **Status:** ACTIVE
- **Artifact:** `ml/models/saved_models/sprsound-disease-cnn-v2.keras` ✅ exists
- **Classes:** `['Healthy Control', 'Pneumonia', 'Wheezing Disorders']`
- **DB preprocessing_config (current):**
  ```json
  {"sample_rate": 8000, "n_mels": 128, "n_fft": 1024, "hop_length": 512,
   "f_min": 50.0, "f_max": 2000.0, "apply_bandpass": true,
   "top_db": 80.0, "normalization": "minmax_01"}
  ```
- **Missing from DB:** `expected_input_shape`, `output_shape`, `output_activation`, `target_conv_layer`

---

## 11. Model Artifact Inventory

| File | Size | Status |
|---|---|---|
| `sprsound-cnn-v1-fold0.keras` | ~5.17 MB | ✅ Active, intact |
| `sprsound-cnn-v1-fold0_config.json` | — | ✅ Present |
| `sprsound-cnn-v1-fold0_history.json` | — | ✅ Present |
| `sprsound-disease-cnn-v2.keras` | ~1.30 MB | ✅ Active, intact |
| `sprsound-disease-cnn-v2_config.json` | — | ✅ Present |
| `sprsound-disease-cnn-v2_history.json` | — | ✅ Present |
| `sprsound-disease-cnn-v1.keras` | — | ⚠️ Present but NOT active (earlier version) |

> [!IMPORTANT]
> No new model artifact from Mohini (e.g. `sprsound-qh-mtlnet-v1.keras`) is present. The only active models are the two registered above.

---

## 12. Known Bugs and Issues

### 🔴 CRITICAL

| # | Issue | Location | Impact |
|---|---|---|---|
| C1 | `sprsound-cnn-v1-fold0` DB config missing `apply_bandpass` → defaults `False` at inference | `ml_inference/services.py` defaults | Bandpass filter NOT applied; model receives unfiltered mel-spec not matching training |
| C2 | `sprsound-cnn-v1-fold0` DB config missing `normalization` → defaults `None` at inference | `ml_inference/services.py` defaults | No min-max normalization; raw dB values fed to model — wrong input distribution |
| C3 | All validation predictions → "Healthy Control" (Mohini's evaluation report) | Disease model | Model may be biased or preprocessing mismatch; **do NOT activate as production classifier** |

### ⚠️ HIGH

| # | Issue | Location | Impact |
|---|---|---|---|
| H1 | `sprsound-disease-cnn-v2` missing `target_conv_layer` in DB | DB config | Grad-CAM uses auto-detect fallback; may target wrong layer |
| H2 | No maximum audio duration limit at upload | `recordings/validators.py` | Long recordings → 128-frame center-crop loses up to 34% of annotated events |
| H3 | Resampling uses `scipy.signal.resample` (FFT-based) vs training `librosa.soxr_hq` | `services.py:load_audio()` | Mismatch for non-8 kHz uploads; native 8 kHz unaffected |
| H4 | Demo users (`patient`, `doctor`, `demo`, `admin`) have no UserProfile linked | DB | Cannot create assessments; role returned as fallback string |

### 🟡 MEDIUM

| # | Issue | Location | Impact |
|---|---|---|---|
| M1 | Hardcoded secret key in `settings.py` | `config/settings.py` | Security issue for any non-local deployment |
| M2 | `DEBUG=True` by default → CORS allows all origins | `config/settings.py` | Acceptable for dev; must be changed for production |
| M3 | No rate limiting on inference endpoint | `ml_inference/views.py` | Model could be called repeatedly without throttling |
| M4 | 128-frame fixed center-crop discards temporal context | `services.py:prepare_cnn_input()` | Clinically relevant sound events at audio boundaries are lost |

---

## 13. Files Inspected

| File | Status |
|---|---|
| `backend/accounts/models.py` | ✅ Read |
| `backend/accounts/views.py` | ✅ Read |
| `backend/accounts/serializers.py` | ✅ Read |
| `backend/accounts/urls.py` | ✅ Read |
| `backend/accounts/permissions.py` | ✅ Read |
| `backend/recordings/models.py` | ✅ Read |
| `backend/recordings/views.py` | ✅ Read |
| `backend/recordings/serializers.py` | ✅ Read |
| `backend/recordings/urls.py` | ✅ Read |
| `backend/recordings/validators.py` | ✅ Read |
| `backend/ml_inference/models.py` | ✅ Read |
| `backend/ml_inference/views.py` | ✅ Read |
| `backend/ml_inference/urls.py` | ✅ Read |
| `backend/ml_inference/services.py` | ✅ Read |
| `backend/ml_inference/serializers.py` | ✅ Read |
| `backend/ml_inference/management/commands/register_model.py` | ✅ Read |
| `backend/ml_inference/management/commands/register_disease_model.py` | ✅ Read |
| `backend/explanations/models.py` | ✅ Read |
| `backend/explanations/views.py` | ✅ Read |
| `backend/explanations/services.py` | ✅ Read |
| `backend/explanations/urls.py` | ✅ Read |
| `backend/assessments/models.py` | ✅ Read |
| `backend/assessments/views.py` | ✅ Read |
| `backend/assessments/serializers.py` | ✅ Read |
| `backend/assessments/urls.py` | ✅ Read |
| `backend/config/urls.py` | ✅ Read |
| `backend/config/settings.py` | ✅ Read |
| `ml/models/saved_models/sprsound-cnn-v1-fold0_config.json` | ✅ Read |
| `ml/models/saved_models/sprsound-disease-cnn-v2_config.json` | ✅ Read |

---

## 14. Files Modified

**None.** This is a read-only audit. No source files were modified.

---

## 15. Project Status Summary

| Component | Status | Notes |
|---|---|---|
| Backend structure | ✅ VERIFIED | All 5 apps complete |
| Auth endpoints | ✅ VERIFIED | Token-based, working |
| Upload validation | ✅ VERIFIED | Magic byte check, 25 MB limit |
| ML inference pipeline | ✅ VERIFIED | Thread-safe cache, error handling |
| Grad-CAM | ✅ VERIFIED | Real gradient computation, no faking |
| Test suite | ✅ 49/49 PASSED | 0 failures |
| DB hotfix for sound model | ⚠️ REQUIRES_APPROVAL | Add `apply_bandpass`, `normalization`, `top_db`, `filter_order` |
| DB hotfix for disease model | ⚠️ REQUIRES_APPROVAL | Add `target_conv_layer`, `expected_input_shape`, etc. |
| CNN evaluation bias | 🔴 OPEN | Do NOT activate disease model as production classifier |
| Long audio handling | 🔴 PENDING | Sliding window not yet implemented |
| Demo user profiles | ⚠️ PENDING | Need UserProfile backfill for assessment workflow |
