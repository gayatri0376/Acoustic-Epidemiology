# Acoustic Epidemiology: Database Schema & Entity Relationships

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Date:** 2026-10-21  
**Database Engine:** MySQL 8.0+ (`utf8mb4_unicode_ci`) / Django ORM  
**Migrations Status:** 100% Applied (0 unapplied migrations)

---

## 1. Entity-Relationship Overview

```
auth_user (Django Built-in)
  ├── 1:1 ── user_profiles (accounts)
  ├── 1:N ── recordings (recordings) [as uploader]
  └── 1:N ── clinical_assessments (assessments) [as reviewer]

recordings (recordings)
  └── 1:N ── predictions (ml_inference)

model_versions (ml_inference)
  └── 1:N ── predictions (ml_inference)

predictions (ml_inference)
  ├── 1:N ── explanations (explanations)
  └── 1:N ── clinical_assessments (assessments)
```

---

## 2. Table Specifications

### 2.1 `user_profiles` (`accounts` app)
Extends Django `auth_user` with clinical roles and hospital metadata.

| Column | Type | Constraints / Defaults | Description |
|---|---|---|---|
| `id` | BigAutoField | Primary Key | Auto-incrementing identifier |
| `user_id` | ForeignKey(`auth_user`) | OneToOne, Unique, CASCADE | Linked Django user record |
| `role` | VARCHAR(32) | Default: `'CLINICIAN'` | Role enum: `CLINICIAN`, `RESEARCHER`, `AUDITOR`, `PATIENT_PARENT` |
| `institution` | VARCHAR(255) | Blank | Hospital or research institute |
| `department` | VARCHAR(128) | Blank | Clinical department (e.g., Pediatric Pulmonology) |
| `license_number`| VARCHAR(64) | Blank | Medical registration number |
| `created_at` | DATETIME(6) | Auto-add | Profile creation timestamp |
| `updated_at` | DATETIME(6) | Auto-update | Profile last modified timestamp |

---

### 2.2 `recordings` (`recordings` app)
Stores uploaded pediatric auscultation audio and signal validation metadata.

| Column | Type | Constraints / Defaults | Description |
|---|---|---|---|
| `id` | CHAR(32) / UUID | Primary Key, `uuid4` | Unique recording identifier |
| `uploader_id` | ForeignKey(`auth_user`) | CASCADE, Indexed | User who uploaded the audio |
| `audio_file` | VARCHAR(100) | FileField (`recordings/%Y/%m/%d/`) | Path to stored `.wav` audio |
| `original_filename`| VARCHAR(255) | Not Null | Original filename |
| `file_format` | VARCHAR(16) | Default: `'wav'` | Validated container format |
| `file_size_bytes` | BIGINT | Not Null | Physical file size in bytes |
| `duration_seconds`| DECIMAL(8, 3) | Nullable | Audio length in seconds |
| `sample_rate` | INT | Nullable | Native sample rate (e.g. 8000 Hz) |
| `channels` | SMALLINT | Default: 1 | Audio channels (1=Mono, 2=Stereo) |
| `checksum_sha256` | VARCHAR(64) | Blank, Indexed | SHA-256 hash for duplicate detection |
| `recording_status`| VARCHAR(32) | Default: `'UPLOADED'`, Indexed | Status: `UPLOADED`, `VALIDATING`, `VALIDATED`, `REJECTED`, `PROCESSING_FAILED` |
| `rejection_reason`| TEXT | Blank | Validation failure details if rejected |
| `created_at` | DATETIME(6) | Auto-add, Indexed | Upload timestamp |
| `updated_at` | DATETIME(6) | Auto-update | Last update timestamp |

---

### 2.3 `model_versions` (`ml_inference` app)
Maintains audit trail and lifecycle states of registered CNN models.

| Column | Type | Constraints / Defaults | Description |
|---|---|---|---|
| `id` | CHAR(32) / UUID | Primary Key, `uuid4` | Unique model identifier |
| `version` | VARCHAR(64) | Unique, Not Null | Semantic version (e.g. `'sprsound-cnn-v1-fold0'`) |
| `architecture` | VARCHAR(128) | Default: `'2D-CNN-Spectrogram'` | Model architecture description |
| `status` | VARCHAR(32) | Default: `'DEVELOPMENT'`, Indexed | Status: `DEVELOPMENT`, `STAGING`, `ACTIVE`, `DEPRECATED`, `ARCHIVED` |
| `artifact_path` | VARCHAR(512) | Blank | Relative path to `.keras` or SavedModel weights |
| `target_conv_layer`| VARCHAR(128) | Blank | Convolutional layer name for Grad-CAM (e.g. `'last_conv_layer'`) |
| `output_activation`| VARCHAR(32) | Default: `'softmax'` | Activation: `softmax` or `logits` |
| `verified_classes` | JSON | Default: `list` | Ordered class labels: `['Normal', 'CAS', 'DAS', 'CAS & DAS']` |
| `preprocessing_config`| JSON | Default: `dict` | Preprocessing parameters (sample rate, n_mels, top_db, etc.) |
| `evaluation_metrics`| JSON | Default: `dict` | Independent benchmark metrics |
| `training_provenance`| JSON | Default: `dict` | Training dataset, fold index, and provenance notes |
| `created_at` | DATETIME(6) | Auto-add | Registration timestamp |
| `updated_at` | DATETIME(6) | Auto-update | Last modified timestamp |

---

### 2.4 `predictions` (`ml_inference` app)
Records all inference runs and model classification outputs.

| Column | Type | Constraints / Defaults | Description |
|---|---|---|---|
| `id` | CHAR(32) / UUID | Primary Key, `uuid4` | Unique prediction identifier |
| `recording_id` | ForeignKey(`recordings`) | CASCADE, Indexed | Audio recording evaluated |
| `model_version_id`| ForeignKey(`model_versions`)| SET_NULL, Nullable, Indexed | Model version utilized |
| `predicted_class` | VARCHAR(128) | Blank | Predicted class name |
| `probability_scores`| JSON | Default: `dict` | Normalized probability map `{class: prob}` |
| `prediction_status`| VARCHAR(32) | Default: `'PENDING'`, Indexed | Status: `PENDING`, `SUCCESS`, `FAILED`, `UNAVAILABLE` |
| `inference_duration_ms`| INT | Nullable | Forward pass duration in milliseconds |
| `error_details` | TEXT | Blank | Error message if failed or unavailable |
| `created_at` | DATETIME(6) | Auto-add, Indexed | Execution timestamp |

---

### 2.5 `explanations` (`explanations` app)
Links Grad-CAM saliency artifacts and structured interpretability profiles to predictions.

| Column | Type | Constraints / Defaults | Description |
|---|---|---|---|
| `id` | CHAR(32) / UUID | Primary Key, `uuid4` | Unique explanation identifier |
| `prediction_id` | ForeignKey(`predictions`) | CASCADE, Indexed | Prediction being explained |
| `method` | VARCHAR(64) | Default: `'GRAD_CAM'` | Method: `GRAD_CAM`, `MEL_SPECTROGRAM_SALIENCY` |
| `status` | VARCHAR(32) | Default: `'PENDING'`, Indexed | Status: `PENDING`, `GENERATED`, `UNSUPPORTED`, `FAILED` |
| `artifact_path` | VARCHAR(512) | Blank | Path to saved heatmap PNG |
| `structured_data` | JSON | Default: `dict` | Time-frequency saliency profiles and clinical notice |
| `error_details` | TEXT | Blank | Diagnostic error if unsupported or failed |
| `created_at` | DATETIME(6) | Auto-add, Indexed | Creation timestamp |

---

### 2.6 `clinical_assessments` (`assessments` app)
Captures clinician ground-truth validation, concurrence, and diagnostic review.

| Column | Type | Constraints / Defaults | Description |
|---|---|---|---|
| `id` | CHAR(32) / UUID | Primary Key, `uuid4` | Unique assessment identifier |
| `prediction_id` | ForeignKey(`predictions`) | CASCADE, Indexed | Prediction evaluated by clinician |
| `reviewer_id` | ForeignKey(`auth_user`) | CASCADE, Indexed | Pulmonologist / Clinician reviewer |
| `reviewer_conclusion`| VARCHAR(64) | Not Null | Conclusion: `NORMAL`, `WHEEZING`, `CRACKLES`, `STRIDOR`, `RHONCHI`, `OTHER_ADVENTITIOUS`, `INCONCLUSIVE` |
| `agreement_with_ml`| VARCHAR(32) | Default: `'UNASSESSED'` | Agreement: `CONCURS`, `DISAGREES`, `UNASSESSED` |
| `confidence_rating`| SMALLINT | Default: 3, Min: 1, Max: 5 | Clinician diagnostic confidence (1-5) |
| `clinical_notes` | TEXT | Blank | Free-text clinical notes |
| `created_at` | DATETIME(6) | Auto-add, Indexed | Assessment timestamp |
| `updated_at` | DATETIME(6) | Auto-update | Last modified timestamp |
