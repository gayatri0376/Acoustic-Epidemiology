# Acoustic Epidemiology: RESTful API Reference

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Base URL:** `http://localhost:8000/api/v1/`  
**Authentication Scheme:** `Token <token>` via HTTP `Authorization` Header  
**Date:** 2026-10-21  
**Status:** Verified via Automated Test Suite (47 Django Tests Passing)

---

## 1. System Probes & Health

### 1.1 Root API Index
- **Endpoint:** `GET /api/v1/`
- **Auth:** None (Public)
- **Response:** `HTTP 200 OK`
```json
{
  "system": "Acoustic Epidemiology API",
  "version": "v1",
  "status": "operational",
  "endpoints": { ... }
}
```

### 1.2 Health Probe
- **Endpoint:** `GET /api/v1/health/`
- **Auth:** None (Public)
- **Response:** `HTTP 200 OK`
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2026-10-21T11:45:00Z"
}
```

### 1.3 Readiness Probe
- **Endpoint:** `GET /api/v1/ready/`
- **Auth:** None (Public)
- **Response:** `HTTP 200 OK` (when database and media storage are accessible)

---

## 2. Authentication & User Profiles (`/auth/`)

### 2.1 User Registration
- **Endpoint:** `POST /api/v1/auth/register/`
- **Auth:** None
- **Payload:**
```json
{
  "username": "dr_smith",
  "email": "smith@hospital.org",
  "password": "SecurePassword123!",
  "role": "CLINICIAN",
  "institution": "Pediatric Hospital",
  "department": "Pulmonology"
}
```
- **Response:** `HTTP 201 CREATED` with issued auth token.

### 2.2 User Login
- **Endpoint:** `POST /api/v1/auth/login/`
- **Payload:** `{"username": "...", "password": "..."}`
- **Response:** `HTTP 200 OK` with `token` and user profile object.

### 2.3 User Profile
- **Endpoint:** `GET /api/v1/auth/profile/` | `PUT /api/v1/auth/profile/`
- **Auth:** Authenticated

---

## 3. Audio Recordings (`/recordings/`)

### 3.1 Upload Recording
- **Endpoint:** `POST /api/v1/recordings/`
- **Auth:** Authenticated
- **Content-Type:** `multipart/form-data`
- **Payload:** `audio_file` (RIFF WAV format, $\le 25$ MB, 0.2s to 120s duration)
- **Response:** `HTTP 201 CREATED`
```json
{
  "id": "7b0a70f3-8b7c-4822-a720-3ce7c0cb9154",
  "original_filename": "pediatric_lung_sound.wav",
  "file_format": "wav",
  "sample_rate": 8000,
  "channels": 1,
  "duration_seconds": 9.216,
  "recording_status": "VALIDATED",
  "created_at": "2026-10-21T11:45:10Z"
}
```

### 3.2 List Recordings
- **Endpoint:** `GET /api/v1/recordings/`
- **Auth:** Authenticated (returns user recordings; clinicians/researchers see accessible records)

### 3.3 Recording Detail & Delete
- **Endpoint:** `GET /api/v1/recordings/<uuid:id>/` | `DELETE /api/v1/recordings/<uuid:id>/`

---

## 4. Machine Learning Inference (`/models/`, `/recordings/<id>/predict/`)

### 4.1 List Registered Models
- **Endpoint:** `GET /api/v1/models/`
- **Auth:** Authenticated
- **Response:** `HTTP 200 OK` (list of registered ModelVersions)

### 4.2 Get Active Model
- **Endpoint:** `GET /api/v1/models/active/`
- **Auth:** Authenticated
- **Behavior:**
  - If a model is active and its artifact exists on disk: `HTTP 200 OK`
  - If no model is active or deployed: `HTTP 503 SERVICE UNAVAILABLE` (`code: NO_ACTIVE_MODEL`)

### 4.3 Execute Prediction
- **Endpoint:** `POST /api/v1/recordings/<uuid:recording_id>/predict/`
- **Auth:** Authenticated
- **Behavior:**
  - **When genuine model is active:** Runs 2D Log Mel-Spectrogram extraction and forward inference pass:
    - `HTTP 201 CREATED`
    ```json
    {
      "id": "e6f809a1-b841-4770-96f7-b2f5619e0cf6",
      "recording": "7b0a70f3-8b7c-4822-a720-3ce7c0cb9154",
      "model_version": "sprsound-cnn-v1-fold0",
      "predicted_class": "DAS",
      "probability_scores": {
        "Normal": 0.0812,
        "CAS": 0.0431,
        "DAS": 0.8124,
        "CAS & DAS": 0.0633
      },
      "prediction_status": "SUCCESS",
      "inference_duration_ms": 42
    }
    ```
  - **When model is not deployed:** Enforces transparent failure without fabricating predictions:
    - `HTTP 503 SERVICE UNAVAILABLE`
    ```json
    {
      "error": "ML Inference Service Unavailable",
      "code": "MODEL_NOT_DEPLOYED",
      "prediction_id": "...",
      "message": "No trained convolutional neural network model artifact is currently available in the model registry."
    }
    ```
    Persists audit record with `prediction_status='UNAVAILABLE'`.

### 4.4 Prediction History & Detail
- **Endpoint:** `GET /api/v1/recordings/<uuid:id>/predictions/`
- **Endpoint:** `GET /api/v1/predictions/<uuid:id>/`

---

## 5. Explainability & Grad-CAM (`/predictions/<id>/explanations/`)

### 5.1 Request Grad-CAM Saliency Map
- **Endpoint:** `POST /api/v1/predictions/<uuid:prediction_id>/explanations/`
- **Auth:** Authenticated (Clinician, Researcher, or recording owner)
- **Payload:** `{"method": "GRAD_CAM"}`
- **Behavior:**
  - **When trained CNN weights are available:** Computes authentic gradient backpropagation through `last_conv_layer`, renders heatmap overlay to `media/explanations/`, and computes frequency/time saliency profiles:
    - `HTTP 202 ACCEPTED`
  - **When model artifact is missing or unsupported:** Refuses to fabricate heatmaps:
    - `HTTP 503 SERVICE UNAVAILABLE` (`code: GRADIENT_MODEL_UNAVAILABLE` or `CONV_LAYER_NOT_FOUND`)
    - Persists audit record with `status='UNSUPPORTED'`.

### 5.2 List & Retrieve Explanations
- **Endpoint:** `GET /api/v1/predictions/<uuid:id>/explanations/`
- **Endpoint:** `GET /api/v1/explanations/<uuid:id>/`

---

## 6. Clinical Assessment (`/assessments/`)

### 6.1 Create Clinician Review
- **Endpoint:** `POST /api/v1/predictions/<uuid:prediction_id>/assessments/`
- **Auth:** Clinician or Researcher
- **Payload:**
```json
{
  "reviewer_conclusion": "CRACKLES",
  "agreement_with_ml": "CONCURS",
  "confidence_rating": 4,
  "clinical_notes": "Fine inspiratory crackles clearly audible in left posterior auscultation quadrant."
}
```
- **Response:** `HTTP 201 CREATED`

### 6.2 Retrieve & Update Assessment
- **Endpoint:** `GET /api/v1/assessments/<uuid:id>/`
- **Endpoint:** `PATCH /api/v1/assessments/<uuid:id>/`
