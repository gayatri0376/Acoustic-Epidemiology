# Acoustic Epidemiology API Contract & Integration Guide

This document specifies the REST API contract for the **Acoustic Epidemiology** backend, designed for consumption by the React (Vite) frontend and clinical integration services.

Base URL (Local Development): `http://127.0.0.1:8000/api/v1/`  
Frontend Environment Key: `VITE_API_BASE_URL=http://127.0.0.1:8000`

---

## 1. Authentication & Headers

Standard endpoints require an Authorization header using DRF Token Authentication:
```http
Authorization: Token <user_auth_token>
```
For audio file uploads, requests must use `multipart/form-data`.  
For JSON endpoints, set `Content-Type: application/json` and `Accept: application/json`.

---

## 2. System Endpoints

### 2.1 Basic Health Check
- **Route:** `GET /api/v1/health/`
- **Auth:** None (Public)
- **Response `200 OK`:**
```json
{
  "status": "healthy",
  "service": "Acoustic Epidemiology API",
  "version": "1.0.0",
  "timestamp": "2026-10-16T03:45:00.000000+00:00"
}
```

### 2.2 System Readiness Probe
- **Route:** `GET /api/v1/ready/`
- **Auth:** None (Public)
- **Response `200 OK` (Operational with Model Pending):**
```json
{
  "ready": true,
  "service": "Acoustic Epidemiology Backend",
  "timestamp": "2026-10-16T03:45:00.000000+00:00",
  "components": {
    "database": {
      "status": "connected",
      "engine": "mysql",
      "reachable": true
    },
    "storage": {
      "status": "writable",
      "available": true
    },
    "ml_inference": {
      "status": "unavailable",
      "available": false,
      "detail": "No verified CNN model weights are currently deployed in ml/models/."
    }
  }
}
```

---

## 3. Authentication Endpoints

### 3.1 Register
- **Route:** `POST /api/v1/auth/register/`
- **Request:**
```json
{
  "username": "dr_patel",
  "email": "patel@hospital.org",
  "password": "SecurePassword123!",
  "password_confirm": "SecurePassword123!",
  "first_name": "Rohan",
  "last_name": "Patel",
  "organization": "Pediatric Pulmonary Center",
  "department": "Pediatric Intensive Care"
}
```
- **Response `201 Created`:**
```json
{
  "message": "User registered successfully.",
  "user": {
    "id": 2,
    "username": "dr_patel",
    "email": "patel@hospital.org",
    "first_name": "Rohan",
    "last_name": "Patel",
    "profile": {
      "id": "c1f7b8e3-0d34-4b51-9311-8ecb71a25d20",
      "role": "UPLOADER",
      "role_display": "Acoustic Data Uploader",
      "organization": "Pediatric Pulmonary Center",
      "department": "Pediatric Intensive Care",
      "created_at": "2026-10-16T03:45:00.000000Z",
      "updated_at": "2026-10-16T03:45:00.000000Z"
    }
  },
  "token": "9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b"
}
```

### 3.2 Login
- **Route:** `POST /api/v1/auth/login/`
- **Request:**
```json
{
  "username": "dr_patel",
  "password": "SecurePassword123!"
}
```
- **Response `200 OK`:**
```json
{
  "message": "Authentication successful.",
  "token": "9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b",
  "user": { ... }
}
```

---

## 4. Acoustic Audio Recordings

### 4.1 Upload Audio Recording
- **Route:** `POST /api/v1/recordings/`
- **Headers:** `Authorization: Token <token>`, `Content-Type: multipart/form-data`
- **Form Fields:**
  - `audio_file`: Audio file (`.wav`, `.flac`, `.ogg`, or `.mp3` under 25 MB)
  - `anatomical_location`: Choice (`TRACHEA`, `ANTERIOR_RIGHT_UPPER`, `ANTERIOR_LEFT_UPPER`, `POSTERIOR_RIGHT_LOWER`, `POSTERIOR_LEFT_LOWER`, `LATERAL_RIGHT`, `LATERAL_LEFT`, `UNSPECIFIED`)
  - `patient_age_group`: Choice (`NEONATE`, `INFANT`, `TODDLER`, `PRESCHOOL`, `SCHOOL_AGE`, `ADOLESCENT`, `UNKNOWN`)
  - `recording_device`: String (e.g. `Littmann 3200 Electronic Stethoscope`)
  - `patient_pseudo_id`: De-identified ID (e.g. `PED-RESP-042`)
- **Response `201 Created`:**
```json
{
  "message": "Audio recording uploaded and validated successfully.",
  "recording": {
    "id": "e2a149b5-685b-48ae-8b3d-d60237583489",
    "original_filename": "respiratory_breath_sound.wav",
    "file_format": "wav",
    "file_size_bytes": 160044,
    "duration_seconds": 5.0,
    "sample_rate": 16000,
    "channels": 1,
    "recording_status": "VALIDATED",
    "anatomical_location": "POSTERIOR_RIGHT_LOWER",
    "anatomical_location_display": "Posterior Right Lower Lobe",
    "patient_age_group": "TODDLER",
    "patient_age_group_display": "Toddler (1-3 years)",
    "recording_device": "Littmann 3200 Electronic Stethoscope",
    "patient_pseudo_id": "PED-RESP-042",
    "uploader": 2,
    "uploader_username": "dr_patel",
    "file_url": "/media/recordings/e2a149b5-685b-48ae-8b3d-d60237583489.wav",
    "created_at": "2026-10-16T03:45:00.000000Z",
    "updated_at": "2026-10-16T03:45:00.000000Z"
  }
}
```

### 4.2 List Permitted Recordings
- **Route:** `GET /api/v1/recordings/?page=1`
- **Response `200 OK`:**
```json
{
  "count": 1,
  "next": null,
  "previous": null,
  "results": [
    { ... }
  ]
}
```

---

## 5. Model Registry & Predictions

### 5.1 Active Model Check
- **Route:** `GET /api/v1/models/active/`
- **Response `503 Service Unavailable` (When model weights are pending):**
```json
{
  "error": "No Active Model Deployed",
  "code": "NO_ACTIVE_MODEL",
  "message": "There is currently no verified active CNN model deployed in the registry.",
  "status": "UNAVAILABLE"
}
```

### 5.2 Request Prediction for Recording
- **Route:** `POST /api/v1/recordings/{id}/predict/`
- **Response `503 Service Unavailable` (Transparent handling of missing trained CNN weights):**
```json
{
  "error": "ML Inference Service Unavailable",
  "code": "MODEL_NOT_DEPLOYED",
  "prediction_id": "8bb3f72e-07e1-4566-a36c-949437cf0087",
  "recording_id": "e2a149b5-685b-48ae-8b3d-d60237583489",
  "message": "No trained convolutional neural network model artifact is currently available in the model registry. To perform real-time classification, a trained CNN artifact and verified label configuration must be registered and placed in ml/models/."
}
```

### 5.3 Documented Target Prediction Schema (Once real model is mounted)
- **Response `201 Created`:**
```json
{
  "id": "8bb3f72e-07e1-4566-a36c-949437cf0087",
  "recording_id": "e2a149b5-685b-48ae-8b3d-d60237583489",
  "recording_filename": "respiratory_breath_sound.wav",
  "model_version": "5a415a78-2b81-4fe6-b3e1-e1cf79c5f422",
  "model_version_str": "pediatric-cnn-v1.0.0",
  "predicted_class": "Wheeze",
  "probability_scores": {
    "Normal": 0.0821,
    "Wheeze": 0.8415,
    "Crackle": 0.0642,
    "Stridor": 0.0122
  },
  "prediction_status": "SUCCESS",
  "status_display": "Inference Successful",
  "error_details": "",
  "inference_duration_ms": 142,
  "created_at": "2026-10-16T03:45:00.000000Z"
}
```

---

## 6. Explainability (Grad-CAM)

### 6.1 Request Explanation for Prediction
- **Route:** `POST /api/v1/predictions/{id}/explanations/`
- **Request:**
```json
{
  "method": "GRAD_CAM"
}
```
- **Response `503 Service Unavailable` (When weights unavailable):**
```json
{
  "error": "Explanation Generation Unsupported",
  "code": "GRADIENT_MODEL_UNAVAILABLE",
  "explanation_id": "4168c41f-824c-47ea-a42e-1311054ee5aa",
  "prediction_id": "8bb3f72e-07e1-4566-a36c-949437cf0087",
  "message": "Grad-CAM requires access to the trained convolutional layers and weights of the CNN model. Because no model artifact is currently deployed, authentic saliency maps cannot be computed."
}
```

---

## 7. Clinical Human Assessments

### 7.1 Submit Clinician Assessment
- **Route:** `POST /api/v1/predictions/{id}/assessments/`
- **Headers:** `Authorization: Token <clinician_token>`
- **Request:**
```json
{
  "review_status": "CONFIRMED",
  "reviewer_conclusion": "WHEEZING",
  "diagnostic_agreement": true,
  "clinical_notes": "Polyphonic expiratory wheezes audible over right posterior base."
}
```
- **Response `201 Created`:**
```json
{
  "message": "Clinical assessment recorded successfully.",
  "assessment": {
    "id": "99f36c53-b0e6-422f-8da1-5c8e3bf5c4d2",
    "prediction_id": "8bb3f72e-07e1-4566-a36c-949437cf0087",
    "reviewer": 1,
    "reviewer_username": "dr_pulmonologist",
    "review_status": "CONFIRMED",
    "review_status_display": "Clinically Confirmed",
    "reviewer_conclusion": "WHEEZING",
    "reviewer_conclusion_display": "Wheezing Present (Airway Obstruction)",
    "diagnostic_agreement": true,
    "clinical_notes": "Polyphonic expiratory wheezes audible over right posterior base.",
    "created_at": "2026-10-16T03:45:00.000000Z",
    "updated_at": "2026-10-16T03:45:00.000000Z"
  }
}
```
