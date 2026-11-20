# FINAL END-TO-END SYSTEM VERIFICATION REPORT

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Verification Lead:** Gayatri (Backend & ML Integration Lead)  
**Collaborator:** Shreya (Frontend Owner), Mohini (ML Lead)  
**Date:** November 20, 2026  
**Final Status:** **FULLY VERIFIED FOR COLLEGE DEMONSTRATION**  
*(Non-diagnostic research-assistance tool; CAS recall limitation ~0–1.5% explicitly disclosed in system disclaimers and UI banners).*

---

## 1. System Architecture

The Acoustic Epidemiology platform operates as a decoupled, multi-tier system uniting machine learning inference, explainability engines, relational persistence, and role-based client interfaces:

```
+-----------------------------------------------------------------------------------+
|                                 REACT 19 FRONTEND                                 |
|  - Role Dashboards (Patient, Clinician, Administrator)                            |
|  - Multi-phase Pipeline Ingestion Component (AudioAnalysisSection)                 |
|  - Real-Time Dual-Model Prediction Visualization (AnalysisResults)                |
|  - Grad-CAM Time-Frequency Heatmap Display & Saliency Analysis                    |
+-----------------------------------------+-----------------------------------------+
                                          | JSON over HTTP / Multipart Form-Data
                                          v
+-----------------------------------------------------------------------------------+
|                              DJANGO REST FRAMEWORK                                |
|  - Token Authentication & Access Control (accounts)                               |
|  - Safe Audio Validation & Persistence (recordings)                               |
|  - Task-Filtered Model Dispatcher (ml_inference: Model A & Model B)               |
|  - Authentic Gradient Attribution Engine (explanations: GradCamService)           |
|  - Clinician Human-in-the-Loop Assessments (assessments)                          |
+--------------------+-------------------------------------+------------------------+
                     |                                     |
                     v                                     v
+---------------------------------------+  +----------------------------------------+
|          DATABASE PERSISTENCE         |  |         ML INFERENCE SUBSYSTEM         |
|  - MySQL / SQLite Relational DB       |  |  - TensorFlow / Keras 3.x              |
|  - ModelVersion & Metrics Registry    |  |  - AudioPreprocessor (SOS Bandpass,    |
|  - Prediction & DiseasePrediction     |  |    128 Mel bands, Spec Normalization)  |
|  - Grad-CAM Artifact Filesystem       |  |  - Active Acoustic CNN: Fold 0         |
|  - User Profiles & Diagnostic Notes   |  |  - Active Disease CNN: Model B         |
+---------------------------------------+  +----------------------------------------+
```

---

## 2. Environment Configuration

### 2.1 Backend Environment
- **Python Version:** Python 3.13.5 (Virtual Environment: `backend/.venv`)
- **Framework:** Django 6.1.2, Django REST Framework 3.16.1
- **Deep Learning:** TensorFlow 2.21.0, Keras 3.12.0
- **Audio Processing:** Librosa 0.11.0, SciPy 1.15.2, SoundFile 0.13.1
- **Database Engine:** SQLite / MySQL compatibility configured via environment variables
- **Execution Settings:** `MPLCONFIGDIR=/tmp KERAS_HOME=/tmp/.keras`

### 2.2 Frontend Environment
- **Node / Package Environment:** React 19.2.0, Vite 8.3.4, React Router DOM 7.13.1, Lucide React
- **Dev Server & Reverse Proxy:** Configured in `vite.config.js` to route `/api/*` and `/media/*` to `http://localhost:8000`

### 2.3 Commands to Launch the Application

**Terminal 1 — Backend API Service:**
```bash
cd backend
source .venv/bin/activate
export MPLCONFIGDIR=/tmp KERAS_HOME=/tmp/.keras
python manage.py runserver 8000
```

**Terminal 2 — Frontend Application:**
```bash
cd frontend
npm run dev
```
*(Access application in browser at: `http://localhost:5173`)*

---

## 3. End-to-End Authentication Results (Phase 2)

Tested against the live Django Authentication subsystem (`/api/v1/auth/`):

| Step | Action & Endpoint | Payload / Method | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| 1 | Register new patient account (`/api/v1/auth/register/`) | POST JSON: `username`, `email`, `password` | HTTP 201 Created. User ID assigned, Token generated. | **PASS** |
| 2 | Login with email (`/api/v1/auth/login/`) | POST JSON: `email="e2e_patient@example.com"`, `password` | HTTP 200 OK. Token key returned, user profile populated. | **PASS** |
| 3 | Access current profile (`/api/v1/auth/me/`) | GET Header: `Authorization: Token <token>` | HTTP 200 OK. Returned role: `patient`. | **PASS** |
| 4 | Session persistence & token validation | Verified local storage and auth headers | Persistent session restored on page refresh. | **PASS** |
| 5 | Logout user (`/api/v1/auth/logout/`) | POST Header: `Authorization: Token <token>` | HTTP 200 OK. Token deleted from backend database. | **PASS** |
| 6 | Post-logout guarded access attempt | GET `/api/v1/auth/me/` with invalidated token | HTTP 401 Unauthorized. Access strictly denied. | **PASS** |

---

## 4. Acoustic CNN End-to-End Results (Phase 3)

Tested with actual SPRSound recording `ml/dataset/SPRSound/Classification/train_classification_wav/65058931_3.4_1_p2_1086.wav`:

1. **File Validation & Audio Ingestion:**
   - Accepted file format: Mono WAV, 16-bit PCM, 8000 Hz.
   - Recording persisted in DB: `Recording ID: 808eea3c-430b-41c3-9859-40e78c32c06d`.
2. **Model Selection & Routing:**
   - Active Model: `sprsound-cnn-v1-fold0`.
   - Verified Task Type: `SOUND_CLASSIFICATION` (acoustic model correctly filtered; disease model was NOT selected).
3. **Real Model Output & Prediction:**
   - **Predicted Class:** `Normal`
   - **Softmax Probability Distribution:**
     - `Normal`: **0.5157** (51.57%)
     - `CAS` (Continuous Adventitious Sounds / Wheezes): **0.1263** (12.63%)
     - `DAS` (Discontinuous Adventitious Sounds / Crackles): **0.2975** (29.75%)
     - `CAS & DAS` (Co-occurring anomalies): **0.0606** (6.06%)
     - *Probability sum: 1.0001 (normalized).*
4. **Mandatory Research Disclaimer Display:**
   - Backend returned `research_disclaimer` payload.
   - Frontend `AnalysisResults.jsx` rendered the warning prominently in a dedicated amber alert banner.

---

## 5. Disease Model Routing Verification (Phase 4)

Tested against the dedicated disease screening endpoint:

1. **Endpoint Existence & Dispatch:**
   - `POST /api/v1/recordings/<uuid>/predict-disease/` verified operational.
   - Active Model: `sprsound-disease-cnn-v2` (`DISEASE_CLASSIFICATION`).
2. **Inference Independence:**
   - Disease inference runs via `DiseaseInferenceService`, creating a separate `DiseasePrediction` entity without modifying acoustic prediction records.
3. **Real Disease Prediction Output:**
   - **Predicted Disease:** `Healthy Control`
   - **Softmax Distribution:**
     - `Healthy Control`: **0.3990** (39.90%)
     - `Pneumonia`: **0.2823** (28.23%)
     - `Wheezing Disorders`: **0.3187** (31.87%)
4. **Disease Safety Disclaimer:**
   - Returns separate clinical screening disclaimer indicating results are based on hospital admission categories and must not be used for diagnosis.

---

## 6. Grad-CAM Explainability Results (Phase 5)

Tested attribution generation on actual predictions (`/api/v1/predictions/<uuid>/explanations/`):

1. **Attribution Execution:**
   - Execution Method: `GRAD_CAM`.
   - Status: HTTP 202 Accepted.
   - Activation Extraction: Gradients extracted from convolutional layer `conv2d_3` (last conv feature map).
2. **Artifact Verification:**
   - Output File: `media/explanations/fd4f1710-3c86-42c5-8bbd-c969b6487b18_gradcam.png`.
   - Verified on disk: 19 KB PNG image containing the blended Mel-spectrogram and thermal saliency overlay.
   - Frontend Rendering: Rendered directly in `AnalysisResults.jsx` under the interpretability section.

---

## 7. Database Verification & Cross-User Isolation (Phase 6)

1. **Recording & Prediction Association:**
   - Recordings and predictions are strictly linked to `request.user`.
   - History retrieval endpoint `GET /api/v1/recordings/<uuid>/predictions/` returns accurate database records.
2. **Cross-User Access Control:**
   - Tested unauthorized patient access to another user's recording: **HTTP 403 Forbidden** (Strict isolation maintained).
   - Tested clinician role access to patient recording: **HTTP 200 OK** (Clinician authorization verified).

---

## 8. Frontend & Backend Regression Results (Phase 8)

### 8.1 Frontend Production Build
```bash
> frontend@0.0.0 build
> vite build
✓ 1951 modules transformed.
dist/index.html                   1.16 kB │ gzip:   0.60 kB
dist/assets/index-COQJYxPb.css    8.55 kB │ gzip:   2.66 kB
dist/assets/index-bmeqDOIl.js   327.92 kB │ gzip: 101.54 kB
✓ built in 235ms with 0 errors.
```

### 8.2 Backend Regression Test Suite
```bash
python manage.py test
Ran 80 tests in 28.147s

OK (80/80 passed, 0 failures, 0 errors)
```

---

## 9. Bugs Found & Resolved During Integration & Verification

1. **Dashboard Authentication Guard (`onRequireAuth`) Undefined Callback:**
   - *Problem:* `AudioAnalysisSection` attempted to call `onRequireAuth()` on 401 unauthenticated errors, which was not passed by the dashboards.
   - *Fix:* Injected `useNavigate` and passed `onRequireAuth={() => { logout(); navigate('/login'); }}` in both `PatientDashboard.jsx` and `DoctorDashboard.jsx`.
2. **Static vs Dynamic Research Disclaimer:**
   - *Problem:* The disclaimer on `AnalysisResults.jsx` was previously a static hardcoded note.
   - *Fix:* Replaced with dynamic rendering of `prediction.research_disclaimer` / `diseasePrediction.research_disclaimer` directly from the API response inside a styled alert banner.
3. **Cross-Origin Development Proxy:**
   - *Problem:* Browser-based uploads in local development could encounter CORS preflight edge cases.
   - *Fix:* Configured reverse-proxy rules in `vite.config.js` for `/api` and `/media`.

---

## 10. Remaining Limitations & Disclosures

1. **Model Generalization & CAS Detection:**
   - In accordance with the audited research findings, CAS (wheeze) detection sensitivity remains severely constrained (~0–1.5% recall). The system must never be represented as clinically certified.
2. **Audio Preprocessing Strictness:**
   - Audio files must adhere to standard sampling rates and channel configurations; corrupt or truncated files trigger explicit validation errors rather than silently fabricating outputs.

---

## 11. Final Project Readiness Declaration

### Final Verification Status: **FULLY VERIFIED FOR COLLEGE DEMONSTRATION**

The end-to-end integration across React 19, Django REST Framework, and the real TensorFlow/Keras CNN models is verified functional and reproducible. Users can register, log in, ingest authentic pediatric respiratory audio, observe real CNN classifications and Grad-CAM saliency heatmaps, and retrieve saved patient histories with strict data isolation.
