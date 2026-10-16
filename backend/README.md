# Acoustic Epidemiology: Django REST Framework Backend

Backend service for **Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction**.

---

## 1. Architecture Overview & App Responsibilities

The backend is structured into five domain applications:

| App | Responsibility | Key Models |
|---|---|---|
| **`accounts`** | User authentication, session/token management, and role-based access control (Clinician, Researcher, Uploader, Administrator). | `UserProfile` (linked to Django `User`) |
| **`recordings`** | Pediatric respiratory audio ingestion, strict binary header validation, acoustic metadata (anatomical location, age group, SNR), and secure file storage. | `Recording` |
| **`ml_inference`** | Model version registry, preprocessing orchestration, and CNN classification inference engine. | `ModelVersion`, `Prediction` |
| **`explanations`** | Model interpretability layer generating Grad-CAM heatmaps over Log-Mel Spectrograms. | `Explanation` |
| **`assessments`** | Human clinician review, diagnostic confirmation, and physician auscultation notes. | `Assessment` |

---

## 2. Entity Relationships

```
        ┌───────────────┐
        │   auth_user   │
        └───────┬───────┘
                │ 1:1
        ┌───────▼───────┐
        │  user_profiles│
        └───────────────┘
                │
                │ 1:N
        ┌───────▼───────┐
        │  recordings   │
        └───────┬───────┘
                │ 1:N
        ┌───────▼───────┐        ┌──────────────┐
        │  predictions  │◄───────┤model_versions│
        └───────┬───────┘   N:1  └──────────────┘
                ├──────────────────────┐
                │ 1:N                  │ 1:N
        ┌───────▼───────┐      ┌───────▼──────┐
        │  explanations │      │ assessments  │
        └───────────────┘      └──────────────┘
```

- **Uploader to Recordings:** One uploader may upload multiple recordings (`1:N`).
- **Recording to Predictions:** One recording can have multiple prediction attempts (`1:N`).
- **Model Version to Predictions:** One model version can serve many predictions (`1:N`).
- **Prediction to Explanations:** One prediction can have multiple explanation records (e.g. Grad-CAM, saliency) (`1:N`).
- **Prediction to Assessments:** Clinicians review predictions independently (`1:N`). AI predictions and human conclusions remain strictly segregated.

---

## 3. Database & MySQL Setup

The backend utilizes **MySQL 8.0+** with `utf8mb4` character encoding.

### Create Database
```sql
CREATE DATABASE acoustic_epidemiology CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### Environment Configuration (`.env`)
```ini
DEBUG=True
SECRET_KEY=django-insecure-your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# MySQL Database Settings
DB_ENGINE=django.db.backends.mysql
DB_NAME=acoustic_epidemiology
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=127.0.0.1
DB_PORT=3306
```

---

## 4. Execution Commands

### Setup Virtual Environment
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Apply Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Run Test Suite
```bash
python manage.py test
```

### Start Development Server
```bash
python manage.py runserver 127.0.0.1:8000
```

---

## 5. API Endpoints Catalog

All endpoints are versioned with the `/api/v1/` prefix:

- **System & Health Probes:**
  - `GET /api/v1/` — Complete API endpoint index
  - `GET /api/v1/health/` — Liveness health check
  - `GET /api/v1/ready/` — Readiness check verifying database, storage, and model deployment

- **Authentication (`accounts`):**
  - `POST /api/v1/auth/register/` — Public user registration (default: UPLOADER role)
  - `POST /api/v1/auth/login/` — Authenticate and receive DRF Auth Token
  - `POST /api/v1/auth/logout/` — Invalidate user token
  - `GET /api/v1/auth/me/` — Retrieve authenticated user profile and roles

- **Audio Recordings (`recordings`):**
  - `POST /api/v1/recordings/` — Multipart audio upload with validation
  - `GET /api/v1/recordings/` — List recordings permitted to user
  - `GET /api/v1/recordings/{id}/` — Retrieve recording metadata
  - `DELETE /api/v1/recordings/{id}/` — Delete recording and clean up audio storage

- **Model Registry & Predictions (`ml_inference`):**
  - `GET /api/v1/models/` — List registered CNN model versions
  - `GET /api/v1/models/active/` — Inspect active model or get 503 if none deployed
  - `POST /api/v1/recordings/{id}/predict/` — Request classification inference
  - `GET /api/v1/predictions/` — List predictions accessible to user
  - `GET /api/v1/predictions/{id}/` — Inspect single prediction and softmax probabilities
  - `GET /api/v1/recordings/{id}/predictions/` — Prediction history for a recording

- **Explainability (`explanations`):**
  - `POST /api/v1/predictions/{id}/explanations/` — Generate Grad-CAM saliency
  - `GET /api/v1/predictions/{id}/explanations/` — List explanations for prediction
  - `GET /api/v1/explanations/{id}/` — Retrieve explanation metadata and artifact

- **Clinical Assessments (`assessments`):**
  - `POST /api/v1/predictions/{id}/assessments/` — Submit clinician review (Requires CLINICIAN role)
  - `GET /api/v1/assessments/` — List clinical assessments
  - `GET /api/v1/assessments/{id}/` — Retrieve assessment details
  - `PATCH /api/v1/assessments/{id}/` — Update assessment conclusion or clinical notes

---

## 6. Model Integration & Known Blockers

- **Trained Model Artifact:** Currently, no trained CNN weights are mounted in `ml/models/`.
- **Handling of Unmounted Model:** The inference service does **not** fabricate fake predictions. It records an `UNAVAILABLE` prediction status and responds with HTTP `503 Service Unavailable` (`code: MODEL_NOT_DEPLOYED`).
- **To Deploy a Model:**
  1. Train CNN model and export weights (e.g. `ml/models/saved_models/cnn_v1.keras` or `.h5`).
  2. Create an active `ModelVersion` entry in the database with verified class labels and preprocessing parameters.
  3. The system will automatically detect the weights and transition into live inference mode.
