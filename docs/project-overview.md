# Acoustic Epidemiology: Project Overview

## 1. Project Synopsis
**Project Title:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Domain:** Biomedical Signal Processing & Deep Learning in Pediatric Pulmonology  
**Core Goal:** Build an explainable, non-invasive diagnostic prototype that analyzes pediatric respiratory auscultation recordings to detect sound abnormalities (e.g., wheezing, crackles, stridor) and classify conditions such as pneumonia and bronchiolitis.

---

## 2. Technical Stack
- **Frontend:** React 19, Vite 8, Vanilla CSS Design System, Lucide React Icons
- **Backend:** Python 3.13, Django 6.1, Django REST Framework 3.18, Django CORS Headers
- **Audio & Signal Processing:** Librosa, SciPy, NumPy (Butterworth bandpass filter, STFT, Log Mel-Spectrograms)
- **Machine Learning & Explainability:** TensorFlow / Keras (2D-CNN), Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Database:** SQLite (development fallback), MySQL (production schema planned for Step 3)

---

## 3. Backend Architecture & Django Apps
The backend is structured into five specialized applications:
1. **`accounts`**: Clinician, pediatrician, and researcher authentication, session/token management, and RBAC permissions.
2. **`recordings`**: Audio ingestion endpoint, auscultation metadata (anatomical chest location, age, patient ID), SNR validation, and audio file storage.
3. **`ml_inference`**: Audio preprocessing orchestrator, Mel-spectrogram generator, and 2D-CNN prediction inference engine.
4. **`explanations`**: Grad-CAM saliency generator to highlight time-frequency spectral regions driving the model's diagnostic score.
5. **`assessments`**: Diagnostic aggregation, risk classification index, confidence scoring, and clinician report generation.

---

## 4. API Endpoints
- **Health Verification:** `GET /api/v1/health/` (Returns HTTP 200 with service metadata and ISO timestamp)
- **Planned Endpoints:**
  - `POST /api/v1/recordings/upload/`
  - `POST /api/v1/inference/predict/`
  - `GET /api/v1/explanations/{id}/gradcam/`
  - `GET /api/v1/assessments/{id}/report/`

---

## 5. Research Team
- **Gayatri:** Group Leader & Architecture
- **Shreya:** Frontend Engineering & Client Visualization
- **Samruddhi:** Dataset Curation & Audio Signal Preprocessing
- **Mohini:** Machine Learning Modeling & CNN Architecture
- **Vishal:** Quality Assurance, Integration Testing & Clinical Research Documentation
