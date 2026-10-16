# Acoustic Epidemiology

## Project Title
Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction

## Objective
Develop an explainable clinical research prototype that analyzes pediatric respiratory auscultation audio and classifies respiratory sound categories (normal, wheezing, crackles, stridor) using time-frequency spectrogram representations, convolutional neural networks (CNNs), and Grad-CAM interpretability.

## Technology Stack
- **Frontend:** React 19, Vite 8, Vanilla CSS, Lucide React
- **Backend:** Python 3.13, Django 6.1, Django REST Framework 3.18, PyMySQL
- **Database:** MySQL 8.0+ (`acoustic_epidemiology` schema with utf8mb4)
- **Audio Processing:** SoundFile, Librosa, NumPy, SciPy (Butterworth bandpass filter, STFT, Log Mel-Spectrogram)
- **Machine Learning & Explainability:** TensorFlow/Keras (2D-CNN), Grad-CAM Saliency
- **Version Control:** Git and GitHub (Manual workflow)

## Team
- **Gayatri** — Group Leader & Backend Architecture
- **Shreya** — Frontend Development & Client Visualization
- **Samruddhi** — Dataset & Preprocessing Pipeline
- **Mohini** — Machine Learning Modeling & CNN Architecture
- **Vishal** — Testing, QA & Research Documentation

## Project Status
Backend complete:
- 5 Django apps fully implemented (`accounts`, `recordings`, `ml_inference`, `explanations`, `assessments`).
- MySQL schema generated and migrated with UUID identifiers, indexes, and relationship integrity.
- Role-based access control and token authentication operational.
- Strict audio binary header inspection and secure media storage configured.
- Clean ML inference and Grad-CAM interfaces implemented with transparent `503 Service Unavailable` handling when model weights are pending.
- Complete API contracts documented in `docs/api-contract.md`.
- 23 automated tests passing cleanly on MySQL.

## Documentation Links
- [API Contract & Integration Guide](docs/api-contract.md)
- [Backend Documentation](backend/README.md)
- [Project Overview](docs/project-overview.md)