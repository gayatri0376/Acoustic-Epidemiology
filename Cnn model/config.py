"""
Central Configuration for Pediatric Respiratory & Throat CNN Models.

Guarantees 100% numerical and structural compatibility with the Django
backend (ml_inference, disease_service, and explanations apps).
"""

import os
from pathlib import Path

# Paths
CNN_MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CNN_MODEL_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
ML_DIR = PROJECT_ROOT / "ml"
FEATURES_DIR = ML_DIR / "features" / "extracted"
RECORDS_DIR = FEATURES_DIR / "records"
EVENTS_DIR = FEATURES_DIR / "events"

# Backend Model Storage Directory (enforced by register_model / register_disease_model)
BACKEND_SAVED_MODELS_DIR = ML_DIR / "models" / "saved_models"
LOCAL_SAVED_MODELS_DIR = CNN_MODEL_DIR / "saved_models"

# Ensure directories exist if permitted
try:
    BACKEND_SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass

# Manifest Paths
DISEASE_MANIFEST_PATH = FEATURES_DIR / "disease_split_manifest.csv"
DATASET_SPLIT_MANIFEST_PATH = FEATURES_DIR / "dataset_split_manifest.csv"
TRAIN_EVENT_MANIFEST_PATH = FEATURES_DIR / "train_event_manifest.csv"
LABEL_MAPPING_PATH = FEATURES_DIR / "label_mapping.json"
PREPROCESSING_CONFIG_PATH = FEATURES_DIR / "preprocessing_config.json"

# ==============================================================================
# Verified Class Definitions (Strictly matching Django Models & Services)
# ==============================================================================

# Task 1: Pediatric Respiratory & Throat Disease Classification (Model B)
# Dataset provenance: SPRSound (SJTU Paediatric Respiratory Sound Database) CC BY 4.0
DISEASE_CLASSES = [
    "Healthy Control",      # SPRSound Control Group patients (no confirmed acute disease)
    "Pneumonia",            # Non-severe + severe pneumonia merged (lung consolidation / DAS)
    "Wheezing Disorders"    # Asthma + Bronchitis merged (airway obstruction / CAS)
]

DISEASE_CLASS_TO_IDX = {name: idx for idx, name in enumerate(DISEASE_CLASSES)}
IDX_TO_DISEASE_CLASS = {idx: name for idx, name in enumerate(DISEASE_CLASSES)}

# Task 2: Pediatric Respiratory Sound Classification (Record-level, Model A)
SOUND_CLASSES_4CLASS = [
    "Normal",               # Clear vesicular breath sounds without adventitious noises
    "CAS",                  # Continuous Adventitious Sounds (wheezes, rhonchi)
    "DAS",                  # Discontinuous Adventitious Sounds (fine/coarse crackles)
    "CAS & DAS"             # Co-occurring continuous and discontinuous sounds
]

# Task 3: Event-level Multiclass (includes Stridor - throat/upper airway obstruction)
SOUND_CLASSES_7CLASS = [
    "Normal",
    "Fine Crackle",
    "Wheeze",
    "Coarse Crackle",
    "Rhonchi",
    "Wheeze+Crackle",
    "Stridor"               # Throat / laryngeal / tracheal upper airway obstruction sound
]

# ==============================================================================
# Acoustic Preprocessing Specification
# (Bitwise identical to backend/ml_inference/services.py AudioPreprocessor)
# ==============================================================================
PREPROCESSING_CONFIG = {
    "sample_rate": 8000,            # 8 kHz target sample rate for pediatric lung/throat sounds
    "n_mels": 128,                  # 128 Mel frequency bins
    "n_fft": 1024,                  # 1024-point FFT (~128 ms analysis window)
    "hop_length": 512,              # 512-point hop step (64 ms frame increment)
    "f_min": 50.0,                  # Lower cutoff (suppresses heart and sensor low-freq artifacts)
    "f_max": 2000.0,                # Upper cutoff (covers key pediatric wheezes, stridor, crackles)
    "filter_order": 5,              # 5th-order zero-phase Butterworth bandpass filter
    "apply_bandpass": True,         # Enforce Butterworth bandpass filtering
    "top_db": 80.0,                 # Dynamic range for dB scaling
    "normalization": "minmax_01",   # Normalized to [0.0, 1.0]: clip((dB + 80) / 80, 0, 1)
    "fixed_time_steps": 128,        # Symmetric center-padded or center-cropped to 128 time frames
}

# Expected input shape for 2D CNN: (batch_size, n_mels, time_steps, channels)
CNN_INPUT_SHAPE = (128, 128, 1)

# Target Convolutional Layer Name (Required by backend/explanations/services.py for Grad-CAM)
TARGET_CONV_LAYER_NAME = "last_conv_layer"

# ==============================================================================
# Default Training Hyperparameters
# ==============================================================================
TRAINING_CONFIG = {
    "disease": {
        "learning_rate": 3e-4,
        "batch_size": 32,
        "epochs": 50,
        "patience": 12,
        "reduce_lr_patience": 5,
        "reduce_lr_factor": 0.3,
        "label_smoothing": 0.1,
        "architecture": "3Block-Disease-CNN",
    },
    "sound": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 40,
        "patience": 10,
        "reduce_lr_patience": 4,
        "reduce_lr_factor": 0.5,
        "label_smoothing": 0.05,
        "architecture": "4Block-Sound-CNN",
    }
}
