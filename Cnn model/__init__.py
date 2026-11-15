"""
Acoustic Epidemiology: CNN Modeling Package
Automated Classification of Pediatric Respiratory & Throat Diseases

This package provides the Convolutional Neural Network (CNN) architecture,
acoustic feature preprocessor, patient-grouped dataset loaders, training pipeline,
Grad-CAM explainability, and backend integration bridges tailored for pediatric
breathing sound auscultation.
"""

from .config import (
    DISEASE_CLASSES,
    SOUND_CLASSES_4CLASS,
    SOUND_CLASSES_7CLASS,
    PREPROCESSING_CONFIG,
    SAVED_MODELS_DIR,
)

__version__ = "1.0.0"
__author__ = "Mohini"
