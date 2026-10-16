# Machine Learning Team Handoff & Model Delivery Guide

**To:** Mohini (Machine Learning Lead) & Samruddhi (Dataset & Preprocessing Lead)  
**From:** Gayatri (Backend Lead & Architecture)  
**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Date:** October 2026  
**Status:** Backend Infrastructure Ready for Real CNN Weights

---

## 1. Executive Summary

The Django REST backend architecture, MySQL database, audio validation pipeline, and prediction/explanation APIs are fully implemented and verified (passing all 34 automated integration tests).

To connect your trained Convolutional Neural Network (CNN) to the live system, we need your model artifact, preprocessing parameters, and label ordering. This document specifies exactly what you need to provide and where to place your files.

---

## 2. Deliverables Checklist

Please provide the following items when your training is complete:

### [ ] 1. Trained Model File
- **Destination:** `ml/models/saved_models/<model_filename>.keras` (or `.h5`)
- **Preferred Format:** Keras v3 / TensorFlow SavedModel (`.keras`).
- **Dependencies:** Specify TensorFlow / Keras version used (e.g. `tensorflow==2.18.0`).

### [ ] 2. Exact Class Label Order
The backend maps output probabilities index-by-index to class names. We need the exact Python list matching your model's output neurons:
```python
# EXAMPLE (Please replace with your actual training class ordering):
VERIFIED_CLASSES = [
    "Normal",      # Index 0
    "Wheeze",      # Index 1
    "Crackle",     # Index 2
    "Stridor"      # Index 3 (if applicable)
]
```
> **Important:** If your training used `ImageDataGenerator.class_indices`, please provide that dictionary directly.

### [ ] 3. Audio Preprocessing Parameters
The backend needs the exact parameters used during dataset feature extraction so inference matches training conditions:
- **Audio Sample Rate ($f_s$):** e.g., `16000 Hz` or `22050 Hz`
- **Frequency Bandpass Filter:** Low cutoff (e.g. 50 Hz) and High cutoff (e.g. 2000 Hz)
- **Mel Filterbanks ($n\_mels$):** e.g., `128` or `64`
- **STFT Window Length ($n\_fft$):** e.g., `1024` or `2048`
- **Hop Length ($hop\_length$):** e.g., `512`
- **Audio Duration Window:** Fixed duration (e.g. 2.5s, 5.0s) or variable width with zero-padding
- **Decibel Scaling:** Method used (e.g. `librosa.power_to_db(..., ref=np.max)`)
- **Spectrogram Normalization:** Any z-score standardization ($(\mu, \sigma)$) or min-max normalization applied to pixel values.

### [ ] 4. Expected CNN Input Tensor Shape
Please confirm the input dimensions required by your model's first layer:
- e.g. `(batch_size, 128, 128, 1)` or `(batch_size, 128, 256, 1)`.

### [ ] 5. Grad-CAM Target Layer Name
For explainability heatmap generation, we need the layer name of the last 2D Convolutional layer before global pooling/dense layers:
- e.g. `'conv2d_3'`, `'block5_conv3'`, or `'final_conv'`.
- You can find this by running `model.summary()` and noting the last `Conv2D` layer.

### [ ] 6. Model Evaluation Metadata
For clinical research tracking, please supply your test set metrics:
- Overall Validation Accuracy (e.g., `88.5%`)
- Sensitivity / Recall per class
- Specificity per class
- Confusion matrix array
- Dataset partition description (e.g. 70/15/15 train/val/test split across patient IDs)

---

## 3. Minimal Reproducible Verification Example

Please provide a short script (or fill in this template in `ml/models/verify_model.py`) demonstrating that your saved model executes on a sample `.wav` file:

```python
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

MODEL_PATH = "ml/models/saved_models/pediatric_cnn.keras"
SAMPLE_AUDIO = "path/to/sample_wheeze.wav"

# 1. Load Model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully. Input shape:", model.input_shape)

# 2. Preprocess Sample Audio
audio, sr = sf.read(SAMPLE_AUDIO)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Target sample rate & Mel-Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=16000, n_mels=128, n_fft=1024, hop_length=512, fmin=50, fmax=2000
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Format to 4D tensor: (1, 128, time_steps, 1)
input_tensor = np.expand_dims(mel_spec_db, axis=(0, -1)).astype(np.float32)

# 3. Forward Pass
predictions = model.predict(input_tensor)
print("Raw output:", predictions)
print("Softmax sum:", np.sum(predictions))
```

---

## 4. How the Backend Connects Your Model

Once you place your weights file in `ml/models/saved_models/`, we will register it with one command:

```python
python manage.py shell
```
```python
from ml_inference.models import ModelVersion, ModelStatus

ModelVersion.objects.create(
    version="pediatric-cnn-v1.0.0",
    architecture="2D-CNN-Spectrogram",
    artifact_path="ml/models/saved_models/pediatric_cnn.keras",
    verified_classes=["Normal", "Wheeze", "Crackle"],  # Your verified list
    preprocessing_config={
        "sample_rate": 16000,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
        "f_min": 50.0,
        "f_max": 2000.0,
        "to_db": True
    },
    status=ModelStatus.ACTIVE,
    evaluation_metadata={
        "test_accuracy": 0.892,
        "dataset": "Pediatric Respiratory Auscultation Dataset"
    }
)
```

The backend's `InferenceService` and `ModelCache` will immediately:
1. Detect the active version and load the weights into cached memory.
2. Accept real-time audio uploads via `POST /api/v1/recordings/`.
3. Preprocess audio into Mel-Spectrograms matching your exact parameters.
4. Execute genuine CNN predictions via `POST /api/v1/recordings/{id}/predict/`.
5. Return authentic softmax probability distributions to the React frontend.
6. Generate Grad-CAM heatmaps for clinician review.

If you have any questions or need custom feature transforms (e.g. MFCCs or Chroma in addition to Mel-Spectrograms), let me know and we will adjust the backend pipeline accordingly!
