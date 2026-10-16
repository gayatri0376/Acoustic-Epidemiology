# CNN Model & Backend Integration Contract

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Backend Component:** `ml_inference` & `explanations` apps  
**Status:** Integration-Ready (Awaiting Trained Artifact & Verified Parameters)

---

## 1. Overview & Interface Responsibility

The Django REST backend provides the production-grade orchestration, storage, authentication, and REST API layers. It is designed to host and execute the trained Convolutional Neural Network (CNN) developed by the Machine Learning teammate (**Mohini**) using preprocessing specifications provided by the Dataset teammate (**Samruddhi**).

The backend **strictly refuses to fabricate mock predictions**. If a verified model is not registered and physically mounted on disk, prediction requests transparently return `HTTP 503 Service Unavailable` with error code `MODEL_NOT_DEPLOYED`.

---

## 2. Required Model Artifacts and Configuration Files

The ML teammate must deliver the trained model artifact to the repository under `ml/models/saved_models/`:

### Accepted Artifact Formats
1. **Keras v3 / TensorFlow Format (Recommended):**
   - Path: `ml/models/saved_models/<model_name>.keras`
2. **HDF5 Legacy Format:**
   - Path: `ml/models/saved_models/<model_name>.h5`
3. **TensorFlow SavedModel Directory:**
   - Path: `ml/models/saved_models/<model_name>/` (containing `saved_model.pb` and `variables/`)

### Metadata & Configuration Specification
Accompanying the model weights, the ML teammate must supply a configuration JSON or register the metadata directly into the `model_versions` table:
```json
{
  "version": "pediatric-cnn-v1.0.0",
  "architecture": "2D-CNN-Spectrogram",
  "framework": "TensorFlow 2.18 / Keras 3.x",
  "artifact_path": "ml/models/saved_models/pediatric_cnn_v1.keras",
  "verified_classes": [
    "Normal",
    "Wheeze",
    "Crackle"
  ],
  "preprocessing_config": {
    "sample_rate": 16000,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 512,
    "f_min": 50.0,
    "f_max": 2000.0,
    "to_db": true,
    "expected_input_shape": [1, 128, 128, 1]
  },
  "target_conv_layer": "conv2d_final"
}
```

---

## 3. Audio Preprocessing Pipeline

Audio preprocessing in the backend is encapsulated in `backend/ml_inference/services.py` (`AudioPreprocessor`):

```
Raw Audio (.wav, mono/stereo)
  │
  ▼
Mono Conversion (np.mean across channels)
  │
  ▼
Target Resampling (e.g., 16000 Hz via SciPy/Librosa)
  │
  ▼
5th-Order Butterworth Bandpass Filter (50 Hz – 2000 Hz)
  │
  ▼
Short-Time Fourier Transform (STFT) -> 128 Mel-Bands
  │
  ▼
Log-Amplitude Scaling (dB = 10 * log10(power / max_power))
  │
  ▼
4D Tensor Formatting: (1, n_mels, time_steps, 1) -> float32
```

### Preprocessing Parameters Matrix

| Parameter | Current Default | Verified / Pending | Notes |
|---|---|---|---|
| Target Sample Rate | `16000 Hz` | **[PENDING ML CONFIRMATION]** | Must match training audio rate |
| Audio Channels | `1` (Mono) | **[VERIFIED]** | Auscultation is single-channel |
| Min Audio Duration | `0.2 seconds` | **[VERIFIED]** | Enforced by upload validator |
| Target Fixed Duration | `Pending` | **[PENDING ML CONFIRMATION]** | e.g., 2.5s or 5.0s window if CNN requires fixed width |
| Mel Filterbanks (`n_mels`) | `128` | **[PENDING ML CONFIRMATION]** | Height dimension of CNN input |
| FFT Window Size (`n_fft`) | `1024` | **[PENDING ML CONFIRMATION]** | Window length for STFT |
| Hop Length (`hop_length`)| `512` | **[PENDING ML CONFIRMATION]** | Determines time resolution (width) |
| Bandpass Frequencies | `50 Hz – 2000 Hz` | **[PENDING DATASET CONFIRMATION]**| Physiological respiratory frequency band |
| Amplitude Scaling | `dB (power_to_db)` | **[PENDING ML CONFIRMATION]** | Log energy dynamic range |

---

## 4. Expected Prediction Output Structure

The CNN model must output a 2D tensor representing class probabilities:
- **Shape:** `(1, num_classes)` (e.g. `(1, 3)` for 3 classes)
- **Data Type:** `float32` or `float64`
- **Values:** Softmax probabilities between `0.0` and `1.0`, summing to $\approx 1.0$.

If the model outputs unnormalized raw logits, the ML teammate must document this so that a `tf.nn.softmax()` transformation is applied in `AudioPreprocessor` / `InferenceService`.

---

## 5. Class Label Mapping

The mapping between output tensor indices and human-readable clinical classes is strictly ordered by index:

$$\text{Index } i \iff \text{verified\_classes}[i]$$

### Example Mapping (3-Class Pediatric Model):
```
Output Tensor: [[0.0821, 0.8415, 0.0764]]
verified_classes: ["Normal", "Wheeze", "Crackle"]

Resulting Probability Map:
{
  "Normal": 0.0821,
  "Wheeze": 0.8415,
  "Crackle": 0.0764
}
Predicted Class: "Wheeze" (Argmax)
```

> **CRITICAL RULE:** `verified_classes` must never be guessed. The ordering must be identical to `train_generator.class_indices` or `LabelEncoder.classes_` used during training.

---

## 6. Model Version Registration and Activation

Models are registered in the MySQL `model_versions` table via Django Admin or programmatic scripts:

```python
from ml_inference.models import ModelVersion, ModelStatus

model_v1 = ModelVersion(
    version="pediatric-cnn-v1.0.0",
    architecture="2D-CNN-Spectrogram",
    artifact_path="ml/models/saved_models/pediatric_cnn.keras",
    verified_classes=["Normal", "Wheeze", "Crackle"],
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
        "validation_accuracy": 0.892,
        "val_loss": 0.281,
        "training_epochs": 50,
        "dataset": "Pediatric Respiratory Sound Database"
    }
)
model_v1.full_clean()  # Validates that artifact physically exists on disk
model_v1.save()
```

### Automatic Validation Guards in `ModelVersion.clean()`:
1. Cannot set `status='ACTIVE'` if `artifact_path` does not exist on disk.
2. Cannot set `status='ACTIVE'` if `verified_classes` list is empty.

---

## 7. Inference Caching & Failure Handling

### Reusable In-Memory Cache (`ModelCache`)
To prevent loading multi-megabyte model weights on every single prediction request:
- `ModelCache` loads the model once and retains the instance in memory.
- Uses thread-safe double-checked locking.
- Automatically detects file modification time (`mtime`) on disk and reloads if the artifact is updated.
- Invalidates and reloads if a new `ModelVersion` is activated.

### Failure Modes & Status Codes

| Scenario | HTTP Status | Error Code | Persistence in Database |
|---|---|---|---|
| No active model version in database | `503 Service Unavailable` | `NO_ACTIVE_MODEL` | None |
| Active model configured, but weights file missing | `503 Service Unavailable` | `MODEL_NOT_DEPLOYED` | `Prediction(status='UNAVAILABLE')` |
| Audio recording file missing on disk storage | `500 Internal Error` | `APIException` | `Prediction(status='FAILED')` |
| Audio is empty or unreadable by SoundFile | `500 Internal Error` | `INFERENCE_FAILED` | `Prediction(status='FAILED')` |
| Model output dimension $\neq$ `len(verified_classes)` | `500 Internal Error` | `DIMENSION_MISMATCH` | `Prediction(status='FAILED')` |
| Model output contains `NaN` or `Inf` | `500 Internal Error` | `INVALID_OUTPUT` | `Prediction(status='FAILED')` |

---

## 8. Output Validation Checks

Before committing a prediction to the database, `InferenceService.validate_model_output()` verifies:
1. Output tensor is non-empty and has at least 1 batch prediction.
2. Output probability count matches `len(model_version.verified_classes)` exactly.
3. No values are `NaN` or `Inf`.
4. Values are valid numerical probabilities.

---

## 9. Grad-CAM Explainability Integration

The `explanations` app provides visual interpretability via Gradient-weighted Class Activation Mapping (Grad-CAM):

### Architectural Requirements:
1. **Conv2D Layer Presence:** The CNN must contain at least one 2D convolutional layer (`tf.keras.layers.Conv2D`).
2. **Target Layer Identification:** The ML teammate must designate the target layer (default: final convolutional layer prior to global pooling / dense layers).
3. **Model API:** The model should be constructed using the Keras Functional API (`keras.Model(inputs, outputs)`) or a standard Sequential model so sub-layer activations can be extracted.
4. **Gradient Flow:** Activation functions in the convolutional trunk must allow gradient backpropagation.

If weights are missing or the model topology lacks Conv2D layers, `GradCamService` returns `HTTP 503` (`code: GRADIENT_MODEL_UNAVAILABLE` or `code: CONV_LAYER_NOT_FOUND`) and persists `Explanation(status='UNSUPPORTED')`. Fake heatmaps are never generated.

---

## 10. Checklist for Dataset & ML Teammates

Before connecting the real model, the following items must be provided:

- [ ] **Artifact File:** Trained weights file deposited into `ml/models/saved_models/`.
- [ ] **Exact Class Names & Ordering:** E.g., `["Normal", "Wheeze", "Crackle"]` with verified indices.
- [ ] **Input Sample Rate:** Target audio sampling frequency (e.g. 16,000 Hz).
- [ ] **Input Tensor Shape:** Expected input dimension (e.g. `[batch, 128, 128, 1]`).
- [ ] **Feature Normalization:** Any global mean/std or min-max normalization used during training.
- [ ] **Grad-CAM Conv Layer Name:** Name of the final convolutional layer (e.g. `conv2d_3` or `final_conv`).
- [ ] **Evaluation Metrics:** Test accuracy, sensitivity, specificity, and loss for inclusion in `evaluation_metadata`.
- [ ] **Verification Script:** A standalone Python snippet showing `model.predict()` running on a sample `.wav`.
