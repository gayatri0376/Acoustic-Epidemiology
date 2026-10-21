# CNN Model & Backend Integration Contract

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Backend Component:** `ml_inference` & `explanations` apps  
**Date:** 2026-10-21  
**Status:** Integrated & Verified with Preprocessing Standardization and Authentic Grad-CAM Pipeline

---

## 1. Overview & Interface Responsibility

The Django REST backend provides the production-grade orchestration, storage, authentication, caching, and REST API layers. It is designed to host and execute the trained Convolutional Neural Network (CNN) developed for the SPRSound pediatric respiratory dataset.

The backend **strictly refuses to fabricate mock predictions or synthetic heatmaps**:
- If no verified model is active or deployed, inference requests transparently return `HTTP 503 Service Unavailable` with error code `MODEL_NOT_DEPLOYED` and persist an audit record with `prediction_status='UNAVAILABLE'`.
- If Grad-CAM is requested without a deployed model or on an architecture lacking Conv2D layers, requests return `HTTP 503 Service Unavailable` with error code `GRADIENT_MODEL_UNAVAILABLE` or `CONV_LAYER_NOT_FOUND` and persist `status='UNSUPPORTED'`.

---

## 2. Standardized Preprocessing Specification

The preprocessing pipeline in `backend/ml_inference/services.py` (`AudioPreprocessor`) is verified to be bitwise and numerically identical to `ml/preprocessing/respiratory_preprocessor.py`:

```
Raw Auscultation WAV (8,000 Hz, 16-bit Mono)
  │
  ▼
Mono Channel Averaging (if multi-channel)
  │
  ▼
Zero-Phase 5th-Order Butterworth Bandpass (50 Hz – 2,000 Hz)
  │
  ▼
STFT: n_fft=1024, hop_length=512 (64 ms frame step)
  │
  ▼
Mel Filterbank (n_mels=128, f_min=50 Hz, f_max=2000 Hz)
  │
  ▼
Power-to-dB Scaling (top_db=80.0 relative to peak)
  │
  ▼
Min-Max Normalization to [0.0, 1.0]: clip((dB + 80) / 80, 0, 1)
  │
  ▼
Symmetric Center-Pad / Crop to 128 Frames -> Tensor (1, 128, 128, 1)
```

Numerical comparison confirms:
- **Max Absolute Difference:** `0.0`
- **Mean Absolute Difference:** `0.0`

---

## 3. Verified Classes & Output Mapping

Defined in `ml/features/extracted/label_mapping.json`:

```json
{
  "0": "Normal",
  "1": "CAS",
  "2": "DAS",
  "3": "CAS & DAS"
}
```

- When inference completes, the raw softmax output tensor `(1, 4)` is validated:
  - Dimensionality matches `len(verified_classes) == 4`.
  - Non-finite check: verifies no `NaN` or `Inf` values exist.
  - Normalized probability map `{class_name: round(prob, 4)}` is computed.
  - `predicted_class` is assigned to `argmax(probabilities)`.

---

## 4. Authentic Grad-CAM Pipeline

Implemented in [`backend/explanations/services.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/backend/explanations/services.py):

1. **Target Layer Identification:**
   `find_last_conv_layer(model)` programmatically inspects the model topology to locate the target 2D convolutional layer (`last_conv_layer`).
2. **Gradient Backpropagation:**
   Uses `tf.GradientTape` to compute the gradient of the predicted class logit with respect to the feature map activations of the last convolutional layer.
3. **Channel-Wise Pooling & Weighted Activation:**
   Global average pooling across gradients produces importance weights $\alpha_k^c$:
   $$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Artifact Persistence:**
   The normalized heatmap is rendered as an overlay image and persisted to `media/explanations/<prediction_id>_gradcam.png`.
5. **Structured Explainability Data:**
   Time saliency profiles, frequency saliency profiles, and a mandatory clinical interpretability disclaimer are stored in the database.

---

## 5. Model Registration & Management Command

Models are registered via the management command:

```bash
python manage.py register_model \
  --model-version sprsound-cnn-v1-fold0 \
  --artifact-path ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --classes Normal CAS DAS "CAS & DAS" \
  --status ACTIVE \
  --architecture 2D-CNN-Spectrogram-4Block \
  --target-conv-layer last_conv_layer \
  --set-active
```

### Safety Guards Enforced by `register_model`:
1. **Directory Traversal Guard:** Rejects any path resolving outside the project root.
2. **Storage Convention Guard:** Rejects any path outside `ml/models/`.
3. **Physical Existence Guard:** Rejects missing files before touching the database.
4. **Format Guard:** Enforces valid `.keras`, `.h5`, or SavedModel directories.
5. **Dimension Validation:** Verifies input shape `[None, 128, 128, 1]` and checks output neurons against verified classes.
6. **Conv Layer Validation:** Verifies that `target_conv_layer` physically exists in model layers.
7. **Duplicate Protection:** Rejects registering duplicate versions unless `--overwrite` is specified.
