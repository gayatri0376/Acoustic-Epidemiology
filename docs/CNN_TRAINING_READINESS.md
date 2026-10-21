# CNN Training Pipeline & Environment Readiness Report

**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Date:** 2026-10-21  
**Status:** Pipeline Fully Implemented; Training Blocked Awaiting User Approval to Install ML Framework

---

## 1. Machine-Learning Environment Audit (Read-Only)

A read-only audit of the host environment was performed:

| Dimension | Host Environment | Compatibility Assessment |
|---|---|---|
| **Operating System** | macOS 15.4 (Darwin 24.5.0) | Supported (Apple Silicon Darwin Kernel) |
| **CPU Architecture** | Apple Silicon ARM64 (M2 / T8112) | Fully supported by native ARM64 wheels |
| **Python Version** | **Python 3.13.5** | Supported (Python 3.13 wheels available for TensorFlow 2.21+) |
| **Active Virtualenv** | `backend/.venv` | Active virtual environment |
| **NumPy** | 2.5.4 | Installed and verified |
| **SciPy** | 1.18.1 | Installed and verified |
| **Librosa** | 1.0.0 | Installed and verified |
| **Scikit-learn** | 1.9.1 | Installed and verified |
| **Soundfile / Soxr** | 0.14.0 / 1.1.0 | Installed and verified |
| **TensorFlow** | **NOT INSTALLED** | Required for CNN model training & inference |
| **Keras** | **NOT INSTALLED** | Required for CNN model architecture & layers |
| **h5py** | **NOT INSTALLED** | Required for HDF5 / Keras model serialization |
| **PyTorch** | **NOT INSTALLED** | Optional alternative |

---

## 2. Dependency Isolation & Approval Request

In strict compliance with Absolute Rule 3 (*"Do not install Python packages, upgrade dependencies, or alter virtual environments without my approval"*), zero packages were installed or upgraded.

### Verified Package Resolution (`--dry-run`)
A dry-run resolution confirmed that official Apple Silicon ARM64 wheels are available on PyPI:
- `tensorflow-2.21.0-cp313-cp313-macosx_12_0_arm64.whl` (223.8 MB)
- `keras-3.15.1` (2.4 MB)
- `h5py-3.14.0-cp313-cp313-macosx_11_0_arm64.whl` (2.8 MB)

### Installation Commands Prepared for User Approval

**Option A (Recommended — Direct into existing virtual environment):**
```bash
backend/.venv/bin/pip install tensorflow keras h5py matplotlib
```

**Option B (Isolated ML-Only Environment):**
If you prefer to keep ML dependencies completely isolated from the Django backend environment:
```bash
python3.13 -m venv ml/.ml_venv
ml/.ml_venv/bin/pip install tensorflow keras h5py matplotlib librosa scipy scikit-learn soundfile
```

---

## 3. CNN Architecture Specification

Implemented in [`ml/training/train_cnn.py`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/ml/training/train_cnn.py):

- **Input Tensor:** `(batch_size, 128, 128, 1)` (128 Mel bands $\times$ 128 Time frames, 1 channel)
- **ConvBlock 1:** `Conv2D(32, (3, 3), padding='same')` $\to$ `BatchNormalization` $\to$ `ReLU` $\to$ `MaxPool2D((2, 2))` $\to$ `Dropout(0.2)`
- **ConvBlock 2:** `Conv2D(64, (3, 3), padding='same')` $\to$ `BatchNormalization` $\to$ `ReLU` $\to$ `MaxPool2D((2, 2))` $\to$ `Dropout(0.2)`
- **ConvBlock 3:** `Conv2D(128, (3, 3), padding='same')` $\to$ `BatchNormalization` $\to$ `ReLU` $\to$ `MaxPool2D((2, 2))` $\to$ `Dropout(0.3)`
- **ConvBlock 4:** `Conv2D(256, (3, 3), padding='same', name='last_conv_layer')` $\to$ `BatchNormalization` $\to$ `ReLU` $\to$ `GlobalAveragePooling2D` $\to$ `Dropout(0.4)`
- **Classification Head:** `Dense(128, activation='relu')` $\to$ `Dropout(0.4)` $\to$ `Dense(num_classes, activation='softmax', name='output_probabilities')`
- **Explainability Target:** The final 2D convolutional layer is explicitly named `last_conv_layer`, directly targeting the [`GradCamService`](file:///Users/gayatri/Downloads/Django/Acoustic-Epidemiology/backend/explanations/services.py) gradient extraction hook.

---

## 4. Patient-Grouped Splitting & Imbalance Safeguards

1. **Patient Leakage Prevention:**
   The training and validation sets are split strictly using 5-fold `StratifiedGroupKFold` grouped on `patient_id`. All recordings from any given patient (across chest positions `p1` through `p4`) reside in exactly one fold.
2. **Leakage Verification:**
   Cross-validation fold patient intersection is checked programmatically before training begins:
   $$\text{Train Fold Patients} \cap \text{Val Fold Patients} = \emptyset$$
   If any patient overlap is detected, the pipeline immediately raises a `ValueError`.
3. **Class Imbalance Handling:**
   Balanced class weights are computed using inverse class frequencies based **strictly** on the training fold:
   $$w_c = \frac{N_{\text{train}}}{K \cdot N_{c, \text{train}}}$$
   Validation and test labels are never used to compute weights or transformations.
4. **Independent Benchmark Isolation:**
   Neither the 2022 inter-subject test set (355 recordings) nor the 2023 test set (871 recordings) are ever touched during model training, early stopping, or checkpoint selection.

---

## 5. Command Reference for Execution After Approval

Once TensorFlow is installed, run training and evaluation with these exact commands:

```bash
# 1. Feature extraction for training partition (if not already extracted):
backend/.venv/bin/python ml/features/extract_features.py --mode train

# 2. Dry-run pipeline verification:
backend/.venv/bin/python ml/training/train_cnn.py --task SPRSOUND_RECORD_4CLASS --val-fold 0 --dry-run

# 3. Genuine CNN Training:
backend/.venv/bin/python ml/training/train_cnn.py --task SPRSOUND_RECORD_4CLASS --val-fold 0 --epochs 30 --batch-size 32

# 4. Independent Benchmark Evaluation (2022 Inter-Subject Test):
backend/.venv/bin/python ml/evaluation/evaluate_model.py \
  --model-artifact ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --partition inter_2022 \
  --task SPRSOUND_RECORD_4CLASS

# 5. Production Model Registration in Django:
backend/.venv/bin/python backend/manage.py register_model \
  --model-version sprsound-cnn-v1-fold0 \
  --artifact-path ml/models/saved_models/sprsound-cnn-v1-fold0.keras \
  --classes Normal CAS DAS "CAS & DAS" \
  --status ACTIVE \
  --architecture 2D-CNN-Spectrogram-4Block \
  --target-conv-layer last_conv_layer \
  --set-active
```
