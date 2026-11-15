# Pediatric Respiratory & Throat CNN Modeling Package

**Developer:** Mohini (Machine Learning Modeling & CNN Architecture)  
**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using Convolutional Neural Networks and Time-Frequency Feature Extraction  
**Dataset Provenance:** SPRSound (SJTU Paediatric Respiratory Sound Database) CC BY 4.0  
**Backend Parity:** 100% compliant with Gayatri's Django Backend (`ml_inference`, `disease_service`, and `explanations` apps)  

---

## 1. Project Overview & Clinical Context

In this project, users submit audio recordings of children's breathing sounds (`.wav` format) captured from electronic stethoscopes or microphones. The Convolutional Neural Network (CNN) analyzes the time-frequency acoustic patterns to predict throat and respiratory diseases and adventitious breath sounds.

### Classification Tasks

#### Task 1: Throat & Respiratory Disease Prediction (Model B — Primary)
Trained on patient-level clinical hospital admission and discharge summaries across 100% pediatric subjects (ages 0–16.2 years):
1. **Healthy Control**: Clear acoustic profile from the SPRSound control group with no acute respiratory diagnosis.
2. **Pneumonia**: Lower respiratory infection marked by consolidation, bronchial breath sounds, and crackles (DAS).
3. **Wheezing Disorders**: Merged Asthma and Bronchitis patients presenting with airway constriction, throat/airway reactivity, and expiratory wheezes (CAS).

#### Task 2: Upper Airway & Adventitious Sound Classification (Model A)
1. **Normal**: Clear vesicular breath sounds without adventitious noises.
2. **CAS (Continuous Adventitious Sounds)**: Wheezing and rhonchi caused by airway narrowing.
3. **DAS (Discontinuous Adventitious Sounds)**: Fine and coarse crackles caused by fluid or collapsed alveolar opening.
4. **CAS & DAS**: Simultaneous co-occurrence of continuous and discontinuous sounds.
5. **Stridor** (Event-level): Harsh, high-pitched vibrating sound produced by turbulent airflow through a narrowed or obstructed upper airway / throat (larynx, trachea).

---

## 2. Model Architecture & Backend Compatibility

The CNN architecture is designed specifically to meet the strict requirements enforced by Gayatri's backend:

```
Input: Raw Child Breathing WAV Audio (Mono, 8,000 Hz)
   │
   ▼
[AudioPreprocessor]
   ├── 5th-order Butterworth Bandpass Filter (50 Hz – 2,000 Hz)
   ├── STFT (n_fft=1024, hop_length=512)
   ├── Mel Filterbank (n_mels=128, f_min=50 Hz, f_max=2000 Hz)
   ├── Decibel Scaling (top_db=80.0)
   ├── Min-Max Normalization to [0.0, 1.0]: clip((dB + 80) / 80, 0, 1)
   └── Symmetrical Center-Pad / Center-Crop to 128 frames
   │
   ▼
Input Tensor: (1, 128, 128, 1) [float32]
   │
   ▼
[PediatricThroatDiseaseCNN]
   ├── ConvBlock 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.25)
   ├── ConvBlock 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.25)
   ├── ConvBlock 3: Conv2D(128, 3x3, name='last_conv_layer') -> BatchNorm -> ReLU -> GAP -> Dropout(0.4)
   │                └── *Grad-CAM hook target for authentic explainability*
   ├── Dense Head: Dense(64, ReLU) -> Dropout(0.4)
   └── Output Layer: Dense(3, Softmax, name='disease_output')
   │
   ▼
Predictions: Normalized Probability Map:
   {"Healthy Control": 0.0412, "Pneumonia": 0.8924, "Wheezing Disorders": 0.0664}
```

### Key Backend Guards Satisfied
1. **Input Shape:** Strictly `(None, 128, 128, 1)` matching `backend/ml_inference/services.py:prepare_cnn_input`.
2. **Grad-CAM Layer Name:** The final 2D convolutional layer is explicitly named `last_conv_layer`, directly targeting `backend/explanations/services.py:GradCamService`.
3. **Artifact Serialization:** Saved in standard Keras format (`.keras`), loadable with `tf.keras.models.load_model(artifact_path, compile=False)`.
4. **Storage Convention:** Saved in `ml/models/saved_models/<version>.keras` compliant with Django path guards.
5. **No Data Leakage:** Stratified 5-fold cross-validation grouped strictly by `patient_id` ensures zero patient overlap between train and validation sets ($\text{Train} \cap \text{Val} = \emptyset$).
6. **Class Imbalance:** Balanced class weights ($w_c = \frac{N}{C \cdot N_c}$) mitigate pediatric clinical imbalance.

---

## 3. Package Structure

```
Cnn model/
├── __init__.py           # Package exports and version metadata
├── config.py             # Central config: acoustic specs, class names, file paths, hyperparameters
├── preprocessor.py       # AudioPreprocessor: mono, Butterworth bandpass, Mel-spectrogram, padding
├── architecture.py       # Keras CNN architectures (Disease CNN & Sound CNN) with 'last_conv_layer'
├── dataset_loader.py     # Patient-grouped dataset loader, SpecAugment spectrogram augmentation
├── train.py              # Full training pipeline with checkpoints, early stopping, and metrics
├── predict.py            # Inference engine for classifying children's breathing audio (.wav)
├── explain_gradcam.py    # Grad-CAM saliency heatmaps overlaid on Mel-spectrogram
├── backend_sync.py       # Bridge to register trained models into Django backend
├── test_cnn_pipeline.py  # Unit and integration test suite
└── README.md             # Complete documentation
```

---

## 4. Setup & Installation

Install the required ML and audio dependencies:

```bash
pip install tensorflow keras matplotlib soundfile scipy librosa
```

---

## 5. Usage Guide

### 5.1 Run Integration Tests
Verify that preprocessing, tensor shapes, and patient disjointness pass:
```bash
python test_cnn_pipeline.py
```

### 5.2 Train the CNN Model

#### Train the Disease Model (Healthy Control, Pneumonia, Wheezing Disorders)
```bash
python train.py --task disease --val-fold 0 --epochs 50 --batch-size 32
```

#### Train the Sound Classification Model (Normal, CAS, DAS, CAS&DAS)
```bash
python train.py --task sound --val-fold 0 --epochs 40 --batch-size 32
```

#### Quick Dry-Run Verification (Tests forward pass and backend verification without full training)
```bash
python train.py --task disease --dry-run
```

### 5.3 Classify Children's Breathing Audio (Inference)

Submit any child's respiratory audio recording (`.wav`):
```bash
# Classify disease
python predict.py --audio path/to/child_breath_sound.wav --task disease

# Classify sound characteristics (wheezes, crackles, normal)
python predict.py --audio path/to/child_breath_sound.wav --task sound

# Test pipeline with a synthesized pediatric breath sound
python predict.py --synthesize-test
```

Example Output:
```
=================================================================
      PEDIATRIC RESPIRATORY & THROAT CNN PREDICTION REPORT
=================================================================
 Audio File        : child_breath_recording.wav
 Model Artifact    : sprsound-disease-cnn-v1-fold0.keras
 Task Evaluated    : DISEASE
 Inference Latency : 38 ms
-----------------------------------------------------------------
 PREDICTED CLASS   : >>> WHEEZING DISORDERS <<<
 CONFIDENCE        : 88.42%
-----------------------------------------------------------------
 Probability Distribution:
   Healthy Control     : [##----------------------------]   6.15%
   Pneumonia           : [##----------------------------]   5.43%
   Wheezing Disorders  : [##########################----]  88.42%
-----------------------------------------------------------------
 Clinical Context  : High-pitched airway narrowing detected (CAS / Wheezes).
                     Associated with pediatric reactive airway disease, asthma, or bronchitis.
 Safety Disclaimer : RESEARCH SCREENING OUTPUT ONLY: Does not replace assessment by a clinician.
=================================================================
```

### 5.4 Generate Grad-CAM Explainability Heatmaps

Generate visual evidence showing where the model detected the anomaly in the breathing cycle:
```bash
python explain_gradcam.py --audio path/to/child_breath_sound.wav --task disease
```
Saves `<filename>_gradcam_explanation.png` showing:
1. Log Mel-Spectrogram of breath sound
2. Grad-CAM saliency heatmap highlighting the exact time-frequency bands
3. Temporal activation profile across the breath cycle

---

## 6. Registering the Model with Django Backend

Once training completes, the model artifact is saved directly to `ml/models/saved_models/`. You can register and activate it in Django using either:

### Option A: Using the `backend_sync.py` Bridge
```bash
python backend_sync.py --task disease
```

### Option B: Using Django's Management Command
```bash
python backend/manage.py register_disease_model \
  --model-version sprsound-disease-cnn-v1-fold0 \
  --artifact-path ml/models/saved_models/sprsound-disease-cnn-v1-fold0.keras \
  --classes "Healthy Control" Pneumonia "Wheezing Disorders" \
  --status ACTIVE \
  --architecture 3Block-ThroatDisease-2DCNN \
  --target-conv-layer last_conv_layer \
  --set-active --overwrite
```

Once registered and active:
- Backend readiness endpoint `GET /api/v1/ready/` will report `ml_inference: { status: "available", available: true }`.
- Inference endpoints `POST /api/v1/recordings/<id>/predict/` and `POST /api/v1/recordings/<id>/predict-disease/` will return real-time predictions without `503 Service Unavailable` errors!
