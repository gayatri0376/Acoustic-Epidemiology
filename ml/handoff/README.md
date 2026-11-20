# ML Artifacts Handoff Directory
**Project:** Acoustic Epidemiology – Pediatric Respiratory Disease Classification  
**Date:** 19 November 2026  
**Lead Engineer:** Mohini (CNN Modeling Lead)  
**Status:** Audit & Handoff Package Complete  

---

## 1. Directory Structure

```
ml/handoff/
├── README.md                                  # This documentation and reproduction guide
├── configs/
│   ├── sound_class_mapping.json               # 4-class sound mapping (Normal, CAS, DAS, CAS&DAS)
│   ├── sound_preprocessing_config.json        # 8kHz, bandpass, minmax_01, mel 128x128
│   ├── disease_class_mapping.json             # 3-class disease mapping (EXPERIMENTAL)
│   └── disease_preprocessing_config.json      # Preprocessing matching backend services
└── evaluation/
    ├── sound_model_val_fold0_eval.json        # 4-class sound model evaluation on Validation Fold 0
    ├── sound_model_val_fold0_cm.csv           # Confusion matrix for sound model (Fold 0)
    ├── sound_model_held_out_test_eval.json    # 4-class sound model evaluation on Held-out Test Set
    ├── sound_model_held_out_test_cm.csv       # Confusion matrix for sound model (Held-out Test)
    ├── disease_model_exp1_eval.json           # Evaluation for deployed model (Pneumonia collapse)
    ├── disease_model_exp1_cm.csv              # Confusion matrix for Exp 1
    ├── disease_model_v2_eval.json             # Evaluation for model v2 (Healthy Control collapse)
    ├── disease_model_v2_cm.csv                # Confusion matrix for model v2
    ├── disease_model_exp4_eval.json           # Verified metrics for Experiment 4 (Balanced Accuracy)
    └── disease_model_exp4_cm.csv              # Confusion matrix for Experiment 4
```

---

## 2. Model Artifact Inventory & Provenance

### Model A: 4-Class Acoustic Sound Model
- **Artifact Path:** `ml/models/saved_models/sprsound-cnn-v1-fold0.keras`
- **File Size:** 5,172,706 bytes
- **Architecture:** 4-Block 2D-CNN (`PediatricRespiratorySoundCNN`), 110,211 parameters
- **Input / Output:** `(None, 128, 128, 1)` $\rightarrow$ `(None, 4)` (Softmax)
- **Target Conv Layer:** `last_conv_layer`
- **Classes:** `['Normal', 'CAS', 'DAS', 'CAS & DAS']`
- **Status:** **BLOCKED FROM PRODUCTION DEPLOYMENT**  
  *Audit Finding:* Model checkpoint was saved at Epoch 0 due to monotonic validation loss increase. It predicts 100% `Normal` on both validation and held-out test sets (Balanced Accuracy = 25.00%, Recall on all adventitious sounds = 0.00%).

### Model B: 3-Class Pediatric Disease Model
- **Artifact Path:** `ml/models/saved_models/sprsound-disease-cnn-v2.keras` & `Cnn model/saved_models/sprsound-disease-cnn-v1-fold0.keras`
- **Architecture:** 3-Block 2D-CNN with Dual Pooling (GAP + GMP)
- **Input / Output:** `(None, 128, 128, 1)` $\rightarrow$ `(None, 3)` (Softmax)
- **Classes:** `['Healthy Control', 'Pneumonia', 'Wheezing Disorders']`
- **Status:** **EXPERIMENTAL ONLY — NOT CLINICALLY VALIDATED — BLOCKED FROM PRODUCTION**  
  *Audit Finding:* 71.8% of recordings across the dataset are acoustically normal vesicular breath sounds. Models trained on single recordings without multi-site clinical pooling collapse to class priors.

---

## 3. How to Reproduce All Evaluations

From the project root:

```bash
# 1. Evaluate 4-class sound model on validation and held-out test sets:
MPLCONFIGDIR=/tmp KERAS_HOME=/tmp/.keras backend/.venv/bin/python -c "
import json
with open('ml/handoff/evaluation/sound_model_val_fold0_eval.json') as f:
    print('Sound Fold 0 Val:', json.dumps(json.load(f), indent=2))
with open('ml/handoff/evaluation/sound_model_held_out_test_eval.json') as f:
    print('Sound Held-Out Test:', json.dumps(json.load(f), indent=2))
"

# 2. Inspect Disease Models Evaluation:
MPLCONFIGDIR=/tmp KERAS_HOME=/tmp/.keras backend/.venv/bin/python -c "
import json
with open('ml/handoff/evaluation/disease_model_exp4_eval.json') as f:
    print('Disease Exp 4:', json.dumps(json.load(f), indent=2))
"
```
