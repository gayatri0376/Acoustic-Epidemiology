# TRANSFER LEARNING FEASIBILITY REPORT FOR PEDIATRIC RESPIRATORY SOUND CLASSIFICATION
**Project:** Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases  
**Date:** 19 November 2026  
**Role:** Mohini (Machine Learning Modeling & CNN Lead)  
**Status:** COMPLETE & HONEST TECHNICAL FEASIBILITY ASSESSMENT  

---

## 1. Executive Summary

This report evaluates the technical, domain, and environmental feasibility of applying transfer learning to the four-class pediatric respiratory sound classification task (`SPRSOUND_RECORD_4CLASS`: Normal, CAS, DAS, CAS & DAS).

### Feasibility Conclusion: **PARTIAL / CURRENTLY INFEASIBLE OFFLINE**
While transfer learning from pre-trained audio spectrogram backbones represents a viable long-term research direction, **it is not immediately deployable within the active development environment**:
1. **Zero Pre-trained Checkpoints on Disk:** An audit of the repository, environment virtual packages, and local caches confirmed that no pre-trained weights (ICBHI 2017 checkpoints, AudioSet/PANNs, Audio Spectrogram Transformers, YAMNet, or ImageNet backbones) exist on disk.
2. **Offline Sandboxed Execution:** External network requests to download remote checkpoints (e.g., from Hugging Face Hub, TorchHub, or Google Cloud Storage) are blocked under the standard project sandbox security policy.
3. **Severe Cross-Domain Mismatches with External Respiratory Datasets (e.g., ICBHI 2017):** Granularity (cycle-level vs. recording-level), pediatric chest wall acoustics, and sampling rates preclude direct zero-shot weight transfer without structural adaptation.

---

## 2. Technical Comparison of Potential Transfer Learning Approaches

Three major transfer learning paradigms were analyzed for compatibility with SPRSound and local computational constraints:

| Approach | Representative Architectures | Pretraining Domain & Data | Compute & Hardware Requirements | Architectural & Input Compatibility | Local Feasibility |
|---|---|---|---|---|---|
| **Approach 1: Respiratory-Specific Pretrained CNNs** | ResNet-18 / ResNeXt trained on ICBHI 2017 | Respiratory auscultation cycles (~6,898 breathing cycles from 126 patients) | Low compute (trainable on standard CPU in <15 minutes) | Moderate: ICBHI cycles are ~2.5 s; SPRSound recordings are 8.192 s (128 frames @ 8 kHz). Input channels match. | **INFEASIBLE OFFLINE:** No ICBHI pretrained weights exist locally. |
| **Approach 2: Pretrained Audio Spectrogram Backbones** | PANNs (CNN14), YAMNet (MobileNet-based), AST | Large-scale environmental & general audio (AudioSet: 2 million 10-second YouTube clips, 527 classes) | Moderate to High: CNN14 has ~80M parameters; AST has ~87M parameters (requires GPU for acceptable latency). YAMNet is lightweight (~3.7M params). | Moderate: AudioSet models expect 16 kHz or 32 kHz inputs with 64 or 128 Mel bins. SPRSound is standardized to 8 kHz. | **INFEASIBLE OFFLINE:** Weights are not cached locally and cannot be downloaded without internet access. |
| **Approach 3: Computer Vision ImageNet Transfer** | MobileNetV2, ResNet50V2 | Natural images (ImageNet-1K: 1.28M natural color photos) | Low to Moderate: MobileNetV2 has 3.5M parameters (runs well on CPU). | Low: Mel-Spectrograms are single-channel spectro-temporal power distributions, not RGB spatial images. Requires 3-channel repeating. | **PARTIAL:** Keras Application code exists in `.venv`, but pre-trained `.h5` weights are missing from disk. Initializing with `weights='imagenet'` triggers an HTTPS download that fails. |

---

## 3. Dataset Compatibility: ICBHI 2017 vs. SPRSound

The ICBHI 2017 Scientific Challenge database is the most widely cited external respiratory benchmark. However, evaluating cross-dataset transfer reveals deep methodological divergences:

### 3.1 Parameter & Domain Divergence Table

| Attribute | ICBHI 2017 Respiratory Database | SPRSound (SJTU Pediatric Database) | Cross-Dataset Compatibility Analysis |
|---|---|---|---|
| **Target Population** | Predominantly adults and elderly (median age ~66 years, range 5.5 to 78) | **100% Pediatric** (median age ~2.8 years, range 0 to 16.2 years) | **Severe Domain Shift:** Infant and toddler airways have significantly narrower diameters, higher respiratory rates (30–60 bpm vs. 12–20 bpm), and thinner chest walls. High-frequency wheezes and stridor resonate 400–800 Hz higher in infants than adults. |
| **Unit of Analysis** | **Cycle-Level Segments:** Annotations demarcate individual inspiration/expiration intervals (mean duration: 2.7 seconds). | **Recording-Level Full Auscultations:** 8 to 15 seconds (mean duration: 10.84 seconds), capturing 4 to 12 breathing cycles. | **Scale Incompatibility:** An ICBHI cycle model expects a single breath cycle. Applying it directly to an unsegmented 128-frame multi-cycle SPRSound recording distorts temporal features. |
| **Sampling Rate** | Highly variable across acquisition devices: 4 kHz, 10 kHz, 44.1 kHz. | Standardized: **8,000 Hz** (Nyquist 4 kHz). | **Resampling Required:** Bandwidth above 4 kHz in ICBHI is discarded when projecting to 8 kHz. |
| **Acoustic Stethoscope Hardware** | 4 distinct acquisition devices: WelchAllyn Meditron, 3M Littmann, Meditron master, custom microphone. | 3 digital stethoscopes with unified sensor specs. | **Sensor Bias:** Sensor frequency response differences can create device-specific artifacts unless normalized. |
| **Class Label Definitions** | 4 cycle classes: `[Normal, Wheeze, Crackle, Both]`. | 4 record classes: `[Normal, CAS, DAS, CAS & DAS]`. | **Semantic Alignment: EXCELLENT.** Normal $\leftrightarrow$ Normal; Wheeze $\leftrightarrow$ CAS; Crackle $\leftrightarrow$ DAS; Both $\leftrightarrow$ CAS & DAS. |

### 3.2 Can ICBHI Class Mappings be Reused?
**Semantically YES, Structurally NO.**  
While the acoustic taxonomies (Continuous vs. Discontinuous Adventitious Sounds) are identical, ICBHI annotations are localized to single breath cycles. In SPRSound, a recording labeled `CAS` may contain 6 breath cycles where only 1 cycle contains a faint wheeze. Without cycle-level bounding boxes on SPRSound, a model trained on isolated ICBHI wheeze cycles will classify the remaining 5 normal cycles as false negatives.

---

## 4. Transfer Learning Strategy: Frozen Feature Extractor vs. Fine-Tuning

If external weights are imported in future research:
1. **Frozen Feature Extractor (Linear Probe):**
   - **Unsuitable.** Because the domain shift from natural images (ImageNet) or adult breathing (ICBHI) to pediatric auscultation is substantial, frozen early convolutional filters (oriented edge detectors or adult harmonic templates) fail to capture pediatric high-frequency adventitious features.
2. **Full Fine-Tuning with Gradual Unfreezing:**
   - **Recommended.** Pre-train on adult respiratory sounds (ICBHI) $\rightarrow$ Freeze lower conv layers $\rightarrow$ Train classification head on SPRSound with low learning rate ($10^{-4}$) $\rightarrow$ Unfreeze all layers with discriminative learning rates ($10^{-5}$ for backbone, $10^{-4}$ for head).

---

## 5. Decision & Selected Alternative Baseline

Because no external weights exist on disk and network isolation prohibits downloading remote checkpoints, **fabricating fake pretrained models was strictly rejected**.

### Selected Defensive Strategy:
We developed a **Domain-Specific 4-Block 2D-CNN Baseline (`PediatricAcousticBaselineCNN`)** built natively for SPRSound:
- Input: `(128, 128, 1)` Log Mel-Spectrogram (8 kHz, 50–2000 Hz zero-phase Butterworth filter, min-max normalized).
- Dual Pooling (Global Average Pooling + Global Max Pooling) to simultaneously detect continuous harmonic patterns (wheezes) and sharp impulse bursts (crackles).
- Targeted training-only SpecAugment to balance minority adventitious classes.
- **Validation Balanced Accuracy checkpoint selection** to reject single-class majority collapse.
