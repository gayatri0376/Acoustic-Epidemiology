# Full-Length Audio Handling & Event Preservation Report

**Project:** Acoustic Epidemiology – Pediatric Respiratory Sound Classification using SPRSound  
**Lead:** Gayatri (Backend & CNN Integration)  
**Task:** Task 5 — Investigate Full-Length Audio Handling  
**Date:** November 2026  
**Status:** Complete — Proposed Chunking & MIL Architecture Defined for ML Coordination  

---

## 1. Current Behavior and Defect Analysis

### 1.1. Exact Code Locations of Center-Cropping
Center-cropping is currently hardcoded in both the ML training pipeline and the backend inference service:
1. **ML Training Pipeline:**
   - `ml/preprocessing/respiratory_preprocessor.py:217–218`:
     ```python
     crop_start = (current_frames - target_frames) // 2
     return spectrogram[:, crop_start:crop_start + target_frames]
     ```
   - `ml/features/extract_features.py:231`:
     ```python
     features = preprocessor.extract_whole_record_features(record_path, fixed_frames=128)
     ```
2. **Backend Inference Service:**
   - `backend/ml_inference/services.py:244–246`:
     ```python
     # Center-crop time dimension
     crop_start = (current_w - target_w) // 2
     input_tensor = input_tensor[:, :, crop_start:crop_start + target_w, :]
     ```

### 1.2. Verification of Strategy Consistency
- **Verified:** Both pipelines use the **exact same** symmetric center-cropping strategy. While this ensures technical consistency between training and inference, it embeds a profound structural flaw into both.

### 1.3. CNN Architecture Input Constraint
- Inspection of `ml/models/saved_models/sprsound-cnn-v1-fold0.keras` and `sprsound-disease-cnn-v2.keras` confirms:
  - Input tensor shape: strictly `(None, 128, 128, 1)`.
  - The architectures utilize 2D convolutional blocks followed by a `Flatten()` layer connected to `Dense` layers.
  - Because of the `Flatten()` operation, the model **cannot accept arbitrary time widths** without reshaping weights or replacing Flatten with Global Average Pooling (GAP).

---

## 2. Impact on Clinical Event Preservation

- Native sampling rate: $F_s = 8,000\text{ Hz}$
- Hop size: $H = 512\text{ samples}$ ($64\text{ ms}$ per frame)
- Fixed 128 frames:
  $$T_{\text{fixed}} = 128 \times 0.064\text{ s} = 8.192\text{ seconds}$$
- SPRSound recordings range from $0.3\text{ s}$ to $15.36\text{ s}$ (mean: $10.84\text{ s}$).
- Previously reported dataset audit statistics:
  - **1,854 recordings (69.10%)** exceed 8.192 s.
  - **946 annotated events (14.21%)** are completely cropped out.
  - **1,346 annotated events (20.22%)** are partially clipped.
  - **34.43% of all annotated events** are mutilated or discarded.
- **Consequence:** In clinical auscultation, adventitious sounds frequently occur in peripheral breath cycles. When a patient coughs or wheezes in second 1–3 of a 14-second recording, center-cropping discards the wheeze, forcing the model to evaluate silent or normal lung sounds while training loss penalizes it for the recording's "Wheezing" label.

---

## 3. Recommended Event-Preserving Strategy: Overlapping Sliding-Window Chunking

To preserve 100% of the recording duration while retaining compatibility with the existing `(None, 128, 128, 1)` CNN input shape, we propose an **Overlapping Sliding-Window Chunking Strategy**.

```
Original Audio Recording (e.g. 15.36 s / 241 Frames)
├──────────────────────────────────────────────────────────────────────────────┤
[Chunk 0: Frames 0..128 (0.00s - 8.19s)]
                         [Chunk 1: Frames 64..192 (4.10s - 12.29s)]
                                                  [Chunk 2: Frames 113..241 (7.23s - 15.36s)]
                                                  
Each chunk is passed into CNN -> Predictions [P_0, P_1, P_2]
Aggregation (Noisy-OR / Max-Pooling / Attention-MIL) -> Overall Record Prediction
Chunk Timestamps preserved -> Event Localization & Explanations!
```

### 3.1. Proposed Window and Hop Parameters (Clearly Marked PROPOSED)
* **Window Size:** $W = 128\text{ frames}$ ($8.192\text{ seconds}$) — preserves exact input shape for current CNN.
* **Hop Size (Overlap):** $H_{\text{hop}} = 64\text{ frames}$ ($4.096\text{ seconds}$, 50% overlap).
* **Rationale for 50% Overlap:** Pediatric adventitious events (crackles, wheezes, stridor) have median durations between $0.15\text{ s}$ and $1.8\text{ s}$. With a $4.096\text{ s}$ hop size, **every respiratory sound event is guaranteed to fall entirely inside at least one window**, eliminating boundary clipping.
* **Tail Boundary Handling:** The final window is aligned to the exact end of the recording (`frame_start = total_frames - 128`) rather than zero-padding, ensuring genuine acoustic content throughout.

---

## 4. Aggregation Strategies: Chunk to Whole-Record

Three aggregation strategies are technically feasible:

1. **Max-Pooling Aggregation (Immediate / Heuristic):**
   $$P_{\text{record}}(\text{Class } c) = \max_{k \in \text{Chunks}} P_k(c)$$
   *Pros:* Requires zero model retraining; immediately deployable on backend with existing weights.  
   *Cons:* Can amplify false-positive noise in long recordings.

2. **Noisy-OR Probabilistic Aggregation:**
   $$P_{\text{record}}(\text{Class } c) = 1 - \prod_{k=1}^K (1 - P_k(c))$$
   *Pros:* Clinically intuitive (if any chunk contains wheezing, the probability of the patient having wheezing increases).

3. **Attention-Based Multi-Instance Learning (MIL) (Proposed for QH-MTLNet):**
   A lightweight gated attention mechanism learns instance weights $a_k$ to dynamically focus on chunks containing authentic adventitious events:
   $$z_{\text{record}} = \sum_{k=1}^K a_k h_k, \quad a_k = \frac{\exp(\mathbf{w}^T \tanh(\mathbf{V} h_k^T))}{\sum_{j} \exp(\mathbf{w}^T \tanh(\mathbf{V} h_j^T))}$$
   *Requires coordination with Mohini during QH-MTLNet model training.*

---

## 5. Required Implementation Steps

### 5.1. Training-Side Changes (For Mohini)
1. In `ml/features/extract_features.py`: Replace `extract_whole_record_features` with multi-chunk extraction saving chunks as bag-of-instances `(num_chunks, 128, 128, 1)`.
2. Map chunk timestamps to ground-truth event annotations to generate chunk-level supervised labels.
3. Add MIL pooling layer prior to classification output in QH-MTLNet.

### 5.2. Backend-Side Changes (Gayatri)
1. Add `extract_sliding_window_chunks(audio_path, window=128, hop=64)` to `AudioPreprocessor`.
2. Update `InferenceService.execute_prediction` to predict across all chunks in a batch:
   `raw_chunk_outputs = model.predict(chunk_tensor_batch)`
3. Attach chunk timestamps to `Prediction` metadata:
   ```json
   {
     "chunks": [
       {"chunk_id": 0, "start_time": 0.0, "end_time": 8.192, "probabilities": {...}},
       {"chunk_id": 1, "start_time": 4.096, "end_time": 12.288, "probabilities": {...}},
       {"chunk_id": 2, "start_time": 7.168, "end_time": 15.360, "probabilities": {...}}
     ]
   }
   ```
4. This enables temporal localization in the frontend without requiring separate onset/offset detector networks.

---

## 6. Required Tests

1. **Window Coverage Assertion:**
   Assert that for any recording length $L \in [0.3\text{ s}, 16.0\text{ s}]$, the union of intervals $\bigcup_k [t_{\text{start}}^{(k)}, t_{\text{end}}^{(k)}] = [0, L]$.
2. **Event Intersection Assertion:**
   For all 6,658 annotated events in SPRSound, assert that $\ge 99.5\%$ of events have an intersection-over-event ratio $\frac{|E \cap \text{Chunk}|}{|E|} = 1.0$ for at least one chunk.
3. **Batch Dimension Validation:**
   Assert that `AudioPreprocessor` produces a tensor of shape `(num_chunks, 128, 128, 1)` that matches model input expectations.

