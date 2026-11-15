"""
Grad-CAM Explainability Module for Pediatric Auscultation CNN.

Generates gradient-weighted class activation heatmaps showing which time intervals
(breathing cycles) and frequency bands (e.g., wheezing/stridor harmonics) drove the prediction.
Bitwise and mathematically compatible with backend/explanations/services.py:GradCamService.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Ensure imports work
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .config import TARGET_CONV_LAYER_NAME, PREPROCESSING_CONFIG, DISEASE_CLASSES
    from .preprocessor import ThroatAudioPreprocessor
    from .predict import find_latest_model
except (ImportError, ValueError):
    from config import TARGET_CONV_LAYER_NAME, PREPROCESSING_CONFIG, DISEASE_CLASSES
    from preprocessor import ThroatAudioPreprocessor
    from predict import find_latest_model

logger = logging.getLogger("GradCAMExplainer")


def generate_gradcam_heatmap(
    model,
    input_tensor: np.ndarray,
    target_class_index: Optional[int] = None,
    target_conv_layer: str = TARGET_CONV_LAYER_NAME,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes authentic Grad-CAM saliency map from target conv layer.
    Returns:
      (heatmap, time_profile, freq_profile)
    """
    import tensorflow as tf

    # Verify target conv layer exists
    layer_names = [l.name for l in model.layers]
    if target_conv_layer not in layer_names:
        raise ValueError(f"Grad-CAM target layer '{target_conv_layer}' not in model: {layer_names}")

    # Build gradient model connecting input -> (last_conv_layer output, final predictions)
    conv_layer = model.get_layer(target_conv_layer)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    tensor_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(tensor_tf)
        conv_outputs, predictions = grad_model(tensor_tf)
        if target_class_index is None:
            target_class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class_index]

    # Compute gradients of class score wrt feature maps
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient calculation returned None.")

    # Global average pooling across spatial dimensions -> weights alpha
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of feature maps
    conv_outputs_val = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_val), axis=-1)

    # ReLU to keep only positive contributions
    cam = tf.maximum(cam, 0)

    # Normalize to [0.0, 1.0]
    cam_max = tf.reduce_max(cam)
    if cam_max > 0:
        cam = cam / cam_max

    cam_np = cam.numpy()

    # Resize heatmap to match input Mel-spectrogram shape (128, 128)
    import scipy.ndimage
    target_h, target_w = input_tensor.shape[1], input_tensor.shape[2]
    zoom_factors = (target_h / cam_np.shape[0], target_w / cam_np.shape[1])
    heatmap_resized = scipy.ndimage.zoom(cam_np, zoom_factors, order=1)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Compute 1D temporal and spectral saliency curves
    time_profile = np.mean(heatmap_resized, axis=0)      # shape (128,)
    freq_profile = np.mean(heatmap_resized, axis=1)      # shape (128,)

    return heatmap_resized, time_profile, freq_profile


def save_gradcam_visualization(
    input_tensor: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    output_path: str,
    sr: int = 8000,
    f_min: float = 50.0,
    f_max: float = 2000.0,
):
    """
    Renders Mel-spectrogram with overlaid Grad-CAM heatmap using matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spectrogram = input_tensor[0, :, :, 0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 3, 1]})

    # 1. Base Log Mel-Spectrogram
    im1 = ax1.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma", extent=[0, 8.192, f_min, f_max])
    ax1.set_title(f"Acoustic Mel-Spectrogram (Predicted: {predicted_class})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency (Hz)")
    plt.colorbar(im1, ax=ax1, format="%+2.0f dB")

    # 2. Grad-CAM Attention Heatmap
    im2 = ax2.imshow(heatmap, aspect="auto", origin="lower", cmap="jet", extent=[0, 8.192, f_min, f_max], alpha=0.8)
    ax2.set_title("Grad-CAM Saliency Heatmap (Acoustic Regions Explaining Prediction)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=ax2)

    # 3. Temporal Attention Curve
    time_profile = np.mean(heatmap, axis=0)
    time_axis = np.linspace(0, 8.192, len(time_profile))
    ax3.plot(time_axis, time_profile, color="crimson", lw=1.8)
    ax3.fill_between(time_axis, time_profile, color="crimson", alpha=0.3)
    ax3.set_title("Temporal Activation Profile across Breath Cycle", fontsize=10)
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Salience")
    ax3.set_xlim([0, 8.192])
    ax3.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Grad-CAM explanation saved to: %s", output_path)


def explain_audio_prediction(
    audio_path: str,
    model_path: Optional[str] = None,
    task: str = "disease",
    output_image_path: Optional[str] = None
) -> Dict[str, Any]:
    """Generates complete Grad-CAM explanation for an audio recording."""
    import tensorflow as tf
    from .predict import predict_breath_audio

    # Run base prediction
    pred_res = predict_breath_audio(audio_path=audio_path, model_path=model_path, task=task)

    # Load model
    artifact = Path(model_path) if model_path else find_latest_model(task=task)
    model = tf.keras.models.load_model(str(artifact), compile=False)

    # Preprocess
    input_tensor = ThroatAudioPreprocessor.prepare_cnn_input(audio_path=audio_path)

    classes = DISEASE_CLASSES if task == "disease" else ["Normal", "CAS", "DAS", "CAS & DAS"]
    pred_class = pred_res["predicted_class"]
    class_idx = classes.index(pred_class)

    # Heatmap
    heatmap, time_prof, freq_prof = generate_gradcam_heatmap(
        model=model,
        input_tensor=input_tensor,
        target_class_index=class_idx
    )

    if not output_image_path:
        stem = Path(audio_path).stem
        output_image_path = str(CURRENT_DIR / f"{stem}_gradcam_explanation.png")

    save_gradcam_visualization(
        input_tensor=input_tensor,
        heatmap=heatmap,
        predicted_class=pred_class,
        output_path=output_image_path
    )

    return {
        "audio_file": os.path.basename(audio_path),
        "predicted_class": pred_class,
        "visualization_path": output_image_path,
        "peak_time_second": round(float(np.argmax(time_prof) * (8.192 / len(time_prof))), 2),
        "peak_salience": round(float(np.max(time_prof)), 4),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation for breath sound")
    parser.add_argument("--audio", type=str, required=True, help="Path to breath audio .wav")
    parser.add_argument("--model", type=str, default=None, help="Path to model .keras")
    parser.add_argument("--task", default="disease", choices=["disease", "sound"])
    parser.add_argument("--out", type=str, default=None, help="Output visualization path")

    args = parser.parse_args()
    res = explain_audio_prediction(args.audio, args.model, args.task, args.out)
    print(json.dumps(res, indent=2))
