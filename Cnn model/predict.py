"""
Inference Engine for Pediatric Respiratory & Throat Audio Predictions.

Submits a recording of a child's breathing sound and outputs throat/respiratory
disease or adventitious sound classifications matching the backend contract.

Usage:
  # Predict disease from child breath recording:
  python predict.py --audio sample_breath.wav --task disease

  # Predict sound category (crackles, wheezing, stridor, normal):
  python predict.py --audio sample_breath.wav --task sound

  # Test pipeline with synthesized pediatric breath sound:
  python predict.py --synthesize-test
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Ensure Cnn model and project root are importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .config import (
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        PREPROCESSING_CONFIG,
        BACKEND_SAVED_MODELS_DIR,
        LOCAL_SAVED_MODELS_DIR,
    )
    from .preprocessor import ThroatAudioPreprocessor
except (ImportError, ValueError):
    from config import (
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        PREPROCESSING_CONFIG,
        BACKEND_SAVED_MODELS_DIR,
        LOCAL_SAVED_MODELS_DIR,
    )
    from preprocessor import ThroatAudioPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ThroatPredictor")


DISEASE_CLINICAL_INTERPRETATION = {
    "Healthy Control": (
        "Baseline clear acoustic profile consistent with control group pediatric subjects. "
        "No continuous airway sounds (wheezes) or discontinuous sounds (crackles) detected."
    ),
    "Pneumonia": (
        "Consolidation-type acoustic patterns detected (Discontinuous Adventitious Sounds / Crackles). "
        "Indicates potential lower respiratory tract alveolar involvement or fluid accumulation."
    ),
    "Wheezing Disorders": (
        "High-pitched airway narrowing detected (Continuous Adventitious Sounds / Wheezes). "
        "Associated with pediatric reactive airway disease, asthma, bronchitis, or airway constriction."
    )
}

RESEARCH_DISCLAIMER = (
    "RESEARCH SCREENING OUTPUT ONLY: This model is trained on the SPRSound pediatric "
    "auscultation dataset. It is not an FDA-cleared diagnostic tool and must not be "
    "substituted for assessment by a licensed pediatric clinician."
)


def find_latest_model(task: str = "disease") -> Optional[Path]:
    """Finds the most recent trained model artifact in local or backend model directories."""
    search_dirs = [LOCAL_SAVED_MODELS_DIR, BACKEND_SAVED_MODELS_DIR]
    pattern = f"*disease*.keras" if task == "disease" else f"*sound*.keras"

    found_models = []
    for d in search_dirs:
        if d.exists():
            found_models.extend(list(d.glob(pattern)))
            found_models.extend(list(d.glob("*.keras")))
            found_models.extend(list(d.glob("*.h5")))

    if not found_models:
        return None

    # Return newest by modification time
    found_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found_models[0]


def predict_breath_audio(
    audio_path: str,
    model_path: Optional[str] = None,
    task: str = "disease"
) -> Dict[str, Any]:
    """
    Submits a child's breathing audio file and performs standardized CNN inference.
    Matches backend/ml_inference/services.py:InferenceService and DiseaseInferenceService.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine model artifact
    if model_path:
        artifact = Path(model_path)
    else:
        artifact = find_latest_model(task=task)

    if not artifact or not artifact.exists():
        raise FileNotFoundError(
            f"No trained CNN model artifact found for task '{task}'.\n"
            "Train a model first with: python train.py --task " + task
        )

    # Verified classes
    classes = DISEASE_CLASSES if task == "disease" else SOUND_CLASSES_4CLASS

    # 1. Load trained CNN model
    try:
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        model = tf.keras.models.load_model(str(artifact), compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {artifact}: {e}")

    # 2. Preprocess audio into (1, 128, 128, 1) tensor
    start_time = time.perf_counter()
    input_tensor = ThroatAudioPreprocessor.prepare_cnn_input(
        audio_path=audio_path,
        config=PREPROCESSING_CONFIG,
        expected_shape=(128, 128, 1)
    )

    # 3. Execute CNN Forward Pass
    raw_output = model.predict(input_tensor, verbose=0)
    inference_duration_ms = int((time.perf_counter() - start_time) * 1000)

    # 4. Extract probabilities and predicted class
    output_probs = raw_output[0].tolist() if hasattr(raw_output[0], 'tolist') else list(raw_output[0])

    if len(output_probs) != len(classes):
        raise ValueError(
            f"Model output dimension ({len(output_probs)}) != verified classes ({len(classes)})."
        )

    prob_map = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, output_probs)
    }
    predicted_class = max(prob_map, key=prob_map.get)
    confidence_pct = round(prob_map[predicted_class] * 100.0, 2)

    clinical_note = DISEASE_CLINICAL_INTERPRETATION.get(
        predicted_class, "Adventitious respiratory sound detected."
    )

    result = {
        "status": "SUCCESS",
        "audio_file": os.path.basename(audio_path),
        "task": task,
        "model_artifact": str(artifact.name),
        "predicted_class": predicted_class,
        "confidence_percentage": confidence_pct,
        "probability_scores": prob_map,
        "clinical_interpretation": clinical_note,
        "disclaimer": RESEARCH_DISCLAIMER,
        "inference_duration_ms": inference_duration_ms,
        "input_tensor_shape": list(input_tensor.shape),
    }

    return result


def create_synthetic_test_audio(output_path: str, duration_s: float = 8.192, sr: int = 8000) -> str:
    """Creates a synthetic pediatric breathing sound with wheeze harmonic for pipeline testing."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Vesicular breath base noise + respiratory envelope
    breath_cycle = np.sin(2 * np.pi * 0.4 * t) ** 2
    pink_noise = np.random.normal(0, 0.05, len(t))
    # Add a continuous adventitious wheeze / throat obstruction harmonic at 450 Hz
    wheeze = 0.15 * np.sin(2 * np.pi * 450.0 * t) * breath_cycle
    audio = ((pink_noise * breath_cycle + wheeze) * 32767).clip(-32768, 32767).astype(np.int16)

    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, sr, audio)
    except ImportError:
        import wave
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())

    return output_path


def print_prediction_report(res: Dict[str, Any]):
    """Pretty prints the prediction report."""
    print("\n" + "=" * 65)
    print("      PEDIATRIC RESPIRATORY & THROAT CNN PREDICTION REPORT")
    print("=" * 65)
    print(f" Audio File        : {res['audio_file']}")
    print(f" Model Artifact    : {res['model_artifact']}")
    print(f" Task Evaluated    : {res['task'].upper()}")
    print(f" Inference Latency : {res['inference_duration_ms']} ms")
    print("-" * 65)
    print(f" PREDICTED CLASS   : >>> {res['predicted_class'].upper()} <<<")
    print(f" CONFIDENCE        : {res['confidence_percentage']}%")
    print("-" * 65)
    print(" Probability Distribution:")
    for cls_name, prob in res["probability_scores"].items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"   {cls_name:20s}: [{bar}] {prob * 100:6.2f}%")
    print("-" * 65)
    print(f" Clinical Context  : {res['clinical_interpretation']}")
    print(f" Safety Disclaimer : {res['disclaimer']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify child breath audio using trained CNN")
    parser.add_argument("--audio", type=str, help="Path to child breath sound .wav file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .keras model")
    parser.add_argument("--task", choices=["disease", "sound"], default="disease", help="Target task: 'disease' or 'sound'")
    parser.add_argument("--synthesize-test", action="store_true", help="Generate and test synthetic audio")

    args = parser.parse_args()

    if args.synthesize-test or not args.audio:
        test_audio_path = str(CURRENT_DIR / "synthetic_test_breath.wav")
        logger.info("Creating synthetic breath sound: %s", test_audio_path)
        create_synthetic_test_audio(test_audio_path)
        audio_file = test_audio_path
    else:
        audio_file = args.audio

    try:
        prediction = predict_breath_audio(
            audio_path=audio_file,
            model_path=args.model,
            task=args.task
        )
        print_prediction_report(prediction)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        sys.exit(1)
