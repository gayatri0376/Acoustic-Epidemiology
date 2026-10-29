"""Service layer for machine-learning inference.

HONEST DEFAULT: no trained model ships with this backend, so nothing here invents a prediction.
Until ML_MODEL_PATH points to a real model AND _predict() is implemented, every analysis returns
status "not_configured" (or "failed"), with no label and no confidence.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from django.conf import settings


@dataclass
class InferenceResult:
    status: str                       # 'not_configured' | 'completed' | 'failed'
    label: Optional[str] = None
    confidence: Optional[float] = None
    message: str = ''


def is_configured() -> bool:
    path = getattr(settings, 'ML_MODEL_PATH', '')
    return bool(path) and Path(path).is_file()


def _predict(audio_path: str, model_path: str) -> Tuple[str, float]:
    """>>> YOUR MODEL GOES HERE <<<
    Load the model from `model_path`, run it on the audio file, and return (label, confidence 0..1)."""
    raise NotImplementedError('Model loading and prediction are not implemented yet.')


def run_inference(recording) -> InferenceResult:
    if not is_configured():
        return InferenceResult('not_configured', message='No trained model is configured. No prediction was made.')
    try:
        label, confidence = _predict(recording.audio_file.path, settings.ML_MODEL_PATH)
    except NotImplementedError:
        return InferenceResult('failed', message='A model file is set but _predict() is not implemented. No prediction was made.')
    except Exception:
        return InferenceResult('failed', message='Inference failed. No prediction was made.')
    # Validate the model output before trusting it.
    if not isinstance(label, str) or not label or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return InferenceResult('failed', message='The model returned an invalid result and it was discarded.')
    return InferenceResult('completed', label=label, confidence=float(confidence),
                           message='Research output only. Not clinically validated.')
