"""
Disease Inference Service for Acoustic Epidemiology.

Provides a fully separate inference pathway for the SPRSound 3-class disease model
(Model B), keeping it strictly decoupled from the existing sound-classification
service (Model A).

Dataset provenance:
  SPRSound (SJTU Paediatric Respiratory Sound Database) CC BY 4.0
  Patient summaries with clinical hospital admission/discharge diagnoses.
  100% pediatric population (age range: 0–16.2 years, mean 5.8 years).

Disease classes (from SPRSound patient-summary labels):
  0: Healthy Control   — SPRSound Control Group patients
  1: Pneumonia         — non-severe + severe Pneumonia merged
  2: Wheezing Disorders — Asthma + Bronchitis merged (both produce wheezing;
                          dataset does NOT support separating them acoustically)

IMPORTANT MEDICAL SAFETY NOTE:
  Results from this service are a RESEARCH SCREENING OUTPUT only.
  They must NOT be interpreted as or substituted for clinical diagnosis.
  The model was trained on patient-level (not recording-level) labels,
  which introduces inherent label noise at auscultation-site level.
  A patient's clinical diagnosis applies system-wide; individual
  auscultation sites may present as acoustically normal.

  Do not modify this behaviour or suppress the disclaimer.
"""

import os
import time
import logging
import threading
import numpy as np
from django.conf import settings
from rest_framework.exceptions import APIException
from rest_framework import status

from ml_inference.models import (
    ModelVersion,
    ModelTaskType,
    ModelStatus,
    DiseasePrediction,
    PredictionStatus,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Dataset and model constants
# ------------------------------------------------------------------
DISEASE_CLASSES = ["Healthy Control", "Pneumonia", "Wheezing Disorders"]

DISEASE_CLASS_DESCRIPTIONS = {
    "Healthy Control": (
        "The model's highest-probability class is Healthy Control, indicating patterns "
        "consistent with the control-group patients in the SPRSound dataset. "
        "These patients had no confirmed acute respiratory diagnosis at admission."
    ),
    "Pneumonia": (
        "The model's highest-probability class is Pneumonia. This class combines "
        "non-severe and severe pneumonia from the SPRSound patient summaries. "
        "Acoustic features associated with pneumonia include consolidation-related "
        "crackles (DAS). This is a research screening output, not a diagnosis."
    ),
    "Wheezing Disorders": (
        "The model's highest-probability class is Wheezing Disorders. This class "
        "combines Asthma and Bronchitis patients from the SPRSound dataset, because "
        "both conditions produce wheezing (CAS) and the dataset does NOT contain "
        "sufficient evidence to acoustically distinguish them. "
        "Do NOT interpret this as a specific asthma or bronchitis diagnosis."
    ),
}

RESEARCH_DISCLAIMER = (
    "This result is a research screening output from a CNN trained on the SPRSound "
    "pediatric auscultation dataset (CC BY 4.0). It does not constitute a medical "
    "diagnosis and must not replace assessment by a qualified healthcare professional. "
    "The model predicts against 3 patient-level classes (Healthy Control, Pneumonia, "
    "Wheezing Disorders) derived from hospital admission summaries, not recording-level "
    "acoustic labels. Individual auscultation sites may present as acoustically normal "
    "even when a systemic disease is present."
)


# ------------------------------------------------------------------
# Thread-safe model cache (separate from sound-classification cache)
# ------------------------------------------------------------------
class DiseaseModelCache:
    """
    Thread-safe in-memory cache for the disease CNN model.
    Maintained independently from the sound-classification ModelCache to avoid
    any interference between the two models.
    """
    _lock = threading.Lock()
    _cached_model = None
    _cached_version_id = None
    _cached_mtime = None

    @classmethod
    def get_model(cls, model_version):
        artifact_full_path = cls.resolve_artifact_path(model_version.artifact_path)
        if not os.path.exists(artifact_full_path):
            cls.clear()
            return None

        current_mtime = os.path.getmtime(artifact_full_path)

        with cls._lock:
            if (
                cls._cached_model is not None
                and cls._cached_version_id == model_version.id
                and cls._cached_mtime == current_mtime
            ):
                return cls._cached_model

            try:
                import tensorflow as tf
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                loaded = tf.keras.models.load_model(artifact_full_path, compile=False)
                cls._cached_model = loaded
                cls._cached_version_id = model_version.id
                cls._cached_mtime = current_mtime
                logger.info(
                    "Disease model %s loaded from %s",
                    model_version.version,
                    artifact_full_path,
                )
                return cls._cached_model
            except Exception as exc:
                cls.clear()
                logger.error(
                    "Failed to load disease model artifact from %s: %s",
                    artifact_full_path,
                    exc,
                )
                raise

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._cached_model = None
            cls._cached_version_id = None
            cls._cached_mtime = None

    @staticmethod
    def resolve_artifact_path(path_str):
        if not path_str:
            return ""
        if os.path.isabs(path_str):
            return path_str
        project_root = settings.BASE_DIR.parent
        return os.path.join(project_root, path_str)


# ------------------------------------------------------------------
# Disease Inference Service
# ------------------------------------------------------------------
class DiseaseInferenceService:
    """
    Clean inference orchestration interface for the disease classification model.
    Completely separate from InferenceService to preserve independent model lifecycles.

    Guarantees:
    - Uses only the DISEASE_CLASSIFICATION model version.
    - Returns genuine model outputs; never fabricates predictions.
    - Returns transparent HTTP 503 when no disease model is deployed.
    - Reuses the same AudioPreprocessor as the sound model (identical preprocessing spec).
    """

    @staticmethod
    def get_active_disease_model():
        """
        Retrieves the active DISEASE_CLASSIFICATION model version if and only if:
        1. task_type == DISEASE_CLASSIFICATION
        2. status == ACTIVE
        3. Artifact physically exists on disk
        4. verified_classes is non-empty
        """
        active_models = ModelVersion.objects.filter(
            status=ModelStatus.ACTIVE,
            task_type=ModelTaskType.DISEASE_CLASSIFICATION,
        )
        for model in active_models:
            if model.is_artifact_available() and model.verified_classes:
                return model
        return None

    @classmethod
    def validate_model_output(cls, raw_output, verified_classes):
        """
        Validates output shape and extracts normalized probability map.
        Identical logic to InferenceService.validate_model_output.
        """
        if raw_output is None or len(raw_output) == 0:
            raise ValueError("Disease model produced empty output.")

        output_probs = raw_output[0].tolist() if hasattr(raw_output[0], 'tolist') else list(raw_output[0])

        if len(output_probs) != len(verified_classes):
            raise ValueError(
                f"Disease model output dimension ({len(output_probs)}) does not match "
                f"verified disease class count ({len(verified_classes)})."
            )

        if any(np.isnan(p) or np.isinf(p) for p in output_probs):
            raise ValueError("Disease model output contains NaN or infinite values.")

        prob_map = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(verified_classes, output_probs)
        }
        predicted_disease = max(prob_map, key=prob_map.get)

        return predicted_disease, prob_map

    @classmethod
    def execute_disease_prediction(cls, recording, request_user=None):
        """
        Executes disease-model prediction on an authorized recording.

        Returns a DiseasePrediction instance.
        Raises ServiceUnavailable if no disease model is deployed.
        Raises APIException on inference or file-access errors.

        Does NOT modify or interfere with the sound-classification Prediction table.
        """
        from ml_inference.services import AudioPreprocessor

        active_model = cls.get_active_disease_model()

        if not active_model:
            unavailable = DiseasePrediction.objects.create(
                recording=recording,
                model_version=None,
                prediction_status=PredictionStatus.UNAVAILABLE,
                error_details=(
                    "Disease inference unavailable: No verified DISEASE_CLASSIFICATION CNN "
                    "model artifact is currently deployed. "
                    "The system does not return fabricated predictions."
                )
            )
            from rest_framework.exceptions import APIException as RFAPIException
            raise RFAPIException(
                detail={
                    "error": "Disease ML Inference Service Unavailable",
                    "code": "DISEASE_MODEL_NOT_DEPLOYED",
                    "disease_prediction_id": str(unavailable.id),
                    "recording_id": str(recording.id),
                    "message": (
                        "No trained disease-classification CNN model artifact is currently "
                        "available in the registry. A model must be registered with "
                        "task_type=DISEASE_CLASSIFICATION and status=ACTIVE. "
                        "The system never fabricates disease predictions."
                    ),
                }
            )

        if not recording.audio_file or not os.path.exists(recording.audio_file.path):
            dp = DiseasePrediction.objects.create(
                recording=recording,
                model_version=active_model,
                prediction_status=PredictionStatus.FAILED,
                error_details="Recording audio file is missing or unreadable for disease inference."
            )
            from rest_framework.exceptions import APIException as RFAPIException
            raise RFAPIException("The physical audio file could not be accessed for disease inference.")

        start_time = time.perf_counter()

        try:
            loaded_model = DiseaseModelCache.get_model(active_model)
            if loaded_model is None:
                raise FileNotFoundError(
                    f"Disease model artifact not found: {active_model.artifact_path}"
                )

            expected_shape = getattr(loaded_model, 'input_shape', None)

            input_tensor = AudioPreprocessor.prepare_cnn_input(
                audio_path=recording.audio_file.path,
                config=active_model.preprocessing_config,
                expected_shape=expected_shape,
            )

            raw_output = loaded_model.predict(input_tensor, verbose=0)

            verified_classes = active_model.verified_classes
            predicted_disease, prob_map = cls.validate_model_output(raw_output, verified_classes)

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            dp = DiseasePrediction.objects.create(
                recording=recording,
                model_version=active_model,
                predicted_disease=predicted_disease,
                probability_scores=prob_map,
                prediction_status=PredictionStatus.SUCCESS,
                inference_duration_ms=duration_ms,
            )
            return dp

        except Exception as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.exception("Disease inference failed: %s", exc)
            dp = DiseasePrediction.objects.create(
                recording=recording,
                model_version=active_model,
                prediction_status=PredictionStatus.FAILED,
                error_details=f"Disease inference error: {str(exc)}",
                inference_duration_ms=duration_ms,
            )
            from rest_framework.exceptions import APIException as RFAPIException
            raise RFAPIException(f"Disease model inference failed: {str(exc)}")

