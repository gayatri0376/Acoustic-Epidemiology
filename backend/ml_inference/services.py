import os
import time
import logging
from django.conf import settings
from rest_framework.exceptions import APIException
from rest_framework import status
from ml_inference.models import ModelVersion, ModelStatus, Prediction, PredictionStatus

logger = logging.getLogger(__name__)


class ServiceUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = 'The ML inference service is currently unavailable.'
    default_code = 'service_unavailable'


class InferenceService:
    """
    Clean inference orchestration interface.
    Adheres strictly to verified model registries and prevents mock prediction fabrication.
    """

    @staticmethod
    def get_active_model():
        """
        Retrieves the active model version if and only if:
        1. It is marked ACTIVE in the database
        2. Its artifact physically exists on disk and has verified classes
        """
        active_models = ModelVersion.objects.filter(status=ModelStatus.ACTIVE)
        for model in active_models:
            if model.is_artifact_available() and model.verified_classes:
                return model
        return None

    @classmethod
    def execute_prediction(cls, recording, request_user=None):
        """
        Executes prediction on an authenticated recording.
        If real model artifact is missing, records UNAVAILABLE state and raises ServiceUnavailable.
        """
        active_model = cls.get_active_model()

        if not active_model:
            # Create a clear, transparent record that prediction could not be run
            unavailable_prediction = Prediction.objects.create(
                recording=recording,
                model_version=None,
                prediction_status=PredictionStatus.UNAVAILABLE,
                error_details=(
                    "Inference unavailable: No verified CNN model artifact is currently deployed or active. "
                    "The system does not return fabricated predictions."
                )
            )
            raise ServiceUnavailable(
                detail={
                    "error": "ML Inference Service Unavailable",
                    "code": "MODEL_NOT_DEPLOYED",
                    "prediction_id": str(unavailable_prediction.id),
                    "recording_id": str(recording.id),
                    "message": (
                        "No trained convolutional neural network model artifact is currently available in "
                        "the model registry. To perform real-time classification, a trained CNN artifact "
                        "and verified label configuration must be registered and placed in ml/models/."
                    )
                }
            )

        # A real active model is registered: verify audio file exists
        if not recording.audio_file or not os.path.exists(recording.audio_file.path):
            prediction = Prediction.objects.create(
                recording=recording,
                model_version=active_model,
                prediction_status=PredictionStatus.FAILED,
                error_details="Recording audio file is missing or unreadable on server storage."
            )
            raise APIException("The physical audio file could not be accessed for inference.")

        start_time = time.perf_counter()

        try:
            # Here we integrate with the actual model loader
            # Import tensorflow only if an artifact is actually present
            import tensorflow as tf
            import numpy as np

            # Load actual model
            artifact_full_path = active_model.artifact_path
            if not os.path.isabs(artifact_full_path):
                artifact_full_path = os.path.join(settings.BASE_DIR.parent, artifact_full_path)

            loaded_model = tf.keras.models.load_model(artifact_full_path)

            # Preprocessing according to verified config
            import soundfile as sf
            audio_data, sr = sf.read(recording.audio_file.path)
            
            # Forward pass
            # Reshape input matching model input shape
            input_tensor = np.expand_dims(audio_data, axis=0)
            raw_output = loaded_model.predict(input_tensor)

            output_probs = raw_output[0].tolist()
            verified_classes = active_model.verified_classes

            if len(output_probs) != len(verified_classes):
                raise ValueError(
                    f"Model output dimension ({len(output_probs)}) does not match verified class count ({len(verified_classes)})."
                )

            # Build probability map
            prob_map = {
                cls_name: round(float(prob), 4) 
                for cls_name, prob in zip(verified_classes, output_probs)
            }
            predicted_class = max(prob_map, key=prob_map.get)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            prediction = Prediction.objects.create(
                recording=recording,
                model_version=active_model,
                predicted_class=predicted_class,
                probability_scores=prob_map,
                prediction_status=PredictionStatus.SUCCESS,
                inference_duration_ms=duration_ms
            )
            return prediction

        except Exception as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.exception("Inference execution failed: %s", exc)
            prediction = Prediction.objects.create(
                recording=recording,
                model_version=active_model,
                prediction_status=PredictionStatus.FAILED,
                error_details=f"Inference error: {str(exc)}",
                inference_duration_ms=duration_ms
            )
            raise APIException(f"Model inference failed: {str(exc)}")
