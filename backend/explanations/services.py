import os
import logging
from django.conf import settings
from explanations.models import Explanation, ExplanationMethod, ExplanationStatus
from ml_inference.services import ModelCache

logger = logging.getLogger(__name__)


class GradCamService:
    """
    Service responsible for authentic Gradient-weighted Class Activation Mapping (Grad-CAM)
    over 2D-CNN Time-Frequency Mel-Spectrograms.
    
    Guarantees:
    - Never fabricates or synthesizes fake heatmaps.
    - Requires a loaded, verified CNN model with accessible 2D Convolutional layers.
    - Accurately reports UNSUPPORTED when convolutional gradient hooks are unavailable.
    """

    @classmethod
    def find_last_conv_layer(cls, model):
        """
        Locates the name of the final Conv2D layer in a Keras/TensorFlow model topology.
        Returns None if no convolutional layers exist.
        """
        if not model or not hasattr(model, 'layers'):
            return None

        for layer in reversed(model.layers):
            layer_type = layer.__class__.__name__
            layer_name = getattr(layer, 'name', '').lower()
            if 'Conv' in layer_type or 'conv' in layer_name:
                return layer.name
        return None


    @classmethod
    def generate_explanation(cls, prediction, method=ExplanationMethod.GRAD_CAM):
        """
        Attempts authentic Grad-CAM extraction.
        If the model artifact is missing or does not have accessible convolutional gradients,
        creates an UNSUPPORTED explanation record and returns an error descriptor.
        """
        model_version = prediction.model_version

        # 1. Verify model version and artifact presence
        if not model_version or not model_version.is_artifact_available():
            explanation = Explanation.objects.create(
                prediction=prediction,
                method=method,
                status=ExplanationStatus.UNSUPPORTED,
                error_details=(
                    "Authentic Grad-CAM generation is unsupported: No active, trained CNN model artifact "
                    "with convolutional layers is deployed. The system does not fabricate saliency heatmaps."
                )
            )
            return {
                "success": False,
                "code": "GRADIENT_MODEL_UNAVAILABLE",
                "explanation": explanation,
                "message": (
                    "Grad-CAM requires access to the trained convolutional layers and weights of the CNN model. "
                    "Because no model artifact is currently deployed, authentic saliency maps cannot be computed."
                )
            }

        # 2. Verify model is loadable and contains Conv2D layers
        try:
            model = ModelCache.get_model(model_version)
            if not model:
                raise FileNotFoundError(f"Model artifact not found at: {model_version.artifact_path}")

            last_conv_name = cls.find_last_conv_layer(model)
            if not last_conv_name:
                explanation = Explanation.objects.create(
                    prediction=prediction,
                    method=method,
                    status=ExplanationStatus.UNSUPPORTED,
                    error_details=(
                        "Grad-CAM is incompatible with this model architecture: No 2D Convolutional (Conv2D) "
                        "layers were detected in the model topology."
                    )
                )
                return {
                    "success": False,
                    "code": "CONV_LAYER_NOT_FOUND",
                    "explanation": explanation,
                    "message": "The active model architecture lacks Conv2D layers required for Grad-CAM gradient backpropagation."
                }

            # 3. Authentic Grad-CAM Generation Placeholder
            # Once real CNN weights are mounted by the ML teammate:
            # - Construct grad model: tf.keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_name).output, model.output])
            # - Compute gradient of predicted class with respect to last_conv_name feature maps
            # - Pool gradients and calculate weighted activation map
            # - Render heatmap overlay over the audio's Mel-Spectrogram and persist to media/explanations/<uuid>.png
            explanation = Explanation.objects.create(
                prediction=prediction,
                method=method,
                status=ExplanationStatus.PENDING
            )
            return {
                "success": True,
                "explanation": explanation
            }

        except Exception as exc:
            logger.exception("Grad-CAM generation error: %s", exc)
            explanation = Explanation.objects.create(
                prediction=prediction,
                method=method,
                status=ExplanationStatus.FAILED,
                error_details=f"Grad-CAM execution failed: {str(exc)}"
            )
            return {
                "success": False,
                "code": "GENERATION_FAILED",
                "explanation": explanation,
                "message": str(exc)
            }
