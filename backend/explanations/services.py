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

            # 3. Authentic Grad-CAM Generation
            try:
                import tensorflow as tf
                import numpy as np
                from pathlib import Path
                from ml_inference.services import AudioPreprocessor

                # Construct grad model
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(last_conv_name).output, model.output]
                )

                # Preprocess input audio using model's configured parameters
                input_tensor = AudioPreprocessor.prepare_cnn_input(
                    audio_path=prediction.recording.audio_file.path,
                    config=model_version.preprocessing_config,
                    expected_shape=getattr(model, 'input_shape', None)
                )

                # Target class index
                verified_classes = model_version.verified_classes or []
                if prediction.predicted_class in verified_classes:
                    pred_index = verified_classes.index(prediction.predicted_class)
                else:
                    pred_index = 0

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(input_tensor)
                    loss = predictions[:, pred_index]

                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1).numpy()
                heatmap = np.maximum(heatmap, 0)
                max_val = np.max(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val

                # Structured explainability data
                time_profile = [round(float(v), 4) for v in np.mean(heatmap, axis=0)]
                freq_profile = [round(float(v), 4) for v in np.mean(heatmap, axis=1)]
                structured_data = {
                    "method": "Grad-CAM",
                    "conv_layer": last_conv_name,
                    "target_class": prediction.predicted_class,
                    "time_saliency_profile": time_profile,
                    "frequency_saliency_profile": freq_profile,
                    "interpretability_notice": (
                        "Grad-CAM highlights time-frequency regions in the Mel-Spectrogram that contributed "
                        "most strongly to the model's acoustic classification. It does NOT constitute a clinical diagnosis "
                        "or causal medical explanation."
                    )
                }

                # Save heatmap visualization
                media_dir = Path(settings.MEDIA_ROOT) / "explanations"
                media_dir.mkdir(parents=True, exist_ok=True)
                artifact_filename = f"{prediction.id}_gradcam.png"
                artifact_rel_path = f"media/explanations/{artifact_filename}"
                artifact_full_path = media_dir / artifact_filename

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(heatmap, cmap='inferno', aspect='auto', origin='lower')
                ax.set_title(f"Grad-CAM Saliency: {prediction.predicted_class}")
                ax.set_xlabel("Time Frames")
                ax.set_ylabel("Mel Frequency Bands")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(str(artifact_full_path), dpi=100)
                plt.close(fig)

                explanation = Explanation.objects.create(
                    prediction=prediction,
                    method=method,
                    status=ExplanationStatus.GENERATED,
                    artifact_path=artifact_rel_path,
                    structured_data=structured_data
                )
                return {
                    "success": True,
                    "explanation": explanation
                }

            except ImportError:
                explanation = Explanation.objects.create(
                    prediction=prediction,
                    method=method,
                    status=ExplanationStatus.UNSUPPORTED,
                    error_details="TensorFlow is not installed in the environment to execute gradient backpropagation."
                )
                return {
                    "success": False,
                    "code": "GRADIENT_MODEL_UNAVAILABLE",
                    "explanation": explanation,
                    "message": "TensorFlow gradient backpropagation requires TensorFlow to be installed."
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

