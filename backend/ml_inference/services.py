import os
import time
import logging
import threading
import numpy as np
from django.conf import settings
from rest_framework.exceptions import APIException
from rest_framework import status
from ml_inference.models import ModelVersion, ModelStatus, Prediction, PredictionStatus

logger = logging.getLogger(__name__)


class ServiceUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = 'The ML inference service is currently unavailable.'
    default_code = 'service_unavailable'


class ModelCache:
    """
    Thread-safe in-memory cache for the loaded deep learning model.
    Avoids re-loading model weights from disk on every HTTP request.
    """
    _lock = threading.Lock()
    _cached_model = None
    _cached_version_id = None
    _cached_mtime = None

    @classmethod
    def get_model(cls, model_version):
        """
        Retrieves the cached model if it matches model_version and disk mtime;
        otherwise loads the artifact and updates the cache.
        """
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

            # Load the model using TensorFlow/Keras
            try:
                import tensorflow as tf
                # Disable excessive TensorFlow warning logs
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                loaded = tf.keras.models.load_model(artifact_full_path, compile=False)
                cls._cached_model = loaded
                cls._cached_version_id = model_version.id
                cls._cached_mtime = current_mtime
                logger.info(
                    "Model %s loaded successfully from %s", 
                    model_version.version, 
                    artifact_full_path
                )
                return cls._cached_model
            except Exception as exc:
                cls.clear()
                logger.error("Failed to load model artifact from %s: %s", artifact_full_path, exc)
                raise

    @classmethod
    def clear(cls):
        """Clears the cached model instance."""
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


class AudioPreprocessor:
    """
    Standardized acoustic preprocessing pipeline.
    Transforms raw audio into time-frequency representations (Mel-Spectrograms)
    according to verified model configuration.
    """

    @classmethod
    def load_audio(cls, file_path, target_sr=None):
        """
        Loads audio file safely using soundfile, converting to mono if stereo.
        """
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype='float32')

        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if target_sr is specified and different from native sample rate
        if target_sr and target_sr != sr:
            try:
                import scipy.signal
                num_samples = int(len(audio) * float(target_sr) / sr)
                audio = scipy.signal.resample(audio, num_samples)
                sr = target_sr
            except Exception as exc:
                logger.warning("Scipy resampling failed, keeping original sample rate: %s", exc)

        return audio, sr

    @classmethod
    def compute_mel_spectrogram(
        cls, 
        audio, 
        sr=16000, 
        n_mels=128, 
        n_fft=1024, 
        hop_length=512, 
        f_min=50.0, 
        f_max=2000.0,
        to_db=True
    ):
        """
        Generates 2D Mel-Spectrogram energy distribution from 1D temporal pressure wave.
        """
        try:
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=f_min,
                fmax=f_max,
                power=2.0
            )
            if to_db:
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec
        except ImportError:
            # Fallback using standard numpy/scipy if librosa is not yet loaded
            return cls._fallback_stft_spectrogram(audio, sr, n_mels, n_fft, hop_length)

    @staticmethod
    def _fallback_stft_spectrogram(audio, sr, n_mels, n_fft, hop_length):
        from scipy import signal
        _, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        magnitude = np.abs(Zxx)
        # Interpolate or truncate to n_mels bins
        if magnitude.shape[0] > n_mels:
            magnitude = magnitude[:n_mels, :]
        elif magnitude.shape[0] < n_mels:
            pad = np.zeros((n_mels - magnitude.shape[0], magnitude.shape[1]))
            magnitude = np.vstack([magnitude, pad])
        return 20 * np.log10(np.maximum(magnitude, 1e-6))

    @classmethod
    def prepare_cnn_input(cls, audio_path, config=None, expected_shape=None):
        """
        Orchestrates full feature extraction, dimension validation, and tensor formatting.
        """
        config = config or {}
        target_sr = config.get('sample_rate', 16000)
        n_mels = config.get('n_mels', 128)
        n_fft = config.get('n_fft', 1024)
        hop_length = config.get('hop_length', 512)
        f_min = config.get('f_min', 50.0)
        f_max = config.get('f_max', 2000.0)

        # 1. Load audio
        audio, sr = cls.load_audio(audio_path, target_sr=target_sr)
        if len(audio) == 0:
            raise ValueError("Audio recording stream is empty.")

        # 2. Extract Mel-Spectrogram
        mel_spec = cls.compute_mel_spectrogram(
            audio=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            to_db=True
        )

        # 3. Add batch and channel dimensions for 2D-CNN: (batch, height, width, channels)
        # Shape: (1, n_mels, time_steps, 1)
        input_tensor = np.expand_dims(mel_spec, axis=(0, -1))

        # 4. If the model specifies a fixed input shape, adjust or pad time axis
        if expected_shape and len(expected_shape) == 4:
            target_h, target_w = expected_shape[1], expected_shape[2]
            if target_h and input_tensor.shape[1] != target_h:
                # Adjust height/mel bands
                input_tensor = input_tensor[:, :target_h, :, :]
            if target_w:
                current_w = input_tensor.shape[2]
                if current_w < target_w:
                    # Pad time dimension with zeros
                    pad_w = target_w - current_w
                    input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, pad_w), (0, 0)), mode='constant')
                elif current_w > target_w:
                    # Slice time dimension
                    input_tensor = input_tensor[:, :, :target_w, :]

        return input_tensor.astype(np.float32)


class InferenceService:
    """
    Clean inference orchestration interface for Acoustic Epidemiology.
    Guarantees genuine model evaluation, output dimension validation,
    and transparent failure handling without fabricating predictions.
    """

    @staticmethod
    def get_active_model():
        """
        Retrieves the active model version if and only if:
        1. It is marked ACTIVE in the database
        2. Its artifact physically exists on disk
        3. It defines non-empty verified_classes
        """
        active_models = ModelVersion.objects.filter(status=ModelStatus.ACTIVE)
        for model in active_models:
            if model.is_artifact_available() and model.verified_classes:
                return model
        return None

    @classmethod
    def validate_model_output(cls, raw_output, verified_classes):
        """
        Validates output shape and extracts normalized probability map.
        """
        if raw_output is None or len(raw_output) == 0:
            raise ValueError("Model produced empty output.")

        output_probs = raw_output[0].tolist() if hasattr(raw_output[0], 'tolist') else list(raw_output[0])

        if len(output_probs) != len(verified_classes):
            raise ValueError(
                f"Model output dimension ({len(output_probs)}) does not match "
                f"verified class label count ({len(verified_classes)})."
            )

        # Check for NaN / Inf
        if any(np.isnan(p) or np.isinf(p) for p in output_probs):
            raise ValueError("Model output contains NaN or infinite values.")

        # Build probability map rounded to 4 decimals
        prob_map = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(verified_classes, output_probs)
        }
        predicted_class = max(prob_map, key=prob_map.get)

        return predicted_class, prob_map

    @classmethod
    def execute_prediction(cls, recording, request_user=None):
        """
        Executes prediction on an authorized recording.
        If real model artifact is missing, records UNAVAILABLE state and raises ServiceUnavailable.
        """
        active_model = cls.get_active_model()

        if not active_model:
            # Transparently record that prediction could not be run because no CNN is deployed
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
            # 1. Retrieve or load model from reusable cache
            loaded_model = ModelCache.get_model(active_model)
            if loaded_model is None:
                raise FileNotFoundError(
                    f"Model artifact not found at configured path: {active_model.artifact_path}"
                )

            # 2. Inspect expected model input shape if available
            expected_shape = getattr(loaded_model, 'input_shape', None)

            # 3. Preprocess audio into time-frequency representation
            input_tensor = AudioPreprocessor.prepare_cnn_input(
                audio_path=recording.audio_file.path,
                config=active_model.preprocessing_config,
                expected_shape=expected_shape
            )

            # 4. Forward inference pass
            raw_output = loaded_model.predict(input_tensor, verbose=0)

            # 5. Validate output dimensions against verified class labels
            verified_classes = active_model.verified_classes
            predicted_class, prob_map = cls.validate_model_output(raw_output, verified_classes)

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # 6. Persist successful prediction
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
