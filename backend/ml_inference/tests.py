from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from rest_framework import status
from rest_framework.test import APIClient
from recordings.models import Recording, RecordingStatus
from ml_inference.models import ModelVersion, ModelStatus, Prediction, PredictionStatus
from accounts.models import UserRole


class MLInferenceTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='dr_test', password='Password123!')
        self.recording = Recording.objects.create(
            original_filename='patient_breath.wav',
            file_format='wav',
            file_size_bytes=10240,
            duration_seconds=2.0,
            sample_rate=16000,
            channels=1,
            recording_status=RecordingStatus.VALIDATED,
            uploader=self.user
        )

    def test_model_version_active_requires_artifact(self):
        # Trying to save an ACTIVE model without artifact should raise ValidationError
        model = ModelVersion(
            version='pediatric-cnn-v1',
            status=ModelStatus.ACTIVE,
            artifact_path='',
            verified_classes=['Normal', 'Wheeze', 'Crackle']
        )
        with self.assertRaises(ValidationError):
            model.clean()

    def test_model_list_endpoint(self):
        ModelVersion.objects.create(
            version='pediatric-cnn-v0.1-dev',
            architecture='2D-CNN-Spectrogram',
            status=ModelStatus.DEVELOPMENT,
            verified_classes=['Normal', 'Wheeze', 'Crackle']
        )
        url = reverse('model-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['version'], 'pediatric-cnn-v0.1-dev')

    def test_active_model_returns_503_when_no_artifact_deployed(self):
        url = reverse('model-active')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['code'], 'NO_ACTIVE_MODEL')

    def test_predict_returns_503_and_persists_unavailable_record(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('recording-predict', kwargs={'recording_id': self.recording.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['code'], 'MODEL_NOT_DEPLOYED')

        # Verify that an UNAVAILABLE record was saved in database
        pred_record = Prediction.objects.filter(recording=self.recording).first()
        self.assertIsNotNone(pred_record)
        self.assertEqual(pred_record.prediction_status, PredictionStatus.UNAVAILABLE)
        self.assertIn("No verified CNN model artifact", pred_record.error_details)

    def test_prediction_history_endpoint(self):
        self.client.force_authenticate(user=self.user)
        # Create a mock failed prediction for testing history retrieval
        Prediction.objects.create(
            recording=self.recording,
            prediction_status=PredictionStatus.UNAVAILABLE,
            error_details="Model artifact missing."
        )

        url = reverse('recording-prediction-history', kwargs={'recording_id': self.recording.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['prediction_status'], PredictionStatus.UNAVAILABLE)

    def test_validate_model_output_dimension_mismatch(self):
        from ml_inference.services import InferenceService
        # 3 output probabilities provided, but verified_classes has 4 classes
        raw_output = [[0.7, 0.2, 0.1]]
        classes = ['Normal', 'Wheeze', 'Crackle', 'Stridor']
        with self.assertRaises(ValueError) as ctx:
            InferenceService.validate_model_output(raw_output, classes)
        self.assertIn("does not match verified class label count", str(ctx.exception))

    def test_validate_model_output_nan_inf(self):
        from ml_inference.services import InferenceService
        raw_output = [[float('nan'), 0.5]]
        classes = ['Normal', 'Wheeze']
        with self.assertRaises(ValueError) as ctx:
            InferenceService.validate_model_output(raw_output, classes)
        self.assertIn("contains NaN or infinite values", str(ctx.exception))

    def test_model_version_active_requires_verified_classes(self):
        # Even with artifact path, clean() must require non-empty verified_classes
        model = ModelVersion(
            version='pediatric-cnn-no-classes',
            status=ModelStatus.ACTIVE,
            artifact_path='ml/models/dummy.keras',
            verified_classes=[]
        )
        with self.assertRaises(ValidationError) as ctx:
            model.clean()
        self.assertTrue('status' in ctx.exception.message_dict or 'verified_classes' in ctx.exception.message_dict)

    def test_audio_preprocessor_synthetic_signal(self):
        import numpy as np
        from ml_inference.services import AudioPreprocessor
        # Generate 1-second 16kHz sine wave
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        mel_spec = AudioPreprocessor.compute_mel_spectrogram(
            audio=audio,
            sr=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        self.assertEqual(mel_spec.shape[0], 128)
        self.assertTrue(mel_spec.shape[1] > 0)
        self.assertFalse(np.isnan(mel_spec).any())

    def test_model_cache_clear(self):
        from ml_inference.services import ModelCache
        ModelCache.clear()
        self.assertIsNone(ModelCache._cached_model)
        self.assertIsNone(ModelCache._cached_version_id)

