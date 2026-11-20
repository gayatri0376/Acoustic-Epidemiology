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

    def test_audio_preprocessor_with_8khz_and_normalization(self):
        import numpy as np
        from ml_inference.services import AudioPreprocessor
        # 8kHz synthetic breath signal
        t = np.linspace(0, 1.0, 8000, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

        mel_norm = AudioPreprocessor.compute_mel_spectrogram(
            audio=audio,
            sr=8000,
            n_mels=128,
            n_fft=1024,
            hop_length=512,
            f_min=50.0,
            f_max=2000.0,
            apply_bandpass=True,
            normalization='minmax_01'
        )
        self.assertEqual(mel_norm.shape[0], 128)
        self.assertTrue(mel_norm.min() >= 0.0)
        self.assertTrue(mel_norm.max() <= 1.0)
        self.assertFalse(np.isnan(mel_norm).any())

    def test_audio_preprocessor_prepare_cnn_input_8khz_alignment(self):
        import os
        import numpy as np
        import soundfile as sf
        import tempfile
        from ml_inference.services import AudioPreprocessor

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # 9-second synthetic 8kHz signal
            t = np.linspace(0, 9.0, 72000, endpoint=False)
            audio = (0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
            sf.write(tmp_path, audio, 8000)

            config = {
                'sample_rate': 8000,
                'n_mels': 128,
                'n_fft': 1024,
                'hop_length': 512,
                'f_min': 50.0,
                'f_max': 2000.0,
                'apply_bandpass': True,
                'normalization': 'minmax_01'
            }
            tensor = AudioPreprocessor.prepare_cnn_input(
                audio_path=tmp_path,
                config=config,
                expected_shape=[1, 128, 128, 1]
            )
            self.assertEqual(tensor.shape, (1, 128, 128, 1))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertTrue(tensor.min() >= 0.0)
            self.assertTrue(tensor.max() <= 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_model_cache_clear(self):
        from ml_inference.services import ModelCache
        ModelCache.clear()
        self.assertIsNone(ModelCache._cached_model)
        self.assertIsNone(ModelCache._cached_version_id)


class EndToEndWorkflowTests(TestCase):
    """
    Validates the complete 7-step end-to-end diagnostic workflow:
    1. Authentication (login / token generation)
    2. Audio recording upload and validation
    3. Prediction request (returns 503 when CNN artifact is not deployed)
    4. Prediction retrieval (status='UNAVAILABLE' persisted)
    5. Explanation request (returns 503 when gradient model is unavailable)
    6. Explanation retrieval (status='UNSUPPORTED' persisted)
    7. Clinical assessment submission by clinician
    """

    def setUp(self):
        self.client = APIClient()
        self.uploader = User.objects.create_user(
            username='dr_auscultator',
            password='Password123!'
        )
        self.clinician = User.objects.create_user(
            username='senior_pulmonologist',
            password='Password123!'
        )
        self.clinician.profile.role = UserRole.CLINICIAN
        self.clinician.profile.save()

    def test_complete_diagnostic_lifecycle(self):
        from recordings.tests import generate_valid_wav_file
        from explanations.models import Explanation, ExplanationStatus, ExplanationMethod
        from assessments.models import Assessment, ReviewStatus, ReviewerConclusion

        # Step 1: Login / Token Authentication
        login_res = self.client.post(
            reverse('auth-login'),
            {'username': 'dr_auscultator', 'password': 'Password123!'},
            format='json'
        )
        self.assertEqual(login_res.status_code, status.HTTP_200_OK)
        uploader_token = login_res.data['token']
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {uploader_token}')

        # Step 2: Upload valid audio recording
        audio_file = generate_valid_wav_file(filename="pediatric_chest_sound.wav", duration=1.5, sample_rate=16000)
        upload_res = self.client.post(
            reverse('recording-list-create'),
            {'audio_file': audio_file, 'patient_pseudo_id': 'CASE-771'},
            format='multipart'
        )
        self.assertEqual(upload_res.status_code, status.HTTP_201_CREATED)
        recording_id = upload_res.data['recording']['id']

        # Step 3: Request inference (expect 503 MODEL_NOT_DEPLOYED since real weights are pending)
        predict_res = self.client.post(
            reverse('recording-predict', kwargs={'recording_id': recording_id})
        )
        self.assertEqual(predict_res.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(predict_res.data['code'], 'MODEL_NOT_DEPLOYED')
        prediction_id = predict_res.data['prediction_id']

        # Step 4: Retrieve prediction details and audit record
        pred_detail_res = self.client.get(
            reverse('prediction-detail', kwargs={'pk': prediction_id})
        )
        self.assertEqual(pred_detail_res.status_code, status.HTTP_200_OK)
        self.assertEqual(pred_detail_res.data['prediction_status'], PredictionStatus.UNAVAILABLE)
        self.assertEqual(pred_detail_res.data['predicted_class'], '')
        self.assertEqual(pred_detail_res.data['probability_scores'], {})

        # Step 5: Request Grad-CAM explanation (expect 503 GRADIENT_MODEL_UNAVAILABLE)
        exp_res = self.client.post(
            reverse('prediction-explanations', kwargs={'prediction_id': prediction_id}),
            {'method': ExplanationMethod.GRAD_CAM},
            format='json'
        )
        self.assertEqual(exp_res.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(exp_res.data['code'], 'GRADIENT_MODEL_UNAVAILABLE')
        explanation_id = exp_res.data['explanation_id']

        # Step 6: Retrieve explanation record
        exp_detail_res = self.client.get(
            reverse('explanation-detail', kwargs={'pk': explanation_id})
        )
        self.assertEqual(exp_detail_res.status_code, status.HTTP_200_OK)
        self.assertEqual(exp_detail_res.data['status'], ExplanationStatus.UNSUPPORTED)

        # Step 7: Clinician reviews and submits clinical assessment
        self.client.credentials()  # Clear uploader credentials
        clinician_login = self.client.post(
            reverse('auth-login'),
            {'username': 'senior_pulmonologist', 'password': 'Password123!'},
            format='json'
        )
        clinician_token = clinician_login.data['token']
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {clinician_token}')

        assessment_res = self.client.post(
            reverse('prediction-assessment-create', kwargs={'prediction_id': prediction_id}),
            {
                'review_status': ReviewStatus.CONFIRMED,
                'reviewer_conclusion': ReviewerConclusion.WHEEZING,
                'clinical_notes': 'Auscultatory confirmation of bilateral expiratory wheezing.'
            },
            format='json'
        )
        self.assertEqual(assessment_res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(assessment_res.data['assessment']['reviewer_username'], 'senior_pulmonologist')
        self.assertEqual(assessment_res.data['assessment']['reviewer_conclusion'], ReviewerConclusion.WHEEZING)


class ModelRegistrationCommandTests(TestCase):
    """
    Focused unit and integration tests for the register_model management command.
    Explicitly separates tests for validation guards and error handling
    from mocked unit tests that verify database persistence and schema integrity.
    """

    def setUp(self):
        from django.conf import settings
        from pathlib import Path
        self.project_root = Path(settings.BASE_DIR.parent).resolve()
        self.model_dir = self.project_root / 'ml' / 'models' / 'saved_models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.temp_files = []

    def tearDown(self):
        for path in self.temp_files:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def _create_temp_artifact(self, filename="test_model.keras"):
        path = self.model_dir / filename
        path.write_bytes(b"PK\x03\x04test_mock_keras_payload")
        self.temp_files.append(path)
        return path

    def test_missing_artifact_raises_command_error(self):
        """REAL VALIDATION: Verifies command rejects missing model files."""
        from django.core.management import call_command, CommandError
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'register_model',
                '--model-version', 'test-missing-v1',
                '--artifact-path', 'ml/models/saved_models/nonexistent_model.keras',
                '--classes', 'Normal', 'Wheeze', 'Crackle',
            )
        self.assertIn("Artifact file not found", str(ctx.exception))

    def test_path_traversal_is_forbidden(self):
        """SECURITY GUARD: Verifies directory traversal outside repo root is blocked."""
        from django.core.management import call_command, CommandError
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'register_model',
                '--model-version', 'test-traversal-v1',
                '--artifact-path', '../../etc/passwd',
                '--classes', 'Normal', 'Wheeze',
            )
        self.assertIn("Security violation", str(ctx.exception))
        self.assertIn("Path traversal is strictly forbidden", str(ctx.exception))

    def test_storage_convention_violation_rejected(self):
        """CONVENTION GUARD: Rejects files located outside ml/models/ hierarchy."""
        from django.core.management import call_command, CommandError
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'register_model',
                '--model-version', 'test-convention-v1',
                '--artifact-path', 'backend/manage.py',
                '--classes', 'Normal', 'Wheeze',
            )
        self.assertIn("Model storage convention violation", str(ctx.exception))
        self.assertIn("must be located inside the 'ml/models/' hierarchy", str(ctx.exception))

    def test_unsupported_file_extension_rejected(self):
        """FORMAT GUARD: Rejects artifacts without recognized ML file extensions."""
        from django.core.management import call_command, CommandError
        temp_txt = self._create_temp_artifact("dummy_weights.txt")
        rel_path = str(temp_txt.relative_to(self.project_root))
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'register_model',
                '--model-version', 'test-ext-v1',
                '--artifact-path', rel_path,
                '--classes', 'Normal', 'Wheeze',
            )
        self.assertIn("Unsupported model artifact extension", str(ctx.exception))

    def test_missing_tensorflow_runtime_fails_gracefully(self):
        """DEPENDENCY GUARD: Verifies graceful failure when TensorFlow is not installed."""
        from django.core.management import call_command, CommandError
        from unittest.mock import patch
        temp_artifact = self._create_temp_artifact("valid_name.keras")
        rel_path = str(temp_artifact.relative_to(self.project_root))

        # Simulate tensorflow being uninstalled
        with patch.dict('sys.modules', {'tensorflow': None}):
            with self.assertRaises(CommandError) as ctx:
                call_command(
                    'register_model',
                    '--model-version', 'test-dep-v1',
                    '--artifact-path', rel_path,
                    '--classes', 'Normal', 'Wheeze', 'Crackle',
                )
            self.assertIn("ML Runtime Dependency Error", str(ctx.exception))
            self.assertIn("TensorFlow/Keras is not installed", str(ctx.exception))


    def test_dimension_mismatch_with_mocked_model(self):
        """
        UNIT TEST DOUBLE: Uses mock model to verify that output neuron count
        is strictly checked against verified_classes count.
        """
        from unittest.mock import patch
        from django.core.management import CommandError
        from ml_inference.management.commands.register_model import inspect_and_validate_cnn_artifact
        temp_artifact = self._create_temp_artifact("mock_dim_model.keras")

        class MockLayer:
            name = 'dense_final'
            activation = lambda: None
        MockLayer.activation.__name__ = 'softmax'
        class MockModel:
            input_shape = (None, 128, 128, 1)
            output_shape = (None, 4)
            layers = [MockLayer()]

        mock_tf = type('MockTF', (), {'keras': type('MockKeras', (), {'models': type('MockModels', (), {'load_model': lambda p, compile: MockModel()})})})
        with patch.dict('sys.modules', {'tensorflow': mock_tf}):
            with self.assertRaises(CommandError) as ctx:
                inspect_and_validate_cnn_artifact(
                    resolved_path=temp_artifact,
                    verified_classes=['Normal', 'Wheeze', 'Crackle'],  # 3 classes vs 4 neurons
                )
            self.assertIn("Output dimension mismatch", str(ctx.exception))

    def test_target_conv_layer_not_found_rejected(self):
        """
        UNIT TEST DOUBLE: Verifies that an invalid target_conv_layer for Grad-CAM is rejected.
        """
        from unittest.mock import patch
        from django.core.management import CommandError
        from ml_inference.management.commands.register_model import inspect_and_validate_cnn_artifact
        temp_artifact = self._create_temp_artifact("mock_conv_model.keras")

        class MockLayer:
            name = 'conv2d_initial'
            activation = lambda: None
        MockLayer.activation.__name__ = 'relu'

        class MockModel:
            input_shape = (None, 128, 128, 1)
            output_shape = (None, 3)
            layers = [MockLayer()]

        mock_tf = type('MockTF', (), {'keras': type('MockKeras', (), {'models': type('MockModels', (), {'load_model': lambda p, compile: MockModel()})})})
        with patch.dict('sys.modules', {'tensorflow': mock_tf}):
            with self.assertRaises(CommandError) as ctx:
                inspect_and_validate_cnn_artifact(
                    resolved_path=temp_artifact,
                    verified_classes=['Normal', 'Wheeze', 'Crackle'],
                    target_conv_layer='conv2d_nonexistent',
                )
            self.assertIn("Target Conv2D layer 'conv2d_nonexistent' not found", str(ctx.exception))

    def test_dry_run_does_not_persist_to_database(self):
        """
        UNIT TEST DOUBLE: Verifies --dry-run validates without committing to MySQL.
        """
        from unittest.mock import patch
        from django.core.management import call_command
        from ml_inference.models import ModelVersion
        temp_artifact = self._create_temp_artifact("mock_dry_run.keras")
        rel_path = str(temp_artifact.relative_to(self.project_root))

        mock_inspection = {
            "input_shape": [None, 128, 128, 1],
            "output_shape": [None, 3],
            "output_activation": "softmax",
        }

        with patch('ml_inference.management.commands.register_model.inspect_and_validate_cnn_artifact', return_value=mock_inspection):
            call_command(
                'register_model',
                '--model-version', 'test-dry-run-v1',
                '--artifact-path', rel_path,
                '--classes', 'Normal', 'Wheeze', 'Crackle',
                '--dry-run',
            )

        self.assertFalse(ModelVersion.objects.filter(version='test-dry-run-v1').exists())

    def test_successful_registration_and_persistence(self):
        """
        UNIT TEST DOUBLE: Verifies successful registration, class ordering,
        and database record fields using a verified mock inspection result.
        """
        from unittest.mock import patch
        from io import StringIO
        from django.core.management import call_command
        from ml_inference.models import ModelVersion, ModelStatus
        temp_artifact = self._create_temp_artifact("mock_success.keras")
        rel_path = str(temp_artifact.relative_to(self.project_root))

        mock_inspection = {
            "input_shape": [None, 128, 128, 1],
            "output_shape": [None, 3],
            "output_activation": "softmax",
        }

        out = StringIO()
        with patch('ml_inference.management.commands.register_model.inspect_and_validate_cnn_artifact', return_value=mock_inspection):
            call_command(
                'register_model',
                '--model-version', 'pediatric-cnn-test-v1',
                '--artifact-path', rel_path,
                '--classes', 'Normal', 'Wheeze', 'Crackle',
                '--status', 'ACTIVE',
                '--sample-rate', '16000',
                '--n-mels', '128',
                stdout=out,
            )

        model = ModelVersion.objects.filter(version='pediatric-cnn-test-v1').first()
        self.assertIsNotNone(model)
        self.assertEqual(model.status, ModelStatus.ACTIVE)
        self.assertEqual(model.verified_classes, ['Normal', 'Wheeze', 'Crackle'])
        self.assertEqual(model.preprocessing_config['sample_rate'], 16000)
        self.assertEqual(model.preprocessing_config['output_activation'], 'softmax')
        self.assertEqual(model.preprocessing_config['expected_input_shape'], [None, 128, 128, 1])

    def test_duplicate_version_rejected_without_replace(self):
        """
        UNIT TEST DOUBLE: Verifies duplicate version strings cannot be accidentally overwritten.
        """
        from unittest.mock import patch
        from django.core.management import call_command, CommandError
        from ml_inference.models import ModelVersion
        temp_artifact = self._create_temp_artifact("mock_dup.keras")
        rel_path = str(temp_artifact.relative_to(self.project_root))

        mock_inspection = {
            "input_shape": [None, 128, 128, 1],
            "output_shape": [None, 3],
            "output_activation": "softmax",
        }

        with patch('ml_inference.management.commands.register_model.inspect_and_validate_cnn_artifact', return_value=mock_inspection):
            call_command(
                'register_model',
                '--model-version', 'pediatric-cnn-dup-v1',
                '--artifact-path', rel_path,
                '--classes', 'Normal', 'Wheeze', 'Crackle',
            )

            # Attempting second registration with same version without --replace must fail
            with self.assertRaises(CommandError) as ctx:
                call_command(
                    'register_model',
                    '--model-version', 'pediatric-cnn-dup-v1',
                    '--artifact-path', rel_path,
                    '--classes', 'Normal', 'Wheeze', 'Crackle',
                )
            self.assertIn("already exists in database", str(ctx.exception))
            self.assertIn("Specify --replace", str(ctx.exception))

    def test_register_model_preserves_bandpass_and_normalization(self):
        """Verifies register_model captures apply_bandpass, normalization, filter_order, and top_db."""
        from unittest.mock import patch
        from django.core.management import call_command
        from ml_inference.models import ModelVersion
        temp_artifact = self._create_temp_artifact("mock_bandpass_norm.keras")
        rel_path = str(temp_artifact.relative_to(self.project_root))

        mock_inspection = {
            "input_shape": [None, 128, 128, 1],
            "output_shape": [None, 3],
            "output_activation": "softmax",
        }

        with patch('ml_inference.management.commands.register_model.inspect_and_validate_cnn_artifact', return_value=mock_inspection):
            call_command(
                'register_model',
                '--model-version', 'pediatric-cnn-full-config-v1',
                '--artifact-path', rel_path,
                '--classes', 'Normal', 'Wheeze', 'Crackle',
                '--sample-rate', '8000',
                '--apply-bandpass',
                '--normalization', 'minmax_01',
                '--top-db', '80.0',
                '--filter-order', '5',
            )

        model = ModelVersion.objects.get(version='pediatric-cnn-full-config-v1')
        self.assertEqual(model.preprocessing_config['sample_rate'], 8000)
        self.assertTrue(model.preprocessing_config['apply_bandpass'])
        self.assertEqual(model.preprocessing_config['normalization'], 'minmax_01')
        self.assertEqual(model.preprocessing_config['top_db'], 80.0)
        self.assertEqual(model.preprocessing_config['filter_order'], 5)


class PreprocessingValidationTests(TestCase):
    """
    Regression and safety tests for model-specific preprocessing validation,
    missing configuration key detection, NaN/Inf rejection, and dimension guards.
    """

    def test_validate_preprocessing_config_missing_required_keys(self):
        from ml_inference.services import AudioPreprocessor
        # Missing sample_rate and hop_length
        partial_config = {'n_mels': 128, 'n_fft': 1024}
        with self.assertRaises(ValueError) as ctx:
            AudioPreprocessor.validate_preprocessing_config(partial_config)
        self.assertIn("Missing required parameter(s)", str(ctx.exception))
        self.assertIn("sample_rate", str(ctx.exception))
        self.assertIn("hop_length", str(ctx.exception))

    def test_validate_preprocessing_config_invalid_normalization(self):
        from ml_inference.services import AudioPreprocessor
        invalid_config = {
            'sample_rate': 8000,
            'n_mels': 128,
            'n_fft': 1024,
            'hop_length': 512,
            'normalization': 'invalid_method_xyz'
        }
        with self.assertRaises(ValueError) as ctx:
            AudioPreprocessor.validate_preprocessing_config(invalid_config)
        self.assertIn("Invalid normalization method", str(ctx.exception))

    def test_prepare_cnn_input_rejects_nan_inf(self):
        import numpy as np
        import tempfile
        import soundfile as sf
        import os
        from ml_inference.services import AudioPreprocessor
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            audio = np.zeros(8000, dtype=np.float32)
            sf.write(tmp_path, audio, 8000)
            config = {
                'sample_rate': 8000,
                'n_mels': 128,
                'n_fft': 1024,
                'hop_length': 512,
            }
            # Mock compute_mel_spectrogram to return NaN array
            nan_spec = np.full((128, 128), np.nan, dtype=np.float32)
            with patch.object(AudioPreprocessor, 'compute_mel_spectrogram', return_value=nan_spec):
                with self.assertRaises(ValueError) as ctx:
                    AudioPreprocessor.prepare_cnn_input(tmp_path, config=config)
                self.assertIn("contains NaN or Inf values", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_prepare_cnn_input_dimension_validation(self):
        import numpy as np
        import tempfile
        import soundfile as sf
        import os
        from ml_inference.services import AudioPreprocessor

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            audio = np.zeros(8000, dtype=np.float32)
            sf.write(tmp_path, audio, 8000)
            config = {
                'sample_rate': 8000,
                'n_mels': 128,
                'n_fft': 1024,
                'hop_length': 512,
                'apply_bandpass': True,
                'normalization': 'minmax_01',
            }
            tensor = AudioPreprocessor.prepare_cnn_input(
                tmp_path,
                config=config,
                expected_shape=[1, 128, 128, 1]
            )
            self.assertEqual(tensor.ndim, 4)
            self.assertEqual(tensor.shape, (1, 128, 128, 1))
            self.assertFalse(np.isnan(tensor).any())
            self.assertFalse(np.isinf(tensor).any())
            self.assertTrue(tensor.min() >= 0.0)
            self.assertTrue(tensor.max() <= 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)





# =============================================================================
# TASK 8 — Model Routing, API Safety & Serializer Tests
# Added: 2026-11-20
# =============================================================================

class ModelRoutingTests(TestCase):
    """
    Phase 1 & 2 — Verifies that InferenceService.get_active_model() correctly
    filters by task_type=SOUND_CLASSIFICATION so that:
    - Acoustic endpoint never selects a DISEASE_CLASSIFICATION model.
    - Disease endpoint never selects a SOUND_CLASSIFICATION model.
    - Selection is deterministic when multiple models share ACTIVE status.

    ISSUE-3 regression suite: confirms the fix is permanently in place.
    """

    def _make_model(self, version, task_type, status, artifact_path='ml/models/saved_models/sprsound-cnn-v1-fold0.keras'):
        from ml_inference.models import ModelVersion, ModelTaskType
        return ModelVersion.objects.create(
            version=version,
            task_type=task_type,
            status=status,
            artifact_path=artifact_path,
            verified_classes=['Normal', 'CAS', 'DAS', 'CAS & DAS'] if task_type == 'SOUND_CLASSIFICATION'
                             else ['Healthy Control', 'Pneumonia', 'Wheezing Disorders'],
        )

    def test_get_active_model_returns_only_sound_classification(self):
        """ISSUE-3 FIX: get_active_model() must never return a DISEASE model."""
        from ml_inference.models import ModelTaskType, ModelStatus
        from ml_inference.services import InferenceService

        sound_model = self._make_model(
            'routing-test-sound-v1',
            ModelTaskType.SOUND_CLASSIFICATION,
            ModelStatus.ACTIVE,
        )
        # Disease model is also ACTIVE — without the fix, ordering by -created_at
        # could cause this model to be returned first.
        disease_model = self._make_model(
            'routing-test-disease-v1',
            ModelTaskType.DISEASE_CLASSIFICATION,
            ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-disease-cnn-v2.keras',
        )

        result = InferenceService.get_active_model()
        self.assertIsNotNone(result, "Expected an active sound model to be found.")
        self.assertEqual(
            result.task_type,
            ModelTaskType.SOUND_CLASSIFICATION,
            f"get_active_model() returned task_type={result.task_type!r}; "
            "expected SOUND_CLASSIFICATION. ISSUE-3 regression."
        )
        self.assertNotEqual(
            result.id,
            disease_model.id,
            "get_active_model() must never return the DISEASE_CLASSIFICATION model."
        )

    def test_get_active_model_returns_none_when_only_disease_model_active(self):
        """If only a DISEASE model is active, acoustic endpoint must return None (503)."""
        from ml_inference.models import ModelTaskType, ModelStatus
        from ml_inference.services import InferenceService

        self._make_model(
            'routing-only-disease-v1',
            ModelTaskType.DISEASE_CLASSIFICATION,
            ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-disease-cnn-v2.keras',
        )
        result = InferenceService.get_active_model()
        self.assertIsNone(
            result,
            "get_active_model() must return None when no SOUND_CLASSIFICATION model is active. "
            "Returned a disease model — ISSUE-3 not fixed."
        )

    def test_get_active_disease_model_returns_only_disease_classification(self):
        """DiseaseInferenceService must never return a SOUND_CLASSIFICATION model."""
        from ml_inference.models import ModelTaskType, ModelStatus
        from ml_inference.disease_service import DiseaseInferenceService

        self._make_model(
            'routing-test-sound-v2',
            ModelTaskType.SOUND_CLASSIFICATION,
            ModelStatus.ACTIVE,
        )
        # Only disease model should be found
        disease_model = self._make_model(
            'routing-test-disease-v2',
            ModelTaskType.DISEASE_CLASSIFICATION,
            ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-disease-cnn-v2.keras',
        )

        result = DiseaseInferenceService.get_active_disease_model()
        self.assertIsNotNone(result)
        self.assertEqual(result.task_type, ModelTaskType.DISEASE_CLASSIFICATION)
        self.assertEqual(result.id, disease_model.id)

    def test_get_active_model_returns_none_when_no_active_model(self):
        """503 path: no model at all — must return None, not raise an exception."""
        from ml_inference.services import InferenceService
        result = InferenceService.get_active_model()
        self.assertIsNone(result)

    def test_retired_sound_model_not_selected(self):
        """A RETIRED SOUND_CLASSIFICATION model must not be returned as active."""
        from ml_inference.models import ModelTaskType, ModelStatus
        from ml_inference.services import InferenceService

        self._make_model(
            'routing-retired-sound-v1',
            ModelTaskType.SOUND_CLASSIFICATION,
            ModelStatus.RETIRED,
        )
        result = InferenceService.get_active_model()
        self.assertIsNone(result, "RETIRED model must not be returned as active model.")

    def test_development_sound_model_not_selected(self):
        """A DEVELOPMENT model must not be served as active."""
        from ml_inference.models import ModelTaskType, ModelStatus
        from ml_inference.services import InferenceService

        self._make_model(
            'routing-dev-sound-v1',
            ModelTaskType.SOUND_CLASSIFICATION,
            ModelStatus.DEVELOPMENT,
        )
        result = InferenceService.get_active_model()
        self.assertIsNone(result, "DEVELOPMENT model must not be returned as active model.")

    def test_active_model_endpoint_excludes_disease_model(self):
        """
        GET /api/v1/models/active/ must return the SOUND_CLASSIFICATION model,
        not the disease model, when both are ACTIVE.
        """
        from ml_inference.models import ModelTaskType, ModelStatus
        from django.urls import reverse

        sound_model = self._make_model(
            'routing-endpoint-sound-v1',
            ModelTaskType.SOUND_CLASSIFICATION,
            ModelStatus.ACTIVE,
        )
        self._make_model(
            'routing-endpoint-disease-v1',
            ModelTaskType.DISEASE_CLASSIFICATION,
            ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-disease-cnn-v2.keras',
        )

        client = APIClient()
        url = reverse('model-active')
        response = client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['task_type'], ModelTaskType.SOUND_CLASSIFICATION)
        self.assertEqual(response.data['version'], sound_model.version)

    def test_active_model_endpoint_returns_503_when_only_disease_active(self):
        """
        GET /api/v1/models/active/ must return 503 when no SOUND_CLASSIFICATION model is active.
        It must not return the disease model as a sound model.
        """
        from ml_inference.models import ModelTaskType, ModelStatus
        from django.urls import reverse

        self._make_model(
            'routing-only-disease-endpoint-v1',
            ModelTaskType.DISEASE_CLASSIFICATION,
            ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-disease-cnn-v2.keras',
        )

        client = APIClient()
        url = reverse('model-active')
        response = client.get(url)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data['code'], 'NO_ACTIVE_MODEL')


class PredictionSerializerSafetyTests(TestCase):
    """
    Phase 4 — Verifies API response safety:
    - research_disclaimer is always present on prediction responses.
    - model_task_type is correctly included.
    - Probabilities are valid (non-negative, 4 entries for the acoustic model).
    - Predicted class is one of the verified classes.
    - No clinical diagnosis language is embedded in the core serializer.
    """

    def setUp(self):
        from django.contrib.auth.models import User
        from recordings.models import Recording, RecordingStatus
        from ml_inference.models import ModelVersion, ModelStatus, ModelTaskType, Prediction, PredictionStatus

        self.user = User.objects.create_user(username='safety_tester', password='Password123!')
        self.recording = Recording.objects.create(
            original_filename='test_chest.wav',
            file_format='wav',
            file_size_bytes=16384,
            duration_seconds=2.0,
            sample_rate=8000,
            channels=1,
            recording_status=RecordingStatus.VALIDATED,
            uploader=self.user,
        )
        self.model_version = ModelVersion.objects.create(
            version='safety-test-sound-v1',
            task_type=ModelTaskType.SOUND_CLASSIFICATION,
            status=ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-cnn-v1-fold0.keras',
            verified_classes=['Normal', 'CAS', 'DAS', 'CAS & DAS'],
        )
        self.prediction = Prediction.objects.create(
            recording=self.recording,
            model_version=self.model_version,
            predicted_class='Normal',
            probability_scores={'Normal': 0.5132, 'CAS': 0.1267, 'DAS': 0.2990, 'CAS & DAS': 0.0611},
            prediction_status=PredictionStatus.SUCCESS,
            inference_duration_ms=142,
        )

    def test_disclaimer_always_present(self):
        """research_disclaimer must always be present in prediction response."""
        from ml_inference.serializers import PredictionSerializer
        data = PredictionSerializer(self.prediction).data
        self.assertIn('research_disclaimer', data)
        disclaimer = data['research_disclaimer']
        self.assertIsInstance(disclaimer, str)
        self.assertGreater(len(disclaimer), 10)

    def test_disclaimer_contains_required_text(self):
        """The disclaimer must contain the mandated 'not a medical diagnosis' statement."""
        from ml_inference.serializers import PredictionSerializer
        data = PredictionSerializer(self.prediction).data
        disclaimer = data['research_disclaimer'].lower()
        self.assertIn('not a medical diagnosis', disclaimer,
                      "Mandatory disclaimer text 'not a medical diagnosis' missing from response.")
        self.assertIn('experimental', disclaimer,
                      "Disclaimer must state the model is experimental.")

    def test_disclaimer_does_not_claim_clinical_accuracy(self):
        """Disclaimer must not contain clinical accuracy claims."""
        from ml_inference.serializers import PredictionSerializer
        data = PredictionSerializer(self.prediction).data
        disclaimer = data['research_disclaimer'].lower()
        forbidden_phrases = ['clinically validated', 'fda approved', '100% accurate', 'confirms diagnosis']
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, disclaimer, f"Forbidden clinical claim found: '{phrase}'")

    def test_model_task_type_in_response(self):
        """model_task_type must be SOUND_CLASSIFICATION in prediction response."""
        from ml_inference.serializers import PredictionSerializer
        from ml_inference.models import ModelTaskType
        data = PredictionSerializer(self.prediction).data
        self.assertIn('model_task_type', data)
        self.assertEqual(data['model_task_type'], ModelTaskType.SOUND_CLASSIFICATION)

    def test_probabilities_are_valid(self):
        """All probability scores must be non-negative and plausibly sum to ~1."""
        from ml_inference.serializers import PredictionSerializer
        data = PredictionSerializer(self.prediction).data
        probs = data['probability_scores']
        self.assertEqual(len(probs), 4, "Expected 4 classes in probability map.")
        for cls, prob in probs.items():
            self.assertGreaterEqual(prob, 0.0, f"Negative probability for class {cls!r}.")
            self.assertLessEqual(prob, 1.0, f"Probability > 1.0 for class {cls!r}.")
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, delta=0.01, msg="Probabilities must sum to ~1.0.")

    def test_predicted_class_is_valid_label(self):
        """predicted_class must be one of the verified classes."""
        from ml_inference.serializers import PredictionSerializer
        data = PredictionSerializer(self.prediction).data
        valid_classes = ['Normal', 'CAS', 'DAS', 'CAS & DAS']
        self.assertIn(data['predicted_class'], valid_classes)

    def test_failed_prediction_includes_error_details(self):
        """A FAILED prediction must expose error_details, not hide it."""
        from ml_inference.models import Prediction, PredictionStatus
        from ml_inference.serializers import PredictionSerializer
        failed_pred = Prediction.objects.create(
            recording=self.recording,
            model_version=self.model_version,
            prediction_status=PredictionStatus.FAILED,
            error_details='Inference error: audio file corrupted.',
        )
        data = PredictionSerializer(failed_pred).data
        self.assertEqual(data['prediction_status'], PredictionStatus.FAILED)
        self.assertIn('corrupted', data['error_details'])
        # Disclaimer still present even on failure
        self.assertIn('research_disclaimer', data)

    def test_unavailable_prediction_has_empty_class_and_scores(self):
        """An UNAVAILABLE prediction must not contain fabricated class or scores."""
        from ml_inference.models import Prediction, PredictionStatus
        from ml_inference.serializers import PredictionSerializer
        unavailable_pred = Prediction.objects.create(
            recording=self.recording,
            prediction_status=PredictionStatus.UNAVAILABLE,
            error_details='No model deployed.',
        )
        data = PredictionSerializer(unavailable_pred).data
        self.assertEqual(data['prediction_status'], PredictionStatus.UNAVAILABLE)
        self.assertEqual(data['predicted_class'], '')
        self.assertEqual(data['probability_scores'], {})
        self.assertIsNone(data['model_version'])


class APIRoutingEndpointTests(TestCase):
    """
    Phase 3 — API-level integration tests (mocked).
    Tests authenticated and unauthenticated access patterns,
    routing correctness, and response shape.
    """

    def setUp(self):
        from django.contrib.auth.models import User
        from recordings.models import Recording, RecordingStatus

        self.client = APIClient()
        self.user = User.objects.create_user(username='api_tester', password='Password123!')
        self.other_user = User.objects.create_user(username='other_user', password='Password123!')
        self.recording = Recording.objects.create(
            original_filename='breath.wav',
            file_format='wav',
            file_size_bytes=8192,
            duration_seconds=1.5,
            sample_rate=8000,
            channels=1,
            recording_status=RecordingStatus.VALIDATED,
            uploader=self.user,
        )

    def test_predict_requires_authentication(self):
        """Unauthenticated POST to /predict/ must return 401."""
        from django.urls import reverse
        url = reverse('recording-predict', kwargs={'recording_id': self.recording.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 401)

    def test_predict_forbidden_for_non_owner(self):
        """A different user (non-owner, non-clinician) must receive 403."""
        from django.urls import reverse
        self.client.force_authenticate(user=self.other_user)
        url = reverse('recording-predict', kwargs={'recording_id': self.recording.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 403)

    def test_predict_returns_503_no_sound_model(self):
        """With no active SOUND_CLASSIFICATION model, /predict/ must return 503."""
        from django.urls import reverse
        from ml_inference.models import ModelTaskType, ModelStatus, ModelVersion

        # Only a disease model is active — acoustic endpoint must still return 503
        ModelVersion.objects.create(
            version='api-test-disease-only-v1',
            task_type=ModelTaskType.DISEASE_CLASSIFICATION,
            status=ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-disease-cnn-v2.keras',
            verified_classes=['Healthy Control', 'Pneumonia', 'Wheezing Disorders'],
        )

        self.client.force_authenticate(user=self.user)
        url = reverse('recording-predict', kwargs={'recording_id': self.recording.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data['code'], 'MODEL_NOT_DEPLOYED')
        # Verify UNAVAILABLE record persisted
        from ml_inference.models import Prediction, PredictionStatus
        pred = Prediction.objects.filter(recording=self.recording).first()
        self.assertIsNotNone(pred)
        self.assertEqual(pred.prediction_status, PredictionStatus.UNAVAILABLE)

    def test_model_list_accessible_unauthenticated(self):
        """GET /models/ must be publicly accessible (IsAuthenticatedOrReadOnly)."""
        from django.urls import reverse
        url = reverse('model-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_model_list_includes_task_type(self):
        """Model list response must include task_type for each model."""
        from django.urls import reverse
        from ml_inference.models import ModelVersion, ModelTaskType, ModelStatus
        ModelVersion.objects.create(
            version='api-list-test-v1',
            task_type=ModelTaskType.SOUND_CLASSIFICATION,
            status=ModelStatus.DEVELOPMENT,
            verified_classes=['Normal', 'CAS', 'DAS', 'CAS & DAS'],
        )
        url = reverse('model-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response.data) >= 1)
        for model_data in response.data:
            self.assertIn('task_type', model_data)
            self.assertIn('task_type_display', model_data)

    def test_prediction_list_requires_authentication(self):
        """GET /predictions/ must require authentication."""
        from django.urls import reverse
        url = reverse('prediction-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)

    def test_prediction_list_returns_200_for_authenticated(self):
        """GET /predictions/ returns 200 for authenticated user."""
        from django.urls import reverse
        self.client.force_authenticate(user=self.user)
        url = reverse('prediction-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_prediction_history_for_recording_requires_auth(self):
        """GET /recordings/<id>/predictions/ must require authentication."""
        from django.urls import reverse
        url = reverse('recording-prediction-history', kwargs={'recording_id': self.recording.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)

    def test_prediction_response_includes_disclaimer(self):
        """
        Predictions returned from the history endpoint must include research_disclaimer.
        (mocked — does not invoke the real CNN)
        """
        from django.urls import reverse
        from ml_inference.models import Prediction, PredictionStatus, ModelVersion, ModelStatus, ModelTaskType

        mv = ModelVersion.objects.create(
            version='api-disclaimer-test-v1',
            task_type=ModelTaskType.SOUND_CLASSIFICATION,
            status=ModelStatus.DEVELOPMENT,
            verified_classes=['Normal', 'CAS', 'DAS', 'CAS & DAS'],
        )
        Prediction.objects.create(
            recording=self.recording,
            model_version=mv,
            predicted_class='Normal',
            probability_scores={'Normal': 0.75, 'CAS': 0.1, 'DAS': 0.1, 'CAS & DAS': 0.05},
            prediction_status=PredictionStatus.SUCCESS,
            inference_duration_ms=100,
        )

        self.client.force_authenticate(user=self.user)
        url = reverse('recording-prediction-history', kwargs={'recording_id': self.recording.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        pred_data = response.data[0]
        self.assertIn('research_disclaimer', pred_data)
        self.assertIn('not a medical diagnosis', pred_data['research_disclaimer'].lower())

    def test_disease_model_active_endpoint_separate_from_sound(self):
        """
        GET /models/disease/active/ must be completely separate from /models/active/.
        One returning 503 must not affect the other.
        """
        from django.urls import reverse
        from ml_inference.models import ModelVersion, ModelTaskType, ModelStatus

        # Only a sound model is active
        ModelVersion.objects.create(
            version='api-sep-sound-v1',
            task_type=ModelTaskType.SOUND_CLASSIFICATION,
            status=ModelStatus.ACTIVE,
            artifact_path='ml/models/saved_models/sprsound-cnn-v1-fold0.keras',
            verified_classes=['Normal', 'CAS', 'DAS', 'CAS & DAS'],
        )

        # Sound model endpoint → 200
        sound_response = self.client.get(reverse('model-active'))
        self.assertEqual(sound_response.status_code, 200)
        self.assertEqual(sound_response.data['task_type'], ModelTaskType.SOUND_CLASSIFICATION)

        # Disease model endpoint → 503 (no disease model deployed)
        disease_response = self.client.get(reverse('disease-model-active'))
        self.assertEqual(disease_response.status_code, 503)
        self.assertEqual(disease_response.data['code'], 'NO_ACTIVE_DISEASE_MODEL')

    def test_nonexistent_recording_predict_returns_404(self):
        """POST to /predict/ with invalid recording UUID must return 404."""
        import uuid
        from django.urls import reverse
        self.client.force_authenticate(user=self.user)
        fake_id = uuid.uuid4()
        url = reverse('recording-predict', kwargs={'recording_id': fake_id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 404)

    def test_repeated_predict_requests_persist_separate_records(self):
        """
        Multiple POST calls to /predict/ for the same recording each create
        a separate Prediction record (audit trail preserved).
        """
        from django.urls import reverse
        from ml_inference.models import Prediction
        self.client.force_authenticate(user=self.user)
        url = reverse('recording-predict', kwargs={'recording_id': self.recording.id})
        # Both calls return 503 (no model) but each creates its own record
        self.client.post(url)
        self.client.post(url)
        pred_count = Prediction.objects.filter(recording=self.recording).count()
        self.assertEqual(pred_count, 2, "Each inference request must persist its own Prediction record.")
