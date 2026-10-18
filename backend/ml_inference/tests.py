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
        temp_artifact = self._create_temp_artifact("valid_name.keras")
        rel_path = str(temp_artifact.relative_to(self.project_root))

        # Because tensorflow is not installed in the test environment, this must fail gracefully
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


