from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient
from recordings.models import Recording, RecordingStatus
from ml_inference.models import Prediction, PredictionStatus
from explanations.models import Explanation, ExplanationStatus, ExplanationMethod


class ExplanationsApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='dr_gradcam', password='Password123!')
        self.recording = Recording.objects.create(
            original_filename='auscultation.wav',
            file_format='wav',
            file_size_bytes=8192,
            duration_seconds=1.0,
            sample_rate=16000,
            recording_status=RecordingStatus.VALIDATED,
            uploader=self.user
        )
        self.prediction = Prediction.objects.create(
            recording=self.recording,
            prediction_status=PredictionStatus.UNAVAILABLE,
            error_details="No trained weights."
        )

    def test_gradcam_returns_503_when_model_weights_missing(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('prediction-explanations', kwargs={'prediction_id': self.prediction.id})
        response = self.client.post(url, {'method': ExplanationMethod.GRAD_CAM})
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['code'], 'GRADIENT_MODEL_UNAVAILABLE')

        # Verify record in database marked UNSUPPORTED
        explanation = Explanation.objects.filter(prediction=self.prediction).first()
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation.status, ExplanationStatus.UNSUPPORTED)

    def test_list_explanations_for_prediction(self):
        self.client.force_authenticate(user=self.user)
        Explanation.objects.create(
            prediction=self.prediction,
            method=ExplanationMethod.GRAD_CAM,
            status=ExplanationStatus.UNSUPPORTED,
            error_details="Weights missing."
        )
        url = reverse('prediction-explanations', kwargs={'prediction_id': self.prediction.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['status'], ExplanationStatus.UNSUPPORTED)
