from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient
from recordings.models import Recording, RecordingStatus
from ml_inference.models import Prediction, PredictionStatus
from assessments.models import Assessment, ReviewStatus, ReviewerConclusion
from accounts.models import UserRole


class AssessmentsApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.uploader = User.objects.create_user(username='uploader_nurse', password='Password123!')
        self.clinician = User.objects.create_user(username='pulmonologist_md', password='Password123!')
        self.clinician.profile.role = UserRole.CLINICIAN
        self.clinician.profile.save()

        self.recording = Recording.objects.create(
            original_filename='pediatric_wheeze.wav',
            file_format='wav',
            file_size_bytes=4096,
            recording_status=RecordingStatus.VALIDATED,
            uploader=self.uploader
        )
        self.prediction = Prediction.objects.create(
            recording=self.recording,
            prediction_status=PredictionStatus.UNAVAILABLE,
            error_details="Trained model pending."
        )

    def test_regular_user_cannot_submit_clinical_assessment(self):
        self.client.force_authenticate(user=self.uploader)
        url = reverse('prediction-assessment-create', kwargs={'prediction_id': self.prediction.id})
        data = {
            'review_status': ReviewStatus.CONFIRMED,
            'reviewer_conclusion': ReviewerConclusion.WHEEZING,
            'clinical_notes': 'Audible polyphonic expiratory wheezing noted.'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_clinician_can_submit_assessment(self):
        self.client.force_authenticate(user=self.clinician)
        url = reverse('prediction-assessment-create', kwargs={'prediction_id': self.prediction.id})
        data = {
            'review_status': ReviewStatus.CONFIRMED,
            'reviewer_conclusion': ReviewerConclusion.WHEEZING,
            'diagnostic_agreement': True,
            'clinical_notes': 'High-pitched musical wheeze localized to bilateral lower lobes.'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('assessment', response.data)
        assessment_data = response.data['assessment']
        self.assertEqual(assessment_data['reviewer_conclusion'], ReviewerConclusion.WHEEZING)
        self.assertEqual(assessment_data['review_status'], ReviewStatus.CONFIRMED)
        self.assertEqual(assessment_data['reviewer_username'], 'pulmonologist_md')

    def test_update_assessment_permissions(self):
        # Create assessment by clinician
        assessment = Assessment.objects.create(
            prediction=self.prediction,
            reviewer=self.clinician,
            review_status=ReviewStatus.IN_REVIEW,
            reviewer_conclusion=ReviewerConclusion.WHEEZING
        )

        # Uploader cannot edit review
        self.client.force_authenticate(user=self.uploader)
        detail_url = reverse('assessment-detail', kwargs={'pk': assessment.id})
        res_uploader = self.client.patch(detail_url, {'review_status': ReviewStatus.CONFIRMED}, format='json')
        self.assertEqual(res_uploader.status_code, status.HTTP_403_FORBIDDEN)

        # Authoring clinician can edit review
        self.client.force_authenticate(user=self.clinician)
        res_clinician = self.client.patch(detail_url, {'review_status': ReviewStatus.CONFIRMED}, format='json')
        self.assertEqual(res_clinician.status_code, status.HTTP_200_OK)
        assessment.refresh_from_db()
        self.assertEqual(assessment.review_status, ReviewStatus.CONFIRMED)
