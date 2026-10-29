from accounts.models import DoctorPatientAssignment, Role
from assessments.models import Assessment
from config.testing import BaseAPITestCase
from recordings.models import Recording


class AssessmentTests(BaseAPITestCase):
    def setUp(self):
        super().setUp()
        self.alice = self.make_user('alice@example.com')
        self.bob = self.make_user('bob@example.com')
        self.doctor = self.make_user('doc@example.com', Role.DOCTOR)
        self.other_doctor = self.make_user('doc2@example.com', Role.DOCTOR)
        self.admin = self.make_user('admin@example.com', Role.ADMIN)
        DoctorPatientAssignment.objects.create(doctor=self.doctor, patient=self.alice)
        recording = Recording.objects.create(patient=self.alice, audio_file='recordings/x.wav',
                                             original_filename='x.wav', content_type='audio/wav', file_size=10)
        self.assessment = Assessment.objects.create(recording=recording, patient=self.alice, status='not_configured')
        self.url = f'/api/v1/assessments/{self.assessment.pk}/'

    def test_patient_reads_own_assessment_with_disclaimer(self):
        self.client.force_authenticate(user=self.alice)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('not a clinically validated', response.data['disclaimer'])

    def test_other_patient_gets_404(self):
        self.client.force_authenticate(user=self.bob)
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.client.get('/api/v1/assessments/').data['count'], 0)

    def test_patient_cannot_review(self):
        self.client.force_authenticate(user=self.alice)
        self.assertEqual(self.client.patch(self.url, {'doctor_notes': 'x'}, format='json').status_code, 403)

    def test_assigned_doctor_can_review_but_not_change_results(self):
        self.client.force_authenticate(user=self.doctor)
        response = self.client.patch(self.url, {'doctor_notes': 'Reviewed.', 'confidence': 0.99, 'result_label': 'x'}, format='json')
        self.assertEqual(response.status_code, 200, response.content)
        self.assessment.refresh_from_db()
        self.assertEqual(self.assessment.doctor_notes, 'Reviewed.')
        self.assertEqual(self.assessment.status, 'reviewed')
        self.assertEqual(self.assessment.reviewed_by_id, self.doctor.pk)
        self.assertIsNone(self.assessment.confidence)       # read-only: cannot be edited via API
        self.assertEqual(self.assessment.result_label, '')

    def test_unassigned_doctor_gets_404(self):
        self.client.force_authenticate(user=self.other_doctor)
        self.assertEqual(self.client.patch(self.url, {'doctor_notes': 'x'}, format='json').status_code, 404)

    def test_admin_can_review(self):
        self.client.force_authenticate(user=self.admin)
        self.assertEqual(self.client.patch(self.url, {'doctor_notes': 'ok'}, format='json').status_code, 200)
