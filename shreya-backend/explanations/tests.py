from accounts.models import DoctorPatientAssignment, Role
from assessments.models import Assessment
from config.testing import BaseAPITestCase
from explanations.models import Explanation
from recordings.models import Recording


class ExplanationTests(BaseAPITestCase):
    url = '/api/v1/explanations/'

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
        self.assessment = Assessment.objects.create(recording=recording, patient=self.alice)

    def test_assigned_doctor_can_create(self):
        self.client.force_authenticate(user=self.doctor)
        response = self.client.post(self.url, {'assessment': self.assessment.pk, 'text': 'Explanation'}, format='json')
        self.assertEqual(response.status_code, 201, response.content)
        self.assertEqual(response.data['author'], self.doctor.pk)

    def test_unassigned_doctor_cannot_create(self):
        self.client.force_authenticate(user=self.other_doctor)
        response = self.client.post(self.url, {'assessment': self.assessment.pk, 'text': 'x'}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_patient_cannot_create(self):
        self.client.force_authenticate(user=self.alice)
        self.assertEqual(self.client.post(self.url, {'assessment': self.assessment.pk, 'text': 'x'}, format='json').status_code, 403)

    def test_patient_sees_only_own_explanations(self):
        Explanation.objects.create(assessment=self.assessment, author=self.doctor, text='For Alice')
        self.client.force_authenticate(user=self.alice)
        self.assertEqual(self.client.get(self.url).data['count'], 1)
        self.client.force_authenticate(user=self.bob)
        self.assertEqual(self.client.get(self.url).data['count'], 0)

    def test_admin_can_create(self):
        self.client.force_authenticate(user=self.admin)
        self.assertEqual(self.client.post(self.url, {'assessment': self.assessment.pk, 'text': 'x'}, format='json').status_code, 201)
