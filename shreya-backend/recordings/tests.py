import shutil
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings

from accounts.models import DoctorPatientAssignment, Role
from assessments.models import Assessment
from config.testing import BaseAPITestCase

TEMP_MEDIA = tempfile.mkdtemp()  # uploads from tests go here, not into your real media/ folder
WAV_BYTES = b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt ' + b'\x00' * 32


def wav(name='cough.wav', data=WAV_BYTES):
    return SimpleUploadedFile(name, data, content_type='audio/wav')


@override_settings(MEDIA_ROOT=TEMP_MEDIA, ML_MODEL_PATH='')
class RecordingTests(BaseAPITestCase):
    url = '/api/v1/recordings/'

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)

    def setUp(self):
        super().setUp()
        self.alice = self.make_user('alice@example.com')
        self.bob = self.make_user('bob@example.com')
        self.doctor = self.make_user('doc@example.com', Role.DOCTOR)
        self.other_doctor = self.make_user('doc2@example.com', Role.DOCTOR)
        self.admin = self.make_user('admin@example.com', Role.ADMIN)
        DoctorPatientAssignment.objects.create(doctor=self.doctor, patient=self.alice)

    def upload(self, user, file=None):
        self.client.force_authenticate(user=user)
        return self.client.post(self.url, {'audio_file': file or wav(), 'notes': 'dry cough'}, format='multipart')

    # ---- upload ----
    def test_upload_requires_login(self):
        self.assertEqual(self.client.post(self.url, {'audio_file': wav()}, format='multipart').status_code, 401)

    def test_patient_upload_creates_recording_and_honest_assessment(self):
        response = self.upload(self.alice)
        self.assertEqual(response.status_code, 201, response.content)
        self.assertEqual(response.data['status'], 'uploaded')
        self.assertEqual(response.data['patient'], self.alice.pk)
        assessment = Assessment.objects.get(recording_id=response.data['id'])
        self.assertEqual(assessment.status, 'not_configured')  # no model -> no fabricated prediction
        self.assertIsNone(assessment.confidence)
        self.assertEqual(assessment.result_label, '')

    def test_doctor_and_admin_cannot_upload(self):
        self.assertEqual(self.upload(self.doctor).status_code, 403)
        self.assertEqual(self.upload(self.admin).status_code, 403)

    # ---- validation ----
    def test_rejects_wrong_extension(self):
        self.assertEqual(self.upload(self.alice, wav('notes.txt')).status_code, 400)

    def test_rejects_content_that_does_not_match_extension(self):
        self.assertEqual(self.upload(self.alice, wav('fake.wav', b'MZ this is not audio at all')).status_code, 400)

    def test_rejects_empty_file(self):
        self.assertEqual(self.upload(self.alice, wav('empty.wav', b'')).status_code, 400)

    @override_settings(MAX_AUDIO_UPLOAD_MB=0)
    def test_rejects_file_over_size_limit(self):
        self.assertEqual(self.upload(self.alice).status_code, 400)

    # ---- access control ----
    def test_patient_cannot_access_another_patients_recording(self):
        recording_id = self.upload(self.alice).data['id']
        self.client.force_authenticate(user=self.bob)
        self.assertEqual(self.client.get(f'{self.url}{recording_id}/').status_code, 404)
        self.assertEqual(self.client.get(f'{self.url}{recording_id}/file/').status_code, 404)
        self.assertEqual(self.client.get(self.url).data['count'], 0)

    def test_owner_can_list_view_and_download(self):
        recording_id = self.upload(self.alice).data['id']
        self.client.force_authenticate(user=self.alice)
        self.assertEqual(self.client.get(self.url).data['count'], 1)
        self.assertEqual(self.client.get(f'{self.url}{recording_id}/').status_code, 200)
        download = self.client.get(f'{self.url}{recording_id}/file/')
        self.assertEqual(download.status_code, 200)
        self.assertEqual(b''.join(download.streaming_content), WAV_BYTES)

    def test_doctor_sees_only_assigned_patients(self):
        recording_id = self.upload(self.alice).data['id']
        self.client.force_authenticate(user=self.doctor)
        self.assertEqual(self.client.get(f'{self.url}{recording_id}/').status_code, 200)
        self.client.force_authenticate(user=self.other_doctor)
        self.assertEqual(self.client.get(f'{self.url}{recording_id}/').status_code, 404)
        self.assertEqual(self.client.get(self.url).data['count'], 0)

    def test_admin_sees_everything(self):
        self.upload(self.alice)
        self.upload(self.bob)
        self.client.force_authenticate(user=self.admin)
        self.assertEqual(self.client.get(self.url).data['count'], 2)
