import io
import wave
import struct
from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APIClient
from recordings.models import Recording, RecordingStatus, AnatomicalLocation, PatientAgeGroup
from accounts.models import UserRole


def generate_valid_wav_file(filename="test_respiratory.wav", duration=1.0, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        n_samples = int(sample_rate * duration)
        raw_data = struct.pack('<' + ('h' * n_samples), *([0] * n_samples))
        wav.writeframes(raw_data)
    buf.seek(0)
    return SimpleUploadedFile(
        name=filename,
        content=buf.read(),
        content_type='audio/wav'
    )


class RecordingsApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.uploader = User.objects.create_user(
            username='dr_uploader',
            password='Password123!'
        )
        self.other_user = User.objects.create_user(
            username='other_user',
            password='Password123!'
        )
        self.clinician = User.objects.create_user(
            username='pulmonologist',
            password='Password123!'
        )
        self.clinician.profile.role = UserRole.CLINICIAN
        self.clinician.profile.save()

    def test_upload_valid_wav_audio(self):
        self.client.force_authenticate(user=self.uploader)
        url = reverse('recording-list-create')
        audio_file = generate_valid_wav_file(duration=1.5, sample_rate=16000)

        data = {
            'audio_file': audio_file,
            'anatomical_location': AnatomicalLocation.ANTERIOR_RIGHT_UPPER,
            'patient_age_group': PatientAgeGroup.INFANT,
            'recording_device': 'Littmann 3200 Digital Stethoscope',
            'patient_pseudo_id': 'SUBJ-101'
        }
        response = self.client.post(url, data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('recording', response.data)
        rec_data = response.data['recording']
        self.assertEqual(rec_data['file_format'], 'wav')
        self.assertEqual(rec_data['sample_rate'], 16000)
        self.assertEqual(rec_data['channels'], 1)
        self.assertAlmostEqual(rec_data['duration_seconds'], 1.5, places=1)
        self.assertEqual(rec_data['anatomical_location'], AnatomicalLocation.ANTERIOR_RIGHT_UPPER)

    def test_reject_invalid_audio_signature(self):
        self.client.force_authenticate(user=self.uploader)
        url = reverse('recording-list-create')
        fake_audio = SimpleUploadedFile(
            name="fake_sound.wav",
            content=b"THIS IS NOT A REAL AUDIO FILE",
            content_type="audio/wav"
        )
        data = {'audio_file': fake_audio}
        response = self.client.post(url, data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('audio_file', response.data)

    def test_recording_list_isolation(self):
        # Create recording owned by uploader
        self.client.force_authenticate(user=self.uploader)
        audio = generate_valid_wav_file()
        self.client.post(reverse('recording-list-create'), {'audio_file': audio}, format='multipart')

        # Other regular user shouldn't see it
        self.client.force_authenticate(user=self.other_user)
        list_url = reverse('recording-list-create')
        res_other = self.client.get(list_url)
        self.assertEqual(len(res_other.data['results'] if 'results' in res_other.data else res_other.data), 0)

        # Clinician should be able to view all recordings
        self.client.force_authenticate(user=self.clinician)
        res_clinician = self.client.get(list_url)
        results = res_clinician.data['results'] if 'results' in res_clinician.data else res_clinician.data
        self.assertEqual(len(results), 1)

    def test_unauthenticated_user_cannot_access_recordings(self):
        url = reverse('recording-list-create')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_reject_unsupported_file_extension(self):
        self.client.force_authenticate(user=self.uploader)
        url = reverse('recording-list-create')
        script_file = SimpleUploadedFile(
            name="exploit.sh",
            content=b"#!/bin/bash\necho hello",
            content_type="text/x-shellscript"
        )
        response = self.client.post(url, {'audio_file': script_file}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Unsupported file extension", str(response.data))

    def test_reject_oversized_audio_file(self):
        from recordings.validators import inspect_and_validate_audio
        from django.core.exceptions import ValidationError
        oversized_file = io.BytesIO(b"RIFF" + b"\x00" * 36)
        oversized_file.name = "huge_file.wav"
        oversized_file.size = 26 * 1024 * 1024  # 26 MB
        with self.assertRaises(ValidationError) as ctx:
            inspect_and_validate_audio(oversized_file)
        self.assertIn("exceeds maximum allowed limit", str(ctx.exception))


    def test_unauthorized_user_cannot_access_recording_detail(self):
        self.client.force_authenticate(user=self.uploader)
        audio = generate_valid_wav_file()
        create_res = self.client.post(reverse('recording-list-create'), {'audio_file': audio}, format='multipart')
        rec_id = create_res.data['recording']['id']

        # Unauthorized other user tries to GET recording detail
        self.client.force_authenticate(user=self.other_user)
        detail_url = reverse('recording-detail', kwargs={'pk': rec_id})
        res = self.client.get(detail_url)
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)

    def test_delete_recording_permission(self):

        self.client.force_authenticate(user=self.uploader)
        audio = generate_valid_wav_file()
        create_res = self.client.post(reverse('recording-list-create'), {'audio_file': audio}, format='multipart')
        rec_id = create_res.data['recording']['id']

        # Other user cannot delete
        self.client.force_authenticate(user=self.other_user)
        del_url = reverse('recording-detail', kwargs={'pk': rec_id})
        res_del = self.client.delete(del_url)
        self.assertEqual(res_del.status_code, status.HTTP_403_FORBIDDEN)

        # Uploader can delete
        self.client.force_authenticate(user=self.uploader)
        res_uploader_del = self.client.delete(del_url)
        self.assertEqual(res_uploader_del.status_code, status.HTTP_200_OK)
        self.assertFalse(Recording.objects.filter(pk=rec_id).exists())


