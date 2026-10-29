import os
import uuid

from django.conf import settings
from django.db import models


def recording_upload_path(instance, filename):
    """Store under a random name so users cannot guess or overwrite file paths."""
    ext = os.path.splitext(filename)[1].lower()
    return f'recordings/patient_{instance.patient_id}/{uuid.uuid4().hex}{ext}'


class Recording(models.Model):
    class Status(models.TextChoices):
        UPLOADED = 'uploaded', 'Uploaded'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    patient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='recordings')
    audio_file = models.FileField(upload_to=recording_upload_path)
    original_filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=50)
    file_size = models.PositiveBigIntegerField()
    notes = models.TextField(blank=True, help_text='Symptoms or reason for the recording')
    recorded_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.UPLOADED, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Recording {self.pk} for {self.patient_id}'
