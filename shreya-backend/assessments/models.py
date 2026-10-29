from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Assessment(models.Model):
    class Status(models.TextChoices):
        NOT_CONFIGURED = 'not_configured', 'Model not configured'
        COMPLETED = 'completed', 'Completed (model output)'
        FAILED = 'failed', 'Failed'
        REVIEWED = 'reviewed', 'Reviewed by clinician'

    recording = models.ForeignKey('recordings.Recording', on_delete=models.CASCADE, related_name='assessments')
    patient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='assessments')
    status = models.CharField(max_length=15, choices=Status.choices, default=Status.NOT_CONFIGURED, db_index=True)
    result_label = models.CharField(max_length=100, blank=True)  # empty unless a real model produced it
    confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    summary = models.TextField(blank=True)
    doctor_notes = models.TextField(blank=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='+')
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Assessment {self.pk} ({self.status})'
