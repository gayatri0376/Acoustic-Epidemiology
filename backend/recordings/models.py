import os
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_delete
from django.dispatch import receiver


def recording_upload_path(instance, filename):
    """
    Constructs a safe, non-traversable storage path keyed by the record UUID.
    """
    ext = os.path.splitext(filename)[1].lower()
    return f"recordings/{instance.id}{ext}"


class RecordingStatus(models.TextChoices):
    UPLOADED = 'UPLOADED', 'Uploaded'
    PROCESSING = 'PROCESSING', 'Processing'
    VALIDATED = 'VALIDATED', 'Validated'
    FAILED = 'FAILED', 'Validation Failed'


class AnatomicalLocation(models.TextChoices):
    TRACHEA = 'TRACHEA', 'Trachea'
    ANTERIOR_RIGHT_UPPER = 'ANTERIOR_RIGHT_UPPER', 'Anterior Right Upper Lobe'
    ANTERIOR_LEFT_UPPER = 'ANTERIOR_LEFT_UPPER', 'Anterior Left Upper Lobe'
    POSTERIOR_RIGHT_LOWER = 'POSTERIOR_RIGHT_LOWER', 'Posterior Right Lower Lobe'
    POSTERIOR_LEFT_LOWER = 'POSTERIOR_LEFT_LOWER', 'Posterior Left Lower Lobe'
    LATERAL_RIGHT = 'LATERAL_RIGHT', 'Lateral Right Chest'
    LATERAL_LEFT = 'LATERAL_LEFT', 'Lateral Left Chest'
    UNSPECIFIED = 'UNSPECIFIED', 'Unspecified Chest Location'


class PatientAgeGroup(models.TextChoices):
    NEONATE = 'NEONATE', 'Neonate (0-28 days)'
    INFANT = 'INFANT', 'Infant (1-12 months)'
    TODDLER = 'TODDLER', 'Toddler (1-3 years)'
    PRESCHOOL = 'PRESCHOOL', 'Preschool (3-5 years)'
    SCHOOL_AGE = 'SCHOOL_AGE', 'School Age (6-12 years)'
    ADOLESCENT = 'ADOLESCENT', 'Adolescent (13-18 years)'
    UNKNOWN = 'UNKNOWN', 'Unknown / Not Stated'


class Recording(models.Model):
    """
    Pediatric respiratory auscultation recording entity.
    Maintains validated audio files and de-identified clinical metadata.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audio_file = models.FileField(
        upload_to=recording_upload_path,
        help_text="Path to stored audio recording in media storage."
    )
    original_filename = models.CharField(max_length=255)
    file_format = models.CharField(max_length=16, db_index=True)
    file_size_bytes = models.PositiveBigIntegerField()
    duration_seconds = models.FloatField(null=True, blank=True)
    sample_rate = models.PositiveIntegerField(null=True, blank=True)
    channels = models.PositiveSmallIntegerField(default=1)
    recording_status = models.CharField(
        max_length=32,
        choices=RecordingStatus.choices,
        default=RecordingStatus.VALIDATED,
        db_index=True
    )

    # De-identified clinical metadata
    anatomical_location = models.CharField(
        max_length=64,
        choices=AnatomicalLocation.choices,
        default=AnatomicalLocation.UNSPECIFIED,
        db_index=True
    )

    patient_age_group = models.CharField(
        max_length=32,
        choices=PatientAgeGroup.choices,
        default=PatientAgeGroup.UNKNOWN
    )
    recording_device = models.CharField(
        max_length=128, 
        blank=True, 
        help_text="Digital stethoscope model or recording sensor."
    )
    patient_pseudo_id = models.CharField(
        max_length=64, 
        blank=True, 
        help_text="Anonymized research identifier (e.g. SUBJ-0042)."
    )

    uploader = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='recordings',
        db_index=True
    )
    validation_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'recordings'
        verbose_name = 'Recording'
        verbose_name_plural = 'Recordings'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['uploader', '-created_at']),
            models.Index(fields=['recording_status', '-created_at']),
        ]

    def __str__(self):
        return f"Recording {self.id} ({self.original_filename}) - {self.recording_status}"

    @property
    def file_url(self):
        """Safe relative URL for media retrieval, avoiding filesystem path exposure."""
        if self.audio_file:
            return self.audio_file.url
        return None


@receiver(post_delete, sender=Recording)
def auto_delete_file_on_recording_delete(sender, instance, **kwargs):
    """
    Deletes audio file from disk storage when corresponding database record is deleted.
    """
    if instance.audio_file:
        if os.path.isfile(instance.audio_file.path):
            try:
                os.remove(instance.audio_file.path)
            except OSError:
                pass
