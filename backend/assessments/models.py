import uuid
from django.db import models
from django.contrib.auth.models import User


class ReviewStatus(models.TextChoices):
    IN_REVIEW = 'IN_REVIEW', 'In Review'
    CONFIRMED = 'CONFIRMED', 'Clinically Confirmed'
    DISCORDANT = 'DISCORDANT', 'Clinician Disagrees with AI Finding'
    INCONCLUSIVE = 'INCONCLUSIVE', 'Inconclusive / Artifact / Poor SNR'


class ReviewerConclusion(models.TextChoices):
    NORMAL = 'NORMAL', 'Normal Vesicular Breath Sounds'
    WHEEZING = 'WHEEZING', 'Wheezing Present (Airway Obstruction)'
    CRACKLES = 'CRACKLES', 'Crackles Present (Fluid / Alveolar Opening)'
    STRIDOR = 'STRIDOR', 'Stridor Present (Upper Airway Obstruction)'
    RHONCHI = 'RHONCHI', 'Rhonchi / Low-Pitched Wheeze'
    OTHER = 'OTHER', 'Other Respiratory Anomaly'


class Assessment(models.Model):
    """
    Human clinician assessment record linked to an AI prediction.
    Explicitly maintains separation between machine inference and professional clinical conclusions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prediction = models.ForeignKey(
        'ml_inference.Prediction',
        on_delete=models.CASCADE,
        related_name='assessments',
        db_index=True
    )
    reviewer = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assessments',
        db_index=True
    )
    review_status = models.CharField(
        max_length=32,
        choices=ReviewStatus.choices,
        default=ReviewStatus.IN_REVIEW,
        db_index=True
    )
    reviewer_conclusion = models.CharField(
        max_length=64,
        choices=ReviewerConclusion.choices,
        blank=True,
        help_text="Clinician's validated judgment regarding the acoustic sound characteristics."
    )
    diagnostic_agreement = models.BooleanField(
        null=True,
        blank=True,
        help_text="True if human conclusion agrees with AI predicted class, False if discordant."
    )
    clinical_notes = models.TextField(
        blank=True,
        help_text="Clinician's diagnostic observations and context without unnecessary identifiable patient data."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'assessments'
        verbose_name = 'Assessment'
        verbose_name_plural = 'Assessments'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['prediction', '-created_at']),
            models.Index(fields=['reviewer', '-created_at']),
            models.Index(fields=['review_status', '-created_at']),
        ]

    def __str__(self):
        reviewer_name = self.reviewer.username if self.reviewer else "Unassigned"
        return f"Assessment {self.id} by {reviewer_name} ({self.review_status})"
