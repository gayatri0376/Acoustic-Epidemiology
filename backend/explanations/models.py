import uuid
from django.db import models


class ExplanationMethod(models.TextChoices):
    GRAD_CAM = 'GRAD_CAM', 'Gradient-weighted Class Activation Mapping (Grad-CAM)'
    MEL_SPECTROGRAM_SALIENCY = 'MEL_SPECTROGRAM_SALIENCY', 'Mel-Spectrogram Saliency Map'


class ExplanationStatus(models.TextChoices):
    PENDING = 'PENDING', 'Pending Generation'
    GENERATED = 'GENERATED', 'Explanation Generated'
    UNSUPPORTED = 'UNSUPPORTED', 'Unsupported by Model Architecture'
    FAILED = 'FAILED', 'Generation Failed'


class Explanation(models.Model):
    """
    Model explainability record linking visual or structured saliency artifacts
    (such as Grad-CAM heatmaps over Log Mel-Spectrograms) to a model prediction.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prediction = models.ForeignKey(
        'ml_inference.Prediction',
        on_delete=models.CASCADE,
        related_name='explanations',
        db_index=True
    )
    method = models.CharField(
        max_length=64,
        choices=ExplanationMethod.choices,
        default=ExplanationMethod.GRAD_CAM,
        help_text="XAI algorithm utilized for acoustic model interpretability."
    )
    status = models.CharField(
        max_length=32,
        choices=ExplanationStatus.choices,
        default=ExplanationStatus.PENDING,
        db_index=True
    )
    artifact_path = models.CharField(
        max_length=512,
        blank=True,
        help_text="Relative storage path to generated heatmap overlay image."
    )
    structured_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Frequency band activations, high-saliency time windows (seconds), and acoustic energy indices."
    )
    error_details = models.TextField(
        blank=True,
        help_text="Diagnostic explanation if the method is unsupported or failed."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'explanations'
        verbose_name = 'Explanation'
        verbose_name_plural = 'Explanations'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['prediction', '-created_at']),
            models.Index(fields=['status', '-created_at']),
        ]

    def __str__(self):
        return f"{self.get_method_display()} for Prediction {self.prediction_id} ({self.status})"
