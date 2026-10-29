import os
import uuid
from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError


class ModelStatus(models.TextChoices):
    DEVELOPMENT = 'DEVELOPMENT', 'Under Development'
    ACTIVE = 'ACTIVE', 'Active / In Production'
    RETIRED = 'RETIRED', 'Retired / Deprecated'


class ModelTaskType(models.TextChoices):
    SOUND_CLASSIFICATION = 'SOUND_CLASSIFICATION', 'Respiratory Sound Classification'
    DISEASE_CLASSIFICATION = 'DISEASE_CLASSIFICATION', 'Respiratory Disease Classification'


class PredictionStatus(models.TextChoices):
    PENDING = 'PENDING', 'Pending Inference'
    SUCCESS = 'SUCCESS', 'Inference Successful'
    FAILED = 'FAILED', 'Inference Failed'
    UNAVAILABLE = 'UNAVAILABLE', 'Model Artifact Unavailable'


class ModelVersion(models.Model):
    """
    Registry for trained convolutional neural network model versions.
    Requires verified label sets, preprocessing definitions, and artifact checks.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.CharField(
        max_length=64, 
        unique=True, 
        help_text="Unique model version identifier (e.g., 'pediatric-cnn-v1.0.0')."
    )
    architecture = models.CharField(
        max_length=128, 
        default="2D-CNN-Spectrogram",
        help_text="Base neural network architecture descriptor."
    )
    artifact_path = models.CharField(
        max_length=512, 
        blank=True,
        help_text="Filesystem path to model weights (e.g. .h5, .keras, or SavedModel directory)."
    )
    verified_classes = models.JSONField(
        default=list,
        help_text="Exact ordered list of verified respiratory disease class labels."
    )
    preprocessing_config = models.JSONField(
        default=dict,
        help_text="Acoustic feature settings: sample_rate, n_mels, hop_length, n_fft, bandpass_hz."
    )
    task_type = models.CharField(
        max_length=32,
        choices=ModelTaskType.choices,
        default=ModelTaskType.SOUND_CLASSIFICATION,
        db_index=True,
        help_text='The prediction task this model was trained to perform.'
    )
    status = models.CharField(
        max_length=32,
        choices=ModelStatus.choices,
        default=ModelStatus.DEVELOPMENT,
        db_index=True
    )
    evaluation_metadata = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Verified validation accuracy, confusion matrix, or evaluation metrics."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'model_versions'
        verbose_name = 'Model Version'
        verbose_name_plural = 'Model Versions'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.version} ({self.get_status_display()})"

    def clean(self):
        super().clean()
        # If marked active, verify that artifact exists
        if self.status == ModelStatus.ACTIVE:
            if not self.artifact_path or not self.is_artifact_available():
                raise ValidationError({
                    'status': "A model cannot be marked ACTIVE without an existing, verified artifact on the filesystem."
                })
            if not self.verified_classes:
                raise ValidationError({
                    'verified_classes': "A model cannot be marked ACTIVE without verified class labels."
                })

    def is_artifact_available(self):
        """Checks if the artifact path actually exists on the filesystem."""
        if not self.artifact_path:
            return False
        
        # Check absolute path or path relative to project root
        if os.path.isabs(self.artifact_path):
            return os.path.exists(self.artifact_path)
        
        project_root = settings.BASE_DIR.parent
        return os.path.exists(os.path.join(project_root, self.artifact_path))


class Prediction(models.Model):
    """
    Prediction record capturing CNN model outputs, class probabilities,
    and execution latency for an audio recording.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recording = models.ForeignKey(
        'recordings.Recording', 
        on_delete=models.CASCADE, 
        related_name='predictions',
        db_index=True
    )
    model_version = models.ForeignKey(
        ModelVersion, 
        on_delete=models.PROTECT, 
        null=True, 
        blank=True, 
        related_name='predictions',
        db_index=True
    )
    predicted_class = models.CharField(
        max_length=128, 
        blank=True, 
        help_text="Highest-probability respiratory sound class label."
    )
    probability_scores = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Dictionary mapping verified classes to softmax probability values."
    )
    prediction_status = models.CharField(
        max_length=32,
        choices=PredictionStatus.choices,
        default=PredictionStatus.PENDING,
        db_index=True
    )
    error_details = models.TextField(
        blank=True,
        help_text="Safe technical error diagnostic if prediction failed or was unavailable."
    )
    inference_duration_ms = models.IntegerField(
        null=True, 
        blank=True,
        help_text="Time taken for audio preprocessing and model forward pass in milliseconds."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'predictions'
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recording', '-created_at']),
            models.Index(fields=['prediction_status', '-created_at']),
        ]

    def __str__(self):
        return f"Prediction {self.id} for Recording {self.recording_id} ({self.prediction_status})"

    def clean(self):
        super().clean()
        if self.prediction_status == PredictionStatus.SUCCESS:
            if not self.predicted_class:
                raise ValidationError({'predicted_class': "Successful prediction must define predicted_class."})
            if not self.probability_scores:
                raise ValidationError({'probability_scores': "Successful prediction must contain probability scores."})
            if not self.model_version:
                raise ValidationError({'model_version': "Successful prediction must link to a valid ModelVersion."})
        elif self.prediction_status in (PredictionStatus.FAILED, PredictionStatus.UNAVAILABLE):
            if not self.error_details:
                raise ValidationError({'error_details': "Failed or unavailable prediction must state error details."})


class DiseasePrediction(models.Model):
    """
    Disease prediction record from the separate SPRSound-based 3-class disease model
    (Healthy Control / Pneumonia / Wheezing Disorders).

    IMPORTANT: This prediction is derived from a patient-level multi-site pooling model trained
    on SPRSound patient-summary diagnoses (hospital admission/discharge). It is a research
    screening output and MUST NOT be used as a clinical diagnosis.

    Dataset: SPRSound CC BY 4.0 — 100% pediatric patients (max age 16.2 years).
    Label provenance: Hospital admission/discharge diagnoses from SJTU patient summaries.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recording = models.ForeignKey(
        'recordings.Recording',
        on_delete=models.CASCADE,
        related_name='disease_predictions',
        db_index=True
    )
    model_version = models.ForeignKey(
        ModelVersion,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name='disease_predictions',
        db_index=True,
        limit_choices_to={'task_type': ModelTaskType.DISEASE_CLASSIFICATION}
    )
    predicted_disease = models.CharField(
        max_length=128,
        blank=True,
        help_text=(
            "Highest-probability disease screening class. "
            "Valid values: 'Healthy Control', 'Pneumonia', 'Wheezing Disorders'. "
            "This is a research output; not a clinical diagnosis."
        )
    )
    probability_scores = models.JSONField(
        default=dict,
        blank=True,
        help_text="Dictionary mapping verified disease classes to softmax probability values."
    )
    prediction_status = models.CharField(
        max_length=32,
        choices=PredictionStatus.choices,
        default=PredictionStatus.PENDING,
        db_index=True
    )
    error_details = models.TextField(
        blank=True,
        help_text="Safe technical error diagnostic if disease prediction failed or was unavailable."
    )
    inference_duration_ms = models.IntegerField(
        null=True,
        blank=True,
        help_text="Time taken for audio preprocessing and model forward pass in milliseconds."
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'disease_predictions'
        verbose_name = 'Disease Prediction'
        verbose_name_plural = 'Disease Predictions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recording', '-created_at']),
            models.Index(fields=['prediction_status', '-created_at']),
        ]

    def __str__(self):
        return f"DiseasePrediction {self.id} for Recording {self.recording_id} ({self.prediction_status})"

    def clean(self):
        super().clean()
        if self.prediction_status == PredictionStatus.SUCCESS:
            if not self.predicted_disease:
                raise ValidationError({'predicted_disease': "Successful disease prediction must define predicted_disease."})
            if not self.probability_scores:
                raise ValidationError({'probability_scores': "Successful disease prediction must contain probability scores."})
            if not self.model_version:
                raise ValidationError({'model_version': "Successful disease prediction must link to a valid ModelVersion."})
        elif self.prediction_status in (PredictionStatus.FAILED, PredictionStatus.UNAVAILABLE):
            if not self.error_details:
                raise ValidationError({'error_details': "Failed or unavailable disease prediction must state error details."})
