from rest_framework import serializers
from ml_inference.models import ModelVersion, Prediction, DiseasePrediction
from ml_inference.disease_service import DISEASE_CLASS_DESCRIPTIONS, RESEARCH_DISCLAIMER


class ModelVersionSerializer(serializers.ModelSerializer):
    is_available = serializers.BooleanField(source='is_artifact_available', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    task_type_display = serializers.CharField(source='get_task_type_display', read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'version',
            'architecture',
            'task_type',
            'task_type_display',
            'status',
            'status_display',
            'is_available',
            'verified_classes',
            'preprocessing_config',
            'evaluation_metadata',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'is_available', 'status_display', 'task_type_display', 'created_at', 'updated_at']


class PredictionSerializer(serializers.ModelSerializer):
    recording_id = serializers.UUIDField(source='recording.id', read_only=True)
    recording_filename = serializers.CharField(source='recording.original_filename', read_only=True)
    model_version_str = serializers.CharField(source='model_version.version', read_only=True, default=None)
    status_display = serializers.CharField(source='get_prediction_status_display', read_only=True)

    class Meta:
        model = Prediction
        fields = [
            'id',
            'recording_id',
            'recording_filename',
            'model_version',
            'model_version_str',
            'predicted_class',
            'probability_scores',
            'prediction_status',
            'status_display',
            'error_details',
            'inference_duration_ms',
            'created_at',
        ]
        read_only_fields = [
            'id',
            'recording_id',
            'recording_filename',
            'model_version',
            'model_version_str',
            'predicted_class',
            'probability_scores',
            'prediction_status',
            'status_display',
            'error_details',
            'inference_duration_ms',
            'created_at',
        ]


class DiseasePredictionSerializer(serializers.ModelSerializer):
    """
    Serializer for disease model predictions.
    Enriches the response with clinical context and mandatory research disclaimer.
    """
    recording_id = serializers.UUIDField(source='recording.id', read_only=True)
    recording_filename = serializers.CharField(source='recording.original_filename', read_only=True)
    model_version_str = serializers.CharField(source='model_version.version', read_only=True, default=None)
    status_display = serializers.CharField(source='get_prediction_status_display', read_only=True)
    # Computed fields enriching the response with educational context
    class_description = serializers.SerializerMethodField()
    research_disclaimer = serializers.SerializerMethodField()
    dataset_info = serializers.SerializerMethodField()

    class Meta:
        model = DiseasePrediction
        fields = [
            'id',
            'recording_id',
            'recording_filename',
            'model_version',
            'model_version_str',
            'predicted_disease',
            'probability_scores',
            'prediction_status',
            'status_display',
            'error_details',
            'inference_duration_ms',
            'class_description',
            'research_disclaimer',
            'dataset_info',
            'created_at',
        ]
        read_only_fields = [
            'id', 'recording_id', 'recording_filename', 'model_version',
            'model_version_str', 'predicted_disease', 'probability_scores',
            'prediction_status', 'status_display', 'error_details',
            'inference_duration_ms', 'class_description', 'research_disclaimer',
            'dataset_info', 'created_at',
        ]

    def get_class_description(self, obj):
        """Returns the human-readable description for the predicted disease class."""
        if obj.predicted_disease and obj.prediction_status == 'SUCCESS':
            return DISEASE_CLASS_DESCRIPTIONS.get(obj.predicted_disease, "")
        return ""

    def get_research_disclaimer(self, obj):
        """Always returns the mandatory research safety disclaimer."""
        return RESEARCH_DISCLAIMER

    def get_dataset_info(self, obj):
        """Returns provenance metadata for the disease model's training dataset."""
        return {
            "dataset": "SPRSound (SJTU Paediatric Respiratory Sound Database)",
            "license": "CC BY 4.0",
            "population": "Pediatric patients (0–16.2 years, 100% under 18)",
            "disease_classes": [
                "Healthy Control — control-group patients, no confirmed acute respiratory diagnosis",
                "Pneumonia — non-severe and severe pneumonia (merged from hospital summaries)",
                "Wheezing Disorders — Asthma + Bronchitis merged (both produce wheezing; "
                "dataset cannot acoustically distinguish them)",
            ],
            "label_provenance": (
                "Patient-level hospital admission/discharge diagnoses from SJTU patient summaries. "
                "Labels are NOT recording-level acoustic annotations."
            ),
            "label_mapping_note": (
                "Asthma and Bronchitis are merged into 'Wheezing Disorders' because both "
                "diagnoses produce wheezing and the SPRSound dataset lacks sufficient acoustic "
                "evidence to separate them. Do NOT infer a specific Asthma or Bronchitis "
                "diagnosis from this output."
            ),
            "citation": (
                "Q. Zhang et al., 'SPRSound: Open-Source SJTU Paediatric Respiratory Sound "
                "Database', IEEE TBioCAS, vol. 16, no. 5, pp. 867–881, Oct. 2022."
            ),
        }
