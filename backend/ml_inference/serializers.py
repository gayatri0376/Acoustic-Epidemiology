from rest_framework import serializers
from ml_inference.models import ModelVersion, Prediction


class ModelVersionSerializer(serializers.ModelSerializer):
    is_available = serializers.BooleanField(source='is_artifact_available', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'version',
            'architecture',
            'status',
            'status_display',
            'is_available',
            'verified_classes',
            'preprocessing_config',
            'evaluation_metadata',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'is_available', 'status_display', 'created_at', 'updated_at']


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
