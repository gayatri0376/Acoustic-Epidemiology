from rest_framework import serializers
from explanations.models import Explanation


class ExplanationSerializer(serializers.ModelSerializer):
    method_display = serializers.CharField(source='get_method_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    prediction_id = serializers.UUIDField(source='prediction.id', read_only=True)

    class Meta:
        model = Explanation
        fields = [
            'id',
            'prediction_id',
            'method',
            'method_display',
            'status',
            'status_display',
            'artifact_path',
            'structured_data',
            'error_details',
            'created_at',
        ]
        read_only_fields = [
            'id',
            'prediction_id',
            'status',
            'status_display',
            'method_display',
            'artifact_path',
            'structured_data',
            'error_details',
            'created_at',
        ]
