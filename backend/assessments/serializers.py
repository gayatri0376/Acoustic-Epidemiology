from rest_framework import serializers
from assessments.models import Assessment, ReviewStatus, ReviewerConclusion


class AssessmentSerializer(serializers.ModelSerializer):
    reviewer_username = serializers.CharField(source='reviewer.username', read_only=True)
    review_status_display = serializers.CharField(source='get_review_status_display', read_only=True)
    reviewer_conclusion_display = serializers.CharField(source='get_reviewer_conclusion_display', read_only=True)
    prediction_id = serializers.UUIDField(source='prediction.id', read_only=True)

    class Meta:
        model = Assessment
        fields = [
            'id',
            'prediction_id',
            'reviewer',
            'reviewer_username',
            'review_status',
            'review_status_display',
            'reviewer_conclusion',
            'reviewer_conclusion_display',
            'diagnostic_agreement',
            'clinical_notes',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'prediction_id',
            'reviewer',
            'reviewer_username',
            'review_status_display',
            'reviewer_conclusion_display',
            'created_at',
            'updated_at',
        ]


class AssessmentCreateUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Assessment
        fields = [
            'review_status',
            'reviewer_conclusion',
            'diagnostic_agreement',
            'clinical_notes',
        ]

    def create(self, validated_data):
        prediction = self.context.get('prediction')
        request = self.context.get('request')
        reviewer = request.user if request and request.user.is_authenticated else None

        assessment = Assessment.objects.create(
            prediction=prediction,
            reviewer=reviewer,
            **validated_data
        )
        return assessment
