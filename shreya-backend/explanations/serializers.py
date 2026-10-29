from rest_framework import serializers

from accounts.access import visible_to
from assessments.models import Assessment

from .models import Explanation


class ExplanationSerializer(serializers.ModelSerializer):
    author_name = serializers.CharField(source='author.full_name', read_only=True)

    class Meta:
        model = Explanation
        fields = ['id', 'assessment', 'author', 'author_name', 'text', 'created_at']
        read_only_fields = ['id', 'author', 'author_name', 'created_at']

    def validate_assessment(self, assessment):
        # The author may only explain assessments they are allowed to see.
        user = self.context['request'].user
        if not visible_to(user, Assessment.objects.filter(pk=assessment.pk)).exists():
            raise serializers.ValidationError('Assessment not found.')
        return assessment
