from rest_framework import serializers

from .models import Assessment

DISCLAIMER = 'Research output only. This is not a clinically validated result or a medical diagnosis.'


class AssessmentSerializer(serializers.ModelSerializer):
    disclaimer = serializers.SerializerMethodField()

    class Meta:
        model = Assessment
        fields = ['id', 'recording', 'patient', 'status', 'result_label', 'confidence', 'summary',
                  'doctor_notes', 'reviewed_by', 'reviewed_at', 'disclaimer', 'created_at', 'updated_at']
        # Only doctor_notes can be edited through the API; everything else is system-controlled.
        read_only_fields = [f for f in fields if f != 'doctor_notes']

    def get_disclaimer(self, obj):
        return DISCLAIMER
