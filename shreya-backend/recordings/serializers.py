from django.urls import reverse
from rest_framework import serializers

from .models import Recording
from .validators import CONTENT_TYPES, get_extension, validate_audio_file


class RecordingSerializer(serializers.ModelSerializer):
    audio_file = serializers.FileField(write_only=True, validators=[validate_audio_file])
    download_url = serializers.SerializerMethodField()
    assessment_id = serializers.SerializerMethodField()

    class Meta:
        model = Recording
        fields = ['id', 'patient', 'audio_file', 'original_filename', 'content_type', 'file_size', 'notes',
                  'recorded_at', 'status', 'assessment_id', 'download_url', 'created_at']
        read_only_fields = ['id', 'patient', 'original_filename', 'content_type', 'file_size', 'status', 'created_at']

    def get_download_url(self, obj):
        # Authenticated endpoint; the raw media folder is never exposed.
        return reverse('recording-file', args=[obj.pk])

    def get_assessment_id(self, obj):
        assessments = list(obj.assessments.all())  # prefetched in the view
        return assessments[0].pk if assessments else None

    def create(self, validated_data):
        upload = validated_data['audio_file']
        validated_data['original_filename'] = upload.name.replace('\\', '/').split('/')[-1][:255]
        validated_data['file_size'] = upload.size
        validated_data['content_type'] = CONTENT_TYPES[get_extension(upload.name)]
        return super().create(validated_data)
