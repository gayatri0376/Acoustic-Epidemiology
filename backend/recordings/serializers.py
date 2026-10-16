from rest_framework import serializers
from recordings.models import Recording, RecordingStatus, AnatomicalLocation, PatientAgeGroup
from recordings.validators import inspect_and_validate_audio, sanitize_filename


class RecordingSerializer(serializers.ModelSerializer):
    uploader_username = serializers.CharField(source='uploader.username', read_only=True)
    file_url = serializers.CharField(read_only=True)
    anatomical_location_display = serializers.CharField(source='get_anatomical_location_display', read_only=True)
    patient_age_group_display = serializers.CharField(source='get_patient_age_group_display', read_only=True)

    class Meta:
        model = Recording
        fields = [
            'id',
            'original_filename',
            'file_format',
            'file_size_bytes',
            'duration_seconds',
            'sample_rate',
            'channels',
            'recording_status',
            'anatomical_location',
            'anatomical_location_display',
            'patient_age_group',
            'patient_age_group_display',
            'recording_device',
            'patient_pseudo_id',
            'uploader',
            'uploader_username',
            'file_url',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'original_filename',
            'file_format',
            'file_size_bytes',
            'duration_seconds',
            'sample_rate',
            'channels',
            'recording_status',
            'uploader',
            'uploader_username',
            'file_url',
            'created_at',
            'updated_at',
        ]


class RecordingUploadSerializer(serializers.ModelSerializer):
    audio_file = serializers.FileField(write_only=True, required=True)

    class Meta:
        model = Recording
        fields = [
            'audio_file',
            'anatomical_location',
            'patient_age_group',
            'recording_device',
            'patient_pseudo_id',
        ]

    def validate_audio_file(self, value):
        # Perform deep file inspection
        audio_meta = inspect_and_validate_audio(value)
        self.context['audio_meta'] = audio_meta
        return value

    def create(self, validated_data):
        audio_file = validated_data.pop('audio_file')
        audio_meta = self.context.get('audio_meta', {})
        request = self.context.get('request')
        uploader = request.user if request and request.user.is_authenticated else None

        clean_name = sanitize_filename(getattr(audio_file, 'name', 'recording.wav'))

        recording = Recording(
            original_filename=clean_name,
            file_format=audio_meta.get('format', 'wav'),
            file_size_bytes=audio_meta.get('file_size_bytes', audio_file.size),
            duration_seconds=audio_meta.get('duration_seconds'),
            sample_rate=audio_meta.get('sample_rate'),
            channels=audio_meta.get('channels', 1),
            recording_status=RecordingStatus.VALIDATED,
            uploader=uploader,
            **validated_data
        )
        recording.audio_file.save(clean_name, audio_file, save=True)
        return recording
