from django.contrib import admin
from recordings.models import Recording


@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = (
        'id', 
        'original_filename', 
        'file_format', 
        'duration_seconds', 
        'sample_rate', 
        'recording_status', 
        'anatomical_location', 
        'uploader', 
        'created_at'
    )
    list_filter = ('recording_status', 'file_format', 'anatomical_location', 'patient_age_group')
    search_fields = ('original_filename', 'patient_pseudo_id', 'uploader__username')
    readonly_fields = (
        'id', 
        'file_size_bytes', 
        'duration_seconds', 
        'sample_rate', 
        'channels', 
        'created_at', 
        'updated_at'
    )
