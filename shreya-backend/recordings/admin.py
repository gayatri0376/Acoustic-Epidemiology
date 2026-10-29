from django.contrib import admin

from .models import Recording


@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'original_filename', 'status', 'created_at')
    list_filter = ('status',)
