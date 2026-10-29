from django.contrib import admin

from .models import Assessment


@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'status', 'result_label', 'confidence', 'created_at')
    list_filter = ('status',)
