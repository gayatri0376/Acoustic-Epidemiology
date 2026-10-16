from django.contrib import admin
from assessments.models import Assessment


@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ('id', 'prediction', 'reviewer', 'review_status', 'reviewer_conclusion', 'diagnostic_agreement', 'created_at')
    list_filter = ('review_status', 'reviewer_conclusion', 'diagnostic_agreement')
    search_fields = ('prediction__id', 'reviewer__username', 'clinical_notes')
    readonly_fields = ('id', 'created_at', 'updated_at')
