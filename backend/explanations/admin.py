from django.contrib import admin
from explanations.models import Explanation


@admin.register(Explanation)
class ExplanationAdmin(admin.ModelAdmin):
    list_display = ('id', 'prediction', 'method', 'status', 'created_at')
    list_filter = ('method', 'status')
    search_fields = ('prediction__id', 'prediction__recording__original_filename')
    readonly_fields = ('id', 'created_at')
