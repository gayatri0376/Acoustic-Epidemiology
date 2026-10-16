from django.contrib import admin
from ml_inference.models import ModelVersion, Prediction


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('version', 'architecture', 'status', 'created_at')
    list_filter = ('status', 'architecture')
    search_fields = ('version', 'architecture')
    readonly_fields = ('id', 'created_at', 'updated_at')


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'recording', 'model_version', 'predicted_class', 'prediction_status', 'inference_duration_ms', 'created_at')
    list_filter = ('prediction_status', 'predicted_class')
    search_fields = ('recording__original_filename', 'predicted_class')
    readonly_fields = ('id', 'created_at')
