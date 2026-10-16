from django.urls import path
from ml_inference.views import (
    ModelListView,
    ActiveModelView,
    PredictionListView,
    PredictionDetailView,
    RecordingPredictionsView,
    PredictRecordingView,
)

urlpatterns = [
    path('models/', ModelListView.as_view(), name='model-list'),
    path('models/active/', ActiveModelView.as_view(), name='model-active'),
    path('predictions/', PredictionListView.as_view(), name='prediction-list'),
    path('predictions/<uuid:pk>/', PredictionDetailView.as_view(), name='prediction-detail'),
    path('recordings/<uuid:recording_id>/predict/', PredictRecordingView.as_view(), name='recording-predict'),
    path('recordings/<uuid:recording_id>/predictions/', RecordingPredictionsView.as_view(), name='recording-prediction-history'),
]
