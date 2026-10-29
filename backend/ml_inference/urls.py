from django.urls import path
from ml_inference.views import (
    ModelListView,
    ActiveModelView,
    PredictionListView,
    PredictionDetailView,
    RecordingPredictionsView,
    PredictRecordingView,
    # Disease prediction views (Model B — separate from sound classification)
    ActiveDiseaseModelView,
    PredictDiseaseView,
    DiseasePredictionDetailView,
    RecordingDiseasePredictionsView,
)

urlpatterns = [
    # ---------------------------------------------------------------
    # Model registry
    # ---------------------------------------------------------------
    path('models/', ModelListView.as_view(), name='model-list'),
    path('models/active/', ActiveModelView.as_view(), name='model-active'),
    # Disease-specific model status (separate endpoint for Model B)
    path('models/disease/active/', ActiveDiseaseModelView.as_view(), name='disease-model-active'),

    # ---------------------------------------------------------------
    # Sound classification predictions (Model A — existing)
    # ---------------------------------------------------------------
    path('predictions/', PredictionListView.as_view(), name='prediction-list'),
    path('predictions/<uuid:pk>/', PredictionDetailView.as_view(), name='prediction-detail'),
    path('recordings/<uuid:recording_id>/predict/', PredictRecordingView.as_view(), name='recording-predict'),
    path('recordings/<uuid:recording_id>/predictions/', RecordingPredictionsView.as_view(), name='recording-prediction-history'),

    # ---------------------------------------------------------------
    # Disease predictions (Model B — new, fully separate)
    # ---------------------------------------------------------------
    path('recordings/<uuid:recording_id>/predict-disease/', PredictDiseaseView.as_view(), name='recording-predict-disease'),
    path('recordings/<uuid:recording_id>/disease-predictions/', RecordingDiseasePredictionsView.as_view(), name='recording-disease-prediction-history'),
    path('disease-predictions/<uuid:pk>/', DiseasePredictionDetailView.as_view(), name='disease-prediction-detail'),
]
