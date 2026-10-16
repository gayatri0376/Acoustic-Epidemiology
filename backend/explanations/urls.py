from django.urls import path
from explanations.views import (
    PredictionExplanationListCreateView,
    ExplanationDetailView,
)

urlpatterns = [
    path('predictions/<uuid:prediction_id>/explanations/', PredictionExplanationListCreateView.as_view(), name='prediction-explanations'),
    path('explanations/<uuid:pk>/', ExplanationDetailView.as_view(), name='explanation-detail'),
]
