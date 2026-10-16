from django.urls import path
from assessments.views import (
    PredictionAssessmentCreateView,
    AssessmentListView,
    AssessmentDetailView,
)

urlpatterns = [
    path('predictions/<uuid:prediction_id>/assessments/', PredictionAssessmentCreateView.as_view(), name='prediction-assessment-create'),
    path('assessments/', AssessmentListView.as_view(), name='assessment-list'),
    path('assessments/<uuid:pk>/', AssessmentDetailView.as_view(), name='assessment-detail'),
]
