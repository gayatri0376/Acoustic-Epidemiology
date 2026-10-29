from django.urls import path

from .views import AssessmentDetailView, AssessmentListView

urlpatterns = [
    path('', AssessmentListView.as_view(), name='assessment-list'),
    path('<int:pk>/', AssessmentDetailView.as_view(), name='assessment-detail'),
]
